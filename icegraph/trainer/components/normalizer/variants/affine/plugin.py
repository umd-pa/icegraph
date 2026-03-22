# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable, Any, cast
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch.nn import ModuleDict

# local package
from icegraph.statistics import StatisticService
from icegraph.types.transforms import TransformSpace
from icegraph.types.data import ModelInputRole, Split
from icegraph.trainer.shared import GroupTransform, BufferedDict, GroupTransformSpec

from icegraph.trainer.components.normalizer import Normalizer

from .config import Config

__all__ = ["AffineNormalizer"]


class AffineNormalizer(Normalizer[Config], ABC):
    _transforms:    ModuleDict
    _scale:         BufferedDict
    _offset:        BufferedDict

    def build(self) -> None:
        """Initialize the normalizer."""
        # get roles
        roles = ModelInputRole.names()

        # transformer cache
        self._transforms = ModuleDict({role: GroupTransform() for role in roles})

        # cache for scale and offset tensors
        # these must be registered buffers
        self._scale = BufferedDict.from_keys(roles)
        self._offset = BufferedDict.from_keys(roles)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def on_attach(self) -> None:
        # load user config
        transform_config = self.config.transforms

        # check for config columns that are not in dataset columns
        invalid_columns: set[str] = set(transform_config.keys())

        # get data service from manager
        data = self._ctx.services.require("data", required_by=Normalizer)

        for role in ModelInputRole.all():
            # get stats, only want training stats; we don't care about val or test
            stats = data.stats(Split.TRAIN, role)

            # get columns from data service
            columns = data.columns(role)
            stats.filter_to(columns)

            # subtract columns as set from invalid_columns
            invalid_columns -= set(columns)

            # build specs
            spec = GroupTransformSpec.from_config(columns, self.config.transforms)

            # configure the transform for this role
            self.get_transform(role).configure_from_spec(spec)

            # eager compute the scale and offset from subclass logic
            # update buffered dicts
            self._scale[role.name]  = self._resolve(self._build_scale, stats, spec)
            self._offset[role.name] = self._resolve(self._build_offset, stats, spec)

        if invalid_columns:
            raise ValueError(f"Got invalid columns in config: {invalid_columns}")

    @abstractmethod
    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        ...

    @abstractmethod
    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        ...

    @staticmethod
    def _resolve(
            build: Callable[[StatisticService, TransformSpace, int], Tensor],
            stats: StatisticService,
            spec: GroupTransformSpec
    ) -> Tensor:
        # number of columns defined by the spec
        spec_count = len(spec)

        # output tensor storing resolved values for each column
        # safe because we enforce that every column is written exactly once
        out = torch.empty(spec_count, dtype=torch.float32)

        # track which columns have been assigned
        # used to detect duplicates and missing columns
        assigned = torch.zeros(spec_count, dtype=torch.bool)

        # cache build results for each (space, base) pair
        # avoids recomputing identical transforms
        cache: dict[tuple[TransformSpace, int], Tensor] = {}

        # iterate through transform groups
        for space, bases, cols in spec.groups:
            for base, col in zip(bases, cols, strict=True):
                key = (space, base)

                # reuse cached build result if available
                tensor = cache.get(key)
                if tensor is None:
                    tensor = build(stats, space, base)

                    # ensure returned tensor covers all columns
                    if tensor.ndim != 1 or tensor.shape[0] != spec_count:
                        raise ValueError(
                            f"build({space=}, {base=}) returned shape {tuple(tensor.shape)}, "
                            f"expected ({spec_count},)"
                        )

                    # reject tensors containing nan/inf
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"build({space=}, {base=}) returned non-finite values")

                    cache[key] = tensor

                # detect duplicate column assignment
                if assigned[col]:
                    raise ValueError(f"Column {col} assigned more than once")

                # assign value for this column
                out[col] = tensor[col]
                assigned[col] = True

        # ensure every column in the spec was assigned
        if not assigned.all():
            missing = (~assigned).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"GroupTransformSpec did not cover all columns, missing: {missing}")

        # final safety check
        if not torch.isfinite(out).all():
            raise ValueError("Resolved affine parameters contain non-finite values")

        return out

    def get_transform(self, role: ModelInputRole) -> GroupTransform:
        return cast(GroupTransform, self._transforms[role.name])

    def _affine(self, tensor: Tensor, role: ModelInputRole, *, inverse: bool = False) -> Tensor:
        # get scale and offset
        scale = self._scale[role.name]
        offset = self._offset[role.name]

        # perform affine transform
        if inverse:
            return tensor.div_(scale).add_(offset)
        return tensor.add_(-offset).mul_(scale)

    @torch.no_grad()
    def forward(self, t: Tensor, /, role: ModelInputRole, *, inverse: bool = False) -> Tensor:
        """Forward pass through the normalizer."""
        # raise if the input is not a tensor
        if not isinstance(t, Tensor):
            raise TypeError(f"Unsupported normalizer input type {type(t)}")

        # ensure float dtype
        if not torch.is_floating_point(t):
            t = t.to(torch.float32)

        # unsqueeze 1D tensors
        if unsqueezed := t.ndim == 1:
            t = t.unsqueeze(1)

        # run normalization
        transform = self.get_transform(role)

        ops = [
            lambda x: transform.forward(x, inverse=inverse),
            lambda x: self._affine(x, role, inverse=inverse),
        ]

        # invert order if inverse
        if inverse:
            ops = reversed(ops)

        # apply ops sequentially
        for op in ops:
            t = op(t)

        # squeeze if unsqueezed
        return t.squeeze(1) if unsqueezed else t