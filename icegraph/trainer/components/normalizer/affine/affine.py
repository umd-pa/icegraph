# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable, Any, cast
from abc import abstractmethod
from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import ModuleDict

# local package
from icegraph.statistics import StatisticService
from icegraph.types.transforms import TransformSpec, TransformSpace
from icegraph.types.data import ModelInputRole, Split
from icegraph.trainer.shared import GroupTransform, BufferedDict

from ..normalizer import Normalizer

from .config import AffineNormalizerConfig

__all__ = ["AffineNormalizer"]


class AffineNormalizer(Normalizer[AffineNormalizerConfig]):
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
    def validate_config(cls, config: dict[str, Any]) -> AffineNormalizerConfig:
        return AffineNormalizerConfig(**config)

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

            # get columns from stat service
            columns = stats.columns

            # subtract columns as set from invalid_columns
            invalid_columns -= set(columns)

            # build specs
            specs = self._build_spec_list(columns)

            # configure the transform for this role
            self.transformer(role).configure_from_specs(specs)

            # eager compute the scale and offset from subclass logic
            # update buffered dicts
            self._scale[role.name]  = self._resolve(self._build_scale, stats, specs)
            self._offset[role.name] = self._resolve(self._build_offset, stats, specs)

        if invalid_columns:
            raise ValueError(f"Got invalid columns in config: {invalid_columns}")

    def _build_spec_list(self, columns: list[str]) -> list[TransformSpec]:
        # load user config
        transform_config = self.params.get("transforms", {})

        # build new spec list, maintain correct order
        specs: list[TransformSpec] = []
        for column in columns:
            column_config = transform_config.get(column, {}).copy()

            # default to linear if no transform config provided
            if not column_config:
                specs.append(TransformSpec(TransformSpace.LINEAR))
                continue

            # obtain selected space (required)
            space_value = column_config.pop("space", None)
            try:
                space = TransformSpace(space_value)
            except ValueError:
                raise ValueError(f"{type(self).__name__}: unknown space '{space_value}' for column '{column}'")

            # other configurations passed directly to the spec
            specs.append(TransformSpec(space, **column_config))

        return specs

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
            specs: list[TransformSpec]
    ) -> Tensor:
        # count of columns
        spec_count = len(specs)

        # group indices by (space, base)
        groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        for i, spec in enumerate(specs):
            groups[(spec.space, spec.base)].append(i)

        # init empty tensor on device
        out = torch.empty(spec_count, dtype=torch.float32)

        # for each unique set of (space, base), use build method to compute
        for (space, base), indices in groups.items():
            # obtain tensor from subclass logic
            tensor = build(stats, space, base)

            # verify tensor dim and shape are correct
            if tensor.ndim != 1 or tensor.shape[0] != spec_count:
                raise ValueError(
                    f"build({space=}, {base=}) returned shape {tuple(tensor.shape)}, expected ({spec_count},)"
                )

            # grab values at relevant indices
            idx = torch.tensor(indices, dtype=torch.long)
            out[idx] = tensor[idx]

        return out

    def get_offset(self, role: ModelInputRole) -> Tensor:
        return self._offset[role.name]

    def get_scale(self, role: ModelInputRole) -> Tensor:
        return self._scale[role.name]

    def get_transform(self, role: ModelInputRole) -> GroupTransform:
        return cast(GroupTransform, self._transforms[role.name])

    def _affine(self, tensor: Tensor, role: ModelInputRole, *, inverse: bool = False) -> Tensor:
        # get scale and offset
        scale = self.get_scale(role)
        offset = self.get_offset(role)

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