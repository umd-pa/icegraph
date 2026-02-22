# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from icegraph.types.transforms import TransformSpace, TransformSpec

from ..modules import BufferedDict

from .factory import TransformFactory
from .transform import Transform

__all__ = ["GroupTransform"]


class GroupTransform(Module):

    def __init__(self) -> None:
        super().__init__()

        # one transform for each space (except linear)
        spaces = TransformSpace.non_linear()
        space_names = [space.name for space in spaces]

        # build empty mapping
        self._mapping: BufferedDict = BufferedDict.from_keys(space_names, dtype=torch.long)

        # transforms needs to be constant shape, build each empty
        self._transforms: ModuleDict = ModuleDict({
            space.name: TransformFactory.create(space.value) for space in spaces
        })

    @staticmethod
    def _parse_specs(specs: list[TransformSpec]) -> dict[TransformSpace, dict[str, list[int]]]:
        # build spec dict by grouping
        parsed: dict[TransformSpace, dict[str, Any]] = {}

        for i, spec in enumerate(specs):
            if spec.space == TransformSpace.LINEAR:
                # skip registration if linear
                continue

            # initialize if not present
            parsed.setdefault(spec.space, {})

            parsed[spec.space].setdefault("cols", []).append(i)
            parsed[spec.space].setdefault("base", []).append(spec.base)

        return parsed

    def configure_from_specs(self, specs: list[TransformSpec]) -> None:
        # clear all mappings first
        for space_name in self._mapping:
            self._mapping[space_name] = torch.empty(0, dtype=torch.long)

        for space, params in self._parse_specs(specs).items():
            # extract mapping, convert to tensor
            self._mapping[space.name] = torch.tensor(params["cols"], dtype=torch.long)

            # build the transform
            tensor_params: dict[str, Tensor] = {}
            if "base" in params:
                tensor_params["base"] = torch.tensor(params["base"], dtype=torch.float32)

            # configure each transform
            self.get_transform(space.name).configure(**tensor_params)

    def get_transform(self, space_name: str) -> Transform:
        return cast(Transform, self._transforms[space_name])

    @torch.no_grad()
    def forward(self, t: Tensor, *, inverse: bool = False) -> Tensor:
        for space_name, cols in self._mapping.items():
            if cols.numel() == 0:
                # skip for empty cols
                continue

            # filter to cols
            selection = t.index_select(dim=-1, index=cols)

            # run vectorized transform
            selection = self.get_transform(space_name).forward(selection, inverse=inverse)

            # copy changes back to original tensor
            t.index_copy_(dim=-1, index=cols, source=selection)

        return t
