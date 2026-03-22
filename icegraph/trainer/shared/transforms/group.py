# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from icegraph.types.transforms import TransformSpace

from ..modules import BufferedDict

from .factory import TransformFactory
from .transform import Transform
from .types import GroupTransformSpec, TransformSpec

__all__ = ["GroupTransform"]


class GroupTransform(Module):

    def __init__(self) -> None:
        super().__init__()

        # one transform for each space (except linear)
        spaces = TransformSpace.non_linear()

        # build empty mapping
        self._mapping: BufferedDict = BufferedDict.from_keys([space.name for space in spaces], dtype=torch.long)

        # transforms needs to be constant shape, build each empty
        self._transforms: ModuleDict = ModuleDict({
            space.name: TransformFactory.create(space.value) for space in spaces
        })

    def configure_from_spec(self, spec: GroupTransformSpec | TransformSpec) -> None:
        # clear all mappings first
        for space_name in self._mapping:
            self._mapping[space_name] = torch.empty(0, dtype=torch.long)

        # normalize spec input
        spec = spec if isinstance(spec, GroupTransformSpec) else GroupTransformSpec([spec])

        for space, bases, cols in spec.groups:
            if space is TransformSpace.LINEAR:
                # no transform for linear
                continue

            # extract mapping, convert to tensor
            self._mapping[space.name] = torch.tensor(cols, dtype=torch.long)

            # configure each transform
            self.get_transform(space.name).configure(base=torch.tensor(bases, dtype=torch.float32))

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
