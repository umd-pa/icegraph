# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC
from functools import lru_cache
import dataclasses

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor
from icegraph.common.data import ColumnarRole, DataRole
from icegraph.common.engine import ComponentKind

from ..component import Component

from .types import TransformerSpec

__all__ = ["Transformer"]


C = TypeVar("C")


class Transformer(Component[C], ABC):

    @property
    def norm_targets(self) -> bool:
        # this pulls from the normalizer
        normalizer = self._ctx.components.require(ComponentKind.NORMALIZER, required_by=type(self))
        return normalizer.norm_targets  # type: ignore

    @final
    @torch.no_grad()
    def forward(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool = False) -> SegmentedTensor:
        """Forward pass through the transformer."""
        # skip if norm is not required for targets
        # if norm is not required, no transform should take place either
        if role == DataRole.TARGETS:
            if not self.norm_targets:
                return t

        out = self.transform(t, role, inverse=inverse)

        # internal validation
        if out.shape != t.data.shape:
            raise ValueError(
                f"Transformer is a value map only, tensors cannot be reshaped. "
                f"Expected shape {t.data.shape}, got {out.shape}"
            )

        # only check in debug mode, since this forces a sync
        if self._ctx.debug:
            if not torch.isfinite(out).all():
                raise ValueError(
                    f"Transformer produced non-finite values (inf/nan); "
                    f"check domain of inputs to the {type(self).__name__} mapping "
                    f"(e.g. log of non-positive values)."
                )

        # run contract validator
        self._run_forward_validator(out)

        return dataclasses.replace(t, data=out)

    @abstractmethod
    def transform(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
        ...

    @lru_cache(maxsize=None)
    def spec_list(self, role: ColumnarRole) -> list[TransformerSpec]:
        return self._build_spec_list(role)

    @abstractmethod
    def _build_spec_list(self, role: ColumnarRole) -> list[TransformerSpec]:
        ...
