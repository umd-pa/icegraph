# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC
import dataclasses

import torch
from torch import Tensor

from icegraph.common.data import ColumnarRole
from icegraph.common.tensors import SegmentedTensor

from ..component import Component

from .types import NormalizerContext

__all__ = ["Normalizer"]


C = TypeVar("C")


class Normalizer(Component[C, NormalizerContext], ABC):

    @final
    @torch.no_grad()
    def forward(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool = False) -> SegmentedTensor:
        """Forward pass through the normalizer."""
        out = self.normalize(t, role, inverse=inverse)

        # internal validation
        if out.shape != t.data.shape:
            raise ValueError(
                f"Normalizer is a value map only, tensors cannot be reshaped. "
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
        self._ctx.contract.forward_validator(out, self._ctx.debug)

        return dataclasses.replace(t, data=out)

    @abstractmethod
    def normalize(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
        ...
