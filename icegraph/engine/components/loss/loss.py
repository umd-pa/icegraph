# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC

from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ..component import Component

from .types import LossContext

__all__ = ["LossFunction"]


C = TypeVar("C")


class LossFunction(Component[C, LossContext], ABC):

    @final
    def forward(self, out: SegmentedTensor, target: SegmentedTensor, /) -> Tensor:
        """Forward pass through the loss function."""
        loss = self.loss(out, target)

        # internal validation
        if loss.ndim != 0:
            raise ValueError(
                f"Loss must reduce to a scalar; got shape {tuple(loss.shape)}."
            )

        # run contract validator
        self._ctx.contract.forward_validator(loss, self._ctx.debug)

        return loss

    @abstractmethod
    def loss(self, out: SegmentedTensor, target: SegmentedTensor, /) -> Tensor:
        ...
