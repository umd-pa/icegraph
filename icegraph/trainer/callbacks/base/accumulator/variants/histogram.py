# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..accumulator import Accumulator

__all__ = ["HistogramAccumulator"]


@dataclass
class HistogramAccumulator(Accumulator):
    _data: Tensor | None = field(init=False, default=None)

    def update(self, data: Tensor, /) -> None:
        if self._data is None:
            self._data = torch.zeros(data.shape, device=data.device, dtype=torch.int64)

        if data.ndim != self._data.ndim:
            raise ValueError(
                f"data must have {self._data.ndim} dimensions, got {data.ndim}"
            )

        if tuple(data.shape) != tuple(self.data.shape):
            raise ValueError(
                f"hist must have shape {tuple(self.data.shape)}, got {tuple(data.shape)}"
            )

        if data.numel() == 0:
            return

        # ensure on accelerator
        data = data.to(device=self.data.device, dtype=torch.int64)

        # accumulate dense tensor
        self._data.add_(data)

    def is_empty(self) -> bool:
        return self._data is None

    @property
    def data(self) -> Tensor:
        if self._data is None:
            raise ValueError("No data found in accumulator.")
        return self._data

    @data.setter
    def data(self, data: Tensor) -> None:
        if self._data is not None and self._data.shape != data.shape:
            raise ValueError("New data tensor must have same shape as existing tensor.")

        self._data = data

    def reset(self) -> None:
        self._data = None

    def to(self, device: torch.device | str) -> Self:
        self._data.to(torch.device(device))
        return self
