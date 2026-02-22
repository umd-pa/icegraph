# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor

from ...metric import Metric

from .config import Config

__all__ = ["MSE"]


class MSE(Metric[Config]):
    name: ClassVar[str] = "mse"
    version: ClassVar[int] = 1

    compatible: ClassVar[tuple[str, ...]] = ("regression",)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _update(self, out: Tensor, target: Tensor) -> None:
        # grab values required, perform ops on GPU
        sse = torch.sum((out - target) ** 2)
        n = out.numel()

        # verify cache
        if self._cache.get("sse") is None or self._cache.get("n") is None:
            # Initialize sum and N on the device of the incoming data (loss)
            self._cache["sse"]  = torch.zeros_like(sse)
            self._cache["n"]    = torch.tensor(0, dtype=torch.long, device=sse.device)

        # accumulate
        self._cache["sse"] += sse
        self._cache["n"]   += n

    def _compute(self) -> float:
        if self._cache["n"].item() == 0:
            return float("nan")

        # calculate mse
        mse: Tensor = self._cache["sse"] / self._cache["n"]

        # only now sync back to host
        return mse.item()

    def merge(self, other: MSE) -> None:
        self._cache["sse"]  += other._cache["sse"]
        self._cache["n"]    += other._cache["n"]
