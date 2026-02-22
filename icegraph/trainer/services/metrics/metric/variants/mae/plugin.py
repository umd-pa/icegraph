# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, ClassVar, Any

import torch
from torch import Tensor

from ...metric import Metric

from .config import Config

__all__ = ["MAE"]


class MAE(Metric[Config]):
    name: ClassVar[str] = "mae"
    version: ClassVar[int] = 1

    compatible: ClassVar[tuple[str, ...]] = ("regression", )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _update(self, out: Tensor, target: Tensor) -> None:
        # grab values required, perform ops on GPU
        ae = torch.sum(torch.abs(out - target))
        n = out.numel()

        # verify cache
        if self._cache.get("ae") is None or self._cache.get("n") is None:
            # Initialize sum and N on the device of the incoming data (loss)
            self._cache["ae"]   = torch.zeros_like(ae)
            self._cache["n"]    = torch.tensor(0, dtype=torch.long, device=ae.device)

        # accumulate
        self._cache["ae"]   += ae
        self._cache["n"]    += n

    def _compute(self) -> float:
        if self._cache["n"].item() == 0:
            return float("nan")

        # calculate mae
        mae: Tensor = self._cache["ae"] / self._cache["n"]

        # only now sync back to host
        return mae.item()

    def merge(self, other: Self) -> None:
        self._cache["ae"]   += other.cache["ae"]
        self._cache["n"]    += other.cache["n"]
