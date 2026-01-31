# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self

import torch
from torch import Tensor

from ..metric import Metric

__all__ = ["MSE"]


class MSE(Metric):

    name = "mse"
    compatible = ["regression"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initialize required caches
        self.cache["sse"] = None
        self.cache["n"]   = None

    def _update(self, out: Tensor, target: Tensor) -> None:
        # grab values required, perform ops on GPU
        sse = torch.sum((out - target) ** 2)
        n = out.numel()

        # verify cache
        if self.cache["sse"] is None:
            # Initialize sum and N on the device of the incoming data (loss)
            self.cache["sse"]:    Tensor = torch.zeros_like(sse)
            self.cache["n"]:      Tensor = torch.tensor(0, dtype=torch.long, device=sse.device)

        # accumulate
        self.cache["sse"] += sse
        self.cache["n"]   += n

    def _compute(self) -> float:
        if self.cache["n"].item() == 0:
            return float("nan")

        # calculate mse
        mse: Tensor = self.cache["sse"] / self.cache["n"]

        # only now sync back to host
        return mse.item()

    def merge(self, other: Self) -> None:
        self.cache["sse"] += other.cache["sse"]
        self.cache["n"]   += other.cache["n"]
