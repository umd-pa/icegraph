# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self

import torch
from torch import Tensor

from ..metric import Metric

__all__ = ["MAE"]


class MAE(Metric):

    name = "mae"
    compatible = ["regression"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initialize required caches
        self.cache["ae"]  = None
        self.cache["n"]   = None

    def _update(self, out: Tensor, target: Tensor) -> None:
        # grab values required, perform ops on GPU
        ae = torch.sum(torch.abs(out - target))
        n = out.numel()

        # verify cache
        if self.cache["ae"] is None:
            # Initialize sum and N on the device of the incoming data (loss)
            self.cache["ae"]: Tensor = torch.zeros_like(ae)
            self.cache["n"]:  Tensor = torch.tensor(0, dtype=torch.long, device=ae.device)

        # accumulate
        self.cache["ae"] += ae
        self.cache["n"]  += n

    def _compute(self) -> float:
        if self.cache["n"].item() == 0:
            return float("nan")

        # calculate mae
        mae: Tensor = self.cache["ae"] / self.cache["n"]

        # only now sync back to host
        return mae.item()

    def merge(self, other: Self) -> None:
        self.cache["ae"]  += other.cache["ae"]
        self.cache["n"]   += other.cache["n"]
