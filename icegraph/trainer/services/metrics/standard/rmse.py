# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

import torch
from torch import Tensor

from .mse import MSE

__all__ = ["RMSE"]


class RMSE(MSE):

    name: ClassVar[str] = "rmse"
    compatible = ["regression"]

    def _compute(self) -> float:
        if self.cache["n"].item() == 0:
            return float("nan")

        # calculate rmse
        rmse = torch.sqrt(self.cache["sse"] / self.cache["n"])

        # only now sync back to host
        return rmse.item()
