# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace

from ..transform import Transform

__all__ = ["Log"]


class Log(Transform):
    name: TransformSpace = TransformSpace.LOG

    def __init__(self) -> None:
        super().__init__()

        # register empty buffer
        self.register_buffer("log_base", torch.empty(0, dtype=torch.float32))

    def configure(self, **kwargs) -> None:
        base: Tensor = kwargs.get("base", torch.empty(0))

        if base.numel() == 0:
            # dont do anything if no new base passed
            return

        if (base <= 0).any() or (base == 1).any():
            raise ValueError("All bases must be positive and not equal to 1.")

        # override buffer
        self._buffers["log_base"] = torch.log(base)

    def transform(self, t: Tensor) -> Tensor:
        self._check(t, "log_base")
        t.log_().div_(self.log_base)
        return t

    def inverse_transform(self, t: Tensor) -> Tensor:
        self._check(t, "log_base")
        t.mul_(self.log_base).exp_()
        return t
