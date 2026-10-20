# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar

from torch import Tensor
import torch

from ..module import TransformerModule

__all__ = ["Log"]


class Log(TransformerModule):
    name: ClassVar[str] = "log"

    def forward(self, t: Tensor, /, log_base: Tensor, *, inverse: bool = False) -> Tensor:
        if inverse:
            return t.mul(log_base).exp()

        # since log undefined for <= 0, need to mask to nan so we dont get infs
        return t.masked_fill(t <= 0, torch.nan).log().div(log_base)
