# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from torch.nn import Module

from icegraph.engine.components.model import Model

__all__ = ["_NoDDP"]


class _NoDDP(Module):
    def __init__(self, module: Model[Any]) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def close(self) -> None:
        self.module.close()