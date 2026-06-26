# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Iterable

from torch.optim import AdamW as _AdamW
from torch.nn import Parameter

from icegraph.common.engine import ComponentKind
from icegraph.engine.components.optimizer import Optimizer

from .config import Config

__all__ = ["AdamW"]


class AdamW(Optimizer[Config]):
    name: ClassVar[str] = "adam"
    version: ClassVar[int] = 1

    _opt: _AdamW | None

    def build(self) -> None:
        self._opt = None

    def on_attach(self) -> None:
        model = self._ctx.components.require(ComponentKind.MODEL, required_by=type(self))
        self._opt = self._build_optimizer(model.parameters())

    def _build_optimizer(self, params: Iterable[Parameter]) -> _AdamW:
        return _AdamW(
            params,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def step(self) -> None:
        if self._opt is None:
            raise RuntimeError("Optimizer not built")
        self._opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._opt is None:
            raise RuntimeError("Optimizer not built")
        self._opt.zero_grad(set_to_none=set_to_none)
