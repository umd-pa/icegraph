# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Iterable

from torch.optim import SGD as _SGD
from torch.nn import Parameter

from icegraph.common.engine import ComponentKind
from icegraph.engine.components.optimizer import Optimizer

from .config import Config

__all__ = ["SGD"]


class SGD(Optimizer[Config]):
    name: ClassVar[str] = "sgd"
    version: ClassVar[int] = 1

    _opt: _SGD | None

    def build(self) -> None:
        self._opt = None

    def on_attach(self) -> None:
        model = self._ctx.components.require(ComponentKind.MODEL, required_by=type(self))
        self._opt = self._build_optimizer(model.parameters())

    def _build_optimizer(self, params: Iterable[Parameter]) -> _SGD:
        return _SGD(
            params,
            lr=self.config.lr,
            momentum=self.config.momentum,
            dampening=self.config.dampening,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.nesterov,
            maximize=self.config.maximize,
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
