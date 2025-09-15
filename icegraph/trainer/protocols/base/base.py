# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Optional, Dict, Any, final, Tuple
from abc import ABC, abstractmethod

from torch import nn
import torch


class TaskStrategy(ABC):
    task: str

    def __init__(self, **kwargs) -> None:
        self._enforced_reduction = "sum"
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        for attr in ["task"]:
            if getattr(cls, attr, None) is None:
                raise NotImplementedError(f"Subclasses of 'TaskStrategy' must implement the '{attr}' class attribute.")

    @abstractmethod
    def loss_function(self) -> nn.Module: ...

    @abstractmethod
    def post_init_check(self, model: torch.nn.Module) -> None: ...

    @abstractmethod
    def adapt_targets(self, batch: torch.Batch, out: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def make_metrics(self) -> Metrics: ...

    @abstractmethod
    def filter_eval(
            self,
            out: torch.Tensor,
            target: torch.Tensor
    ) -> Tuple: ...


class Metrics(ABC):
    _compute_cache: Optional[Dict[str, float]] = None

    @final
    def update(
            self,
            out: torch.Tensor,
            target: torch.Tensor,
            loss: torch.Tensor, *,
            mask: Optional[torch.Tensor] = None
    ) -> None:
        self._update(out, target, loss, mask=mask)
        self._compute_cache = None

    @abstractmethod
    def _update(
            self,
            out: torch.Tensor,
            target: torch.Tensor,
            loss: torch.Tensor, *,
            mask: Optional[torch.Tensor] = None
    ) -> None: ...

    @final
    def compute(self) -> Dict[str, float]:
        if self._compute_cache is None:
            self._compute_cache = self._compute()
        return self._compute_cache

    @abstractmethod
    def _compute(self) -> Dict[str, float]: ...

    @abstractmethod
    def merge_(self, other: Metrics) -> None: ...
