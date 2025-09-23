# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field
from typing import Optional, Dict, Self

import torch

from .base import Metrics
from icegraph.types import ComputedMetrics

__all__ = ["RegressionMetrics", "MulticlassMetrics"]


@dataclass
class RegressionMetrics(Metrics):
    sse_sum: float = field(default=0.0, init=False)
    n_elems: int = field(default=0, init=False)
    loss_sum: float = field(default=0.0, init=False)

    def _update(
            self,
            out: torch.Tensor,
            target: torch.Tensor,
            loss: torch.Tensor, *,
            mask: Optional[torch.Tensor] = None
    ) -> None:
        if mask is not None:
            out, target = out[mask], target[mask]

        self.loss_sum += float(loss.item())
        self.sse_sum += float(loss.item())
        self.n_elems += out.numel()

    def _compute(self) -> ComputedMetrics:
        if self.n_elems == 0:
            return {"loss": float("nan"), "rmse": float("nan")}
        mse = self.loss_sum / self.n_elems
        return {"loss:mse": mse, "rmse": mse ** 0.5}

    def merge_(self, other: Self) -> None:
        self.sse_sum += other.sse_sum
        self.loss_sum += other.loss_sum
        self.n_elems += other.n_elems


@dataclass
class MulticlassMetrics(Metrics):
    loss_sum: float = field(default=0.0, init=False)
    n_samples: int = field(default=0, init=False)
    correct: int = field(default=0, init=False)
    topk_correct: int = field(default=0, init=False)
    k: int = field(default=3, kw_only=True)

    def _update(self, out, target, loss, *, mask=None) -> None:
        if mask is not None:
            out, target = out[mask], target[mask]
            if out.numel() == 0: return

        self.loss_sum += float(loss.item())
        self.n_samples += int(target.numel())
        pred1 = out.argmax(dim=1)
        self.correct += int((pred1 == target).sum().item())

        if self.k and out.size(1) >= self.k:
            topk = out.topk(self.k, dim=1).indices
            self.topk_correct += int((topk == target.unsqueeze(1)).any(dim=1).sum().item())

    def _compute(self) -> ComputedMetrics:
        if self.n_samples == 0:
            return {"loss:mse": float("nan"), "acc": float("nan"), f"top{self.k}_acc": float("nan")}

        return {
            "loss:mse": self.loss_sum / self.n_samples,
            "acc": self.correct / self.n_samples,
            f"top{self.k}_acc": self.topk_correct / self.n_samples if self.k else float("nan")
        }

    def merge_(self, other: Self) -> None:
        self.loss_sum += other.loss_sum
        self.n_samples += other.n_samples
        self.correct += other.correct
        self.topk_correct += other.topk_correct
