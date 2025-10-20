# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field
from typing import Optional, Self

import torch

from .base import Metrics
from icegraph.types import ComputedMetrics

__all__ = ["RegressionMetrics", "MulticlassMetrics"]


@dataclass
class RegressionMetrics(Metrics):
    sse_sum: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0), init=False)
    n_elems: int = field(default=0, init=False)

    def _update(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor, *,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        if mask is not None:
            out, target = out[mask], target[mask]

        # accumulate on the same device as loss, no sync
        self.sse_sum = self.sse_sum.to(loss.device)
        self.sse_sum += loss.detach() * out.numel()
        self.n_elems += out.numel()

    def _compute(self) -> ComputedMetrics:
        if self.n_elems == 0:
            return {"loss": float("nan"), "rmse": float("nan")}
        # only now sync back to host
        mse = (self.sse_sum / self.n_elems).item()
        return {"loss:mse": mse, "rmse": mse ** 0.5}

    def merge_(self, other: Self) -> None:
        self.sse_sum += other.sse_sum
        self.n_elems += other.n_elems


@dataclass
class MulticlassMetrics(Metrics):
    # tensor accumulators
    loss_sum: torch.Tensor = field(default=None, init=False, repr=False)
    correct:  torch.Tensor = field(default=None, init=False, repr=False)
    topk_correct: torch.Tensor = field(default=None, init=False, repr=False)

    # stays on CPU, counting doesnt sync
    n_samples: int = field(default=0, init=False)

    k: int = field(default=3, kw_only=True)

    def _ensure_inited(self, device: torch.device):
        if self.loss_sum is None:
            self.loss_sum   = torch.tensor(0.0, device=device)
            self.correct    = torch.tensor(0,   device=device, dtype=torch.long)
            self.topk_correct = torch.tensor(0, device=device, dtype=torch.long)

    def _update(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor, *,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        if mask is not None:
            out, target = out[mask], target[mask]
            if out.numel() == 0:
                return

        self._ensure_inited(out.device)

        # Accumulate a sample-weighted loss on device so averaging is correct even with uneven batches
        bs = target.numel()
        self.loss_sum = self.loss_sum + loss.detach() * bs
        self.n_samples += int(bs)  # no sync

        # top-1 correct
        pred1 = out.argmax(dim=1)
        self.correct = self.correct + (pred1 == target).sum().detach()

        # top-k correct
        if self.k and out.size(1) >= self.k:
            topk = out.topk(self.k, dim=1).indices
            self.topk_correct = self.topk_correct + (topk == target.unsqueeze(1)).any(dim=1).sum().detach()

    def _compute(self) -> ComputedMetrics:
        if self.n_samples == 0:
            return {
                "loss:mse": float("nan"),
                "acc": float("nan"),
                f"top{self.k}_acc": float("nan")
            }

        # Single sync point per epoch
        loss_mean = (self.loss_sum / self.n_samples).item()
        acc       = (self.correct.to(torch.float32) / self.n_samples).item()
        topk_acc  = (self.topk_correct.to(torch.float32) / self.n_samples).item() if self.k else float("nan")

        return {"loss:mse": loss_mean, "acc": acc, f"top{self.k}_acc": topk_acc}

    def merge_(self, other: Self) -> None:
        # verify tensors exist and live on the same device
        if self.loss_sum is None and other.loss_sum is not None:
            self._ensure_inited(other.loss_sum.device)
        if other.loss_sum is not None:
            self.loss_sum    = self.loss_sum + other.loss_sum.to(self.loss_sum.device)
            self.correct     = self.correct + other.correct.to(self.correct.device)
            self.topk_correct = self.topk_correct + other.topk_correct.to(self.topk_correct.device)

        self.n_samples += other.n_samples
