# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, TYPE_CHECKING, List, Set

import torch
from torch_geometric.data import Batch

from icegraph.console import Console
from icegraph.utils import Statistics
from .base import NormCallback

if TYPE_CHECKING:
    from .. import Trainer
else:
    Trainer = None

__all__ = ["MinMaxNormalizer"]


class MinMaxNormalizer(NormCallback):
    """
    Normalization callback for scaling input features and target labels to the [0, 1] range
    using precomputed minimum and maximum statistics. Supports optional log-scaling of
    selected target labels prior to normalization.
    """

    def __init__(self) -> None:
        """
        Initialize the MinMaxNormalizer with empty offsets/scales for features and labels.
        """
        # call to super
        super().__init__()

        self._x_off:        Optional[torch.Tensor] = None
        self._x_scale:      Optional[torch.Tensor] = None

        self._y_off:        Optional[torch.Tensor] = None
        self._y_scale:      Optional[torch.Tensor] = None
        self._y_logmask:    Optional[torch.Tensor] = None

        self._on_device:    bool = False

    def on_init(self, trainer) -> None:
        # call to super
        super().on_init(trainer)

        self._set_minmax_from_stats(
            f_stats=self.f_stats,
            t_stats=self.t_stats,
            target_labels=trainer.target_labels,
            log_labels=set(trainer.apply_log_scaling or []),  # names to log-scale
        )

    def _set_minmax_from_stats(
        self,
        *,
        f_stats: Statistics,
        t_stats: Statistics,
        target_labels: List[str],
        log_labels: Set[str],
        eps: float = 1e-8,
    ) -> None:
        """
        Configure min/max-based normalization parameters from dataset statistics.

        Args:
            f_stats (Statistics): Feature statistics containing `.min` and `.max` values for each feature column.
            t_stats (Statistics): Target (label) statistics containing `.min` and `.max` values for each label column.
            target_labels (List[str]): Ordered list of target label names to normalize.
            log_labels (Set[str]): Subset of `target_labels` to apply log-scaling (log1p) to before normalization.
            eps (float): Minimum value for scaling denominators to avoid division by zero. Defaults to 1e-8.
        """
        # Features
        f_min = torch.tensor(f_stats.min, dtype=torch.float32)
        f_max = torch.tensor(f_stats.max, dtype=torch.float32)
        self._x_off   = f_min
        self._x_scale = (f_max - f_min).clamp_min(eps).reciprocal()

        # Truth: select labels in target order; build log mask for those
        if target_labels:
            idx = [t_stats.columns.index(name) for name in target_labels]
            t_min = torch.tensor(t_stats.min, dtype=torch.float32)[idx]
            t_max = torch.tensor(t_stats.max, dtype=torch.float32)[idx]

            logmask = torch.tensor([name in log_labels for name in target_labels], dtype=torch.bool)

            # Validate: disable log for labels with negative mins
            if torch.any(logmask & (t_min < 0)):
                bad = [name for name, m in zip(target_labels, (logmask & (t_min < 0)).tolist()) if m]
                Console.out(f"Warning: negative values in log-scaled labels {bad}; "
                            f"disabling log for those.", severity=2)
                # turn off log for those problematic labels
                for i, name in enumerate(target_labels):
                    if name in bad:
                        logmask[i] = False

            # Compute offsets/scales:
            # - for log labels, use log1p(min/max) to define the range in log-space
            y_min = t_min.clone()
            y_max = t_max.clone()

            if torch.any(logmask):
                y_min_log = torch.log1p(t_min[logmask])
                y_max_log = torch.log1p(t_max[logmask])
                y_min[logmask] = y_min_log
                y_max[logmask] = y_max_log

            self._y_off   = y_min
            self._y_scale = (y_max - y_min).clamp_min(eps).reciprocal()
            self._y_logmask = logmask
        else:
            self._y_off = self._y_scale = self._y_logmask = None

        self._on_device = False  # move lazily

    def _ensure_on_device(self, device: torch.device) -> None:
        """
        Lazily move normalization parameters to the specified device.

        Args:
            device (torch.device): The target device (CPU or GPU) to move normalization parameters onto.
        """
        if self._on_device:
            return
        if self._x_off is not None:
            self._x_off   = self._x_off.to(device, non_blocking=True)
            self._x_scale = self._x_scale.to(device, non_blocking=True)
        if self._y_off is not None:
            self._y_off      = self._y_off.to(device, non_blocking=True)
            self._y_scale    = self._y_scale.to(device, non_blocking=True)
        if self._y_logmask is not None:
            self._y_logmask  = self._y_logmask.to(device)
        self._on_device = True

    def _normalize_inplace(self, trainer: Trainer, batch: Batch) -> None:
        """
        Apply in-place normalization to a PyG Batch object.

        Normalizes:
            - `batch.x` (feature matrix) using `_x_off` and `_x_scale`.
            - `batch.y` (target labels) using `_y_off` and `_y_scale`, with optional log-scaling for selected labels.

        Args:
            trainer (Trainer): The active training loop instance (provides target device).
            batch (Batch): PyTorch Geometric `Batch` containing graph features (`x`) and labels (`y`).
        """
        self._ensure_on_device(trainer.device)

        x = getattr(batch, "x", None)
        if x is not None and self._x_off is not None:
            if x.dtype is not torch.float32:
                batch.x = x = x.float()
            x.add_(-self._x_off).mul_(self._x_scale)

        y = getattr(batch, "y", None)
        if y is not None and self._y_off is not None:
            if not torch.is_floating_point(y):
                y = y.float()

            # Ensure shape [B, L] for broadcast; keep single-label case efficient
            if y.ndim == 1:
                if self._y_off.numel() == 1:
                    if self._y_logmask is not None and self._y_logmask.numel() == 1 and bool(self._y_logmask[0]):
                        y = torch.log1p(y)
                    y = y.add(-self._y_off).mul(self._y_scale)
                else:
                    y = y.unsqueeze(1)
                    # fall through to 2D path
            if y.ndim == 2:
                if self._y_logmask is not None and torch.any(self._y_logmask):
                    # Apply log1p only to selected label columns
                    cols = torch.nonzero(self._y_logmask, as_tuple=False).squeeze(1)
                    y[:, cols] = torch.log1p(y[:, cols])
                y = y.add(-self._y_off).mul(self._y_scale)
                if y.shape[1] == 1:
                    y = y.squeeze(1)

            batch.y = y