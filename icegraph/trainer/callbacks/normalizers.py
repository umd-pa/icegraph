# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Literal
import sys

import torch

from icegraph.console import Console
from .base import Normalizer

__all__ = ["MinMaxNormalizer", "resolve_normalizer"]


def resolve_normalizer(name: str) -> Normalizer:
    __module__ = sys.modules[__name__]
    try:
        cls = getattr(__module__, name)
    except AttributeError:
        raise ValueError(f"Normalizer class '{name}' not found in {__module__.__name__}")
    if not callable(cls):
        raise TypeError(f"{name} is not callable.")
    return cls()

class MinMaxNormalizer(Normalizer):
    """
    Normalization callback for scaling input features and target labels to the [0, 1] range
    using precomputed minimum and maximum statistics. Supports optional log-scaling of
    selected target labels prior to normalization.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MinMaxNormalizer with empty offsets/scales for features and labels.
        """
        _param_list = ["_x_off", "_x_scale", "_y_off", "_y_scale", "_y_logmask"]

        # call to super
        super().__init__(_param_list, **kwargs)

    def _normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        """
        Apply min-max normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            field (Literal['x', 'y']): Whether this tensor represents labels.

        Returns:
            Tensor: Normalized tensor (same shape).
        """
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        if field == "y":
            # ensure float
            if not torch.is_floating_point(tensor):
                tensor = tensor.float()

            # always work in [B, L] then squeeze back if needed
            squeezed = False
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
                squeezed = True

            y_logmask = self._params["_y_logmask"]

            # (optional) log1p selected columns
            if y_logmask is not None and torch.any(y_logmask):
                cols = torch.nonzero(y_logmask, as_tuple=False).squeeze(1)
                # handle single-column mask robustly
                if cols.numel() > 0:
                    tensor[:, cols] = torch.log1p(tensor[:, cols])

            # min-max normalize
            tensor = tensor.add_(-self._params["_y_off"]).mul_(self._params["_y_scale"])

            if squeezed or tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)

        elif field == 'x':
            tensor = tensor.add_(-self._params["_x_off"]).mul_(self._params["_x_scale"])

        return tensor

    def _configure(self, trainer) -> None:
        """
        Configure min/max-based normalization parameters from dataset statistics.
        """
        target_labels = list(trainer.target_labels or [])
        log_labels = set(trainer.apply_log_scaling or [])

        # Features
        f_min = torch.as_tensor(self.f_stats.min, dtype=torch.float32)
        f_max = torch.as_tensor(self.f_stats.max, dtype=torch.float32)
        self._params["_x_off"] = f_min
        self._params["_x_scale"] = (f_max - f_min).clamp_min(self._eps).reciprocal()

        if not target_labels:
            self._params["_y_off"] = self._params["_y_scale"] = self._params["_y_logmask"] = None
            return

        # Labels (ordered to match trainer.target_labels)
        idx = [self.t_stats.columns.index(n) for n in target_labels]
        t_min = torch.as_tensor(self.t_stats.min, dtype=torch.float32)[idx]
        t_max = torch.as_tensor(self.t_stats.max, dtype=torch.float32)[idx]

        logmask = torch.tensor([n in log_labels for n in target_labels], dtype=torch.bool)

        # Disable log for labels with negative mins
        bad_mask = logmask & (t_min < 0)
        if torch.any(bad_mask):
            bad = [n for n, b in zip(target_labels, bad_mask.tolist()) if b]
            Console.out(f"Warning: negative values in log-scaled labels {bad}; disabling log.", severity=2)
            logmask &= ~bad_mask

        # Define y_min/y_max in (optional) log-space
        y_min = t_min.clone()
        y_max = t_max.clone()
        if torch.any(logmask):
            y_min[logmask] = torch.log1p(y_min[logmask])
            y_max[logmask] = torch.log1p(y_max[logmask])

        self._params["_y_off"] = y_min
        self._params["_y_scale"] = (y_max - y_min).clamp_min(self._eps).reciprocal()
        self._params["_y_logmask"] = logmask