# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Literal

import torch

from icegraph.console import Console
from .normalizer import Normalizer

__all__ = ["MinMaxNormalizer"]


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

    def normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
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

        else:
            raise ValueError("field must be 'x' or 'y'")

        return tensor

    def inverse_normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        """
        Apply inverse of min-max normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            field (Literal['x', 'y']): Whether this tensor represents features or labels.

        Returns:
            Tensor: De-normalized tensor (same shape).
        """
        # ensure float dtype and params on the same device
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        self._ensure_on_device(tensor.device)

        if field == 'y':
            # always work in [B, L] then squeeze back if needed
            squeezed = False
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
                squeezed = True

            y_logmask = self._params["_y_logmask"]

            # undo min-max: y = y / scale + off   (scale was stored as 1/(max-min))
            tensor = tensor.div_(self._params["_y_scale"]).add_(self._params["_y_off"])

            # undo optional log1p on selected columns
            if y_logmask is not None and torch.any(y_logmask):
                cols = torch.nonzero(y_logmask, as_tuple=False).squeeze(1)
                if cols.numel() > 0:
                    tensor[:, cols] = torch.expm1(tensor[:, cols])

            if squeezed or tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)

        elif field == 'x':
            # undo min-max for features
            tensor = tensor.div_(self._params["_x_scale"]).add_(self._params["_x_off"])

        else:
            raise ValueError("field must be 'x' or 'y'")

        return tensor

    def _configure(self, trainer) -> None:
        """
        Configure min/max-based normalization parameters from dataset statistics.
        """
        target_labels = trainer.registry.global_attrs["target_labels"]
        log_labels = trainer.registry.global_attrs["apply_log_scaling_y"]

        # Features
        f_min = torch.as_tensor(self.f_stats.min, dtype=torch.float32)
        f_max = torch.as_tensor(self.f_stats.max, dtype=torch.float32)
        self._params["_x_off"] = f_min
        self._params["_x_scale"] = (f_max - f_min).clamp_min(self._eps).reciprocal()

        if not target_labels:
            self._params["_y_off"] = self._params["_y_scale"] = self._params["_y_logmask"] = None
            return

        # Labels (ordered to match trainer.target_labels)
        idx = [self.l_stats.columns.index(n) for n in target_labels]
        t_min = torch.as_tensor(self.l_stats.min, dtype=torch.float32)[idx]
        t_max = torch.as_tensor(self.l_stats.max, dtype=torch.float32)[idx]

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