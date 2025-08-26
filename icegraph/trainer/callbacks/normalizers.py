# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Literal, Sequence
import sys
import math

import torch

from icegraph.console import Console
from .base import Normalizer
from icegraph.data.base import IGData

__all__ = ["MinMaxNormalizer", "ZScoreNormalizer", "resolve_normalizer"]


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
        target_labels = IGData.attrs[0]["global"]["target_labels"]
        log_labels = IGData.attrs[0]["global"]["apply_log_scaling_y"]

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


LN10 = math.log(10.0)

class ZScoreNormalizer(Normalizer):
    """
    Z-score normalizer with optional asinh(base-10) pre-transform per column.

    For any column where the asinh mask is True, we apply:
        y = asinh(x / c) / ln(10)
    then standardize: (y - off) * scale   where scale = 1/std_y
    """

    def __init__(self, **kwargs) -> None:
        # store per-field means/scales in transformed space, plus masks and cofactors
        _param_list = [
            "_x_off", "_x_scale", "_x_asinh_mask", "_x_c",
            "_y_off", "_y_scale", "_y_asinh_mask", "_y_c",
        ]
        super().__init__(_param_list, **kwargs)

    @staticmethod
    def _asinh10_torch(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # y = asinh(x/c) / ln(10)
        return torch.asinh(x / c) / LN10

    @staticmethod
    def _inv_asinh10_torch(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x = c * sinh(y * ln(10))
        return c * torch.sinh(y * LN10)

    def normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        self._ensure_on_device(tensor.device)

        if field == "y":
            squeezed = False
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
                squeezed = True

            # optional asinh(base-10) on masked label columns
            mask = self._params["_y_asinh_mask"]
            if mask is not None and torch.any(mask):
                cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if cols.numel() > 0:
                    c = self._params["_y_c"][cols]
                    tensor[:, cols] = self._asinh10_torch(tensor[:, cols], c)

            # z-score
            tensor = tensor.add_(-self._params["_y_off"]).mul_(self._params["_y_scale"])

            if squeezed or tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            return tensor

        elif field == "x":
            # expect [N, F]
            mask = self._params["_x_asinh_mask"]
            if mask is not None and torch.any(mask):
                cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if cols.numel() > 0:
                    c = self._params["_x_c"][cols]
                    tensor[:, cols] = self._asinh10_torch(tensor[:, cols], c)

            tensor = tensor.add_(-self._params["_x_off"]).mul_(self._params["_x_scale"])
            return tensor

        else:
            raise ValueError("field must be 'x' or 'y'")

    def inverse_normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        self._ensure_on_device(tensor.device)

        if field == 'y':
            squeezed = False
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
                squeezed = True

            # undo z-score
            tensor = tensor.div_(self._params["_y_scale"]).add_(self._params["_y_off"])

            # undo asinh on masked columns
            mask = self._params["_y_asinh_mask"]
            if mask is not None and torch.any(mask):
                cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if cols.numel() > 0:
                    c = self._params["_y_c"][cols]
                    tensor[:, cols] = self._inv_asinh10_torch(tensor[:, cols], c)

            if squeezed or tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            return tensor

        elif field == 'x':
            tensor = tensor.div_(self._params["_x_scale"]).add_(self._params["_x_off"])

            mask = self._params["_x_asinh_mask"]
            if mask is not None and torch.any(mask):
                cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
                if cols.numel() > 0:
                    c = self._params["_x_c"][cols]
                    tensor[:, cols] = self._inv_asinh10_torch(tensor[:, cols], c)
            return tensor

        else:
            raise ValueError("field must be 'x' or 'y'")

    def _configure(self, trainer) -> None:
        """
        Configure z-score parameters in the correct transform space.
        """
        eps = self._eps
        device = torch.device("cpu")

        g = IGData.attrs[0]["global"]
        apply_asinh_y_names: Sequence[str] = g.get("apply_asinh_scaling_y", g.get("apply_log_scaling_y", []))
        apply_asinh_x_names: Sequence[str] = g.get("apply_asinh_scaling_x", [])

        # Features mask
        f_cols = self.f_stats.columns
        x_mask_list = [name in apply_asinh_x_names for name in f_cols]
        x_mask = torch.tensor(x_mask_list, dtype=torch.bool)

        # Labels mask (ordered by the active target label order)
        target_labels = IGData.attrs[0]["global"]["target_labels"]
        if not target_labels:
            y_mask = None
        else:
            y_mask_list = [name in apply_asinh_y_names for name in target_labels]
            y_mask = torch.tensor(y_mask_list, dtype=torch.bool)

        # features
        f_c_np = self.f_stats.asinh_cofactor()  # shape [F]
        f_c = torch.as_tensor(f_c_np, dtype=torch.float32, device=device)

        # labels (align stats to target label order)
        if target_labels:
            idx = [self.l_stats.columns.index(n) for n in target_labels]
            l_stats = self.l_stats.aligned_to(list(target_labels))
            l_c_np = l_stats.asinh_cofactor()
            l_c = torch.as_tensor(l_c_np, dtype=torch.float32, device=device)
        else:
            l_c = None

        # helper to compute transformed-space mean/std via delta-method
        def transformed_moments(mu: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # mean_y ≈ asinh(mu/c)/ln10 - mu*sigma^2 / (2 ln10 (mu^2+c^2)^(3/2))
            # std_y  ≈ sigma / (ln10 * sqrt(mu^2 + c^2))
            mu2 = mu * mu
            c2 = c * c
            denom_root = torch.sqrt(mu2 + c2).clamp_min(1e-30)
            std_y = sigma / (LN10 * denom_root)
            corr = (mu * (sigma * sigma)) / (2.0 * LN10 * (mu2 + c2) * denom_root)
            mean_y = torch.asinh(mu / c) / LN10 - corr
            return mean_y, std_y

        f_mu = torch.as_tensor(self.f_stats.mean, dtype=torch.float32, device=device)
        f_sigma = torch.as_tensor(self.f_stats.stddev(unbiased=False), dtype=torch.float32, device=device)
        f_sigma = torch.nan_to_num(f_sigma, nan=0.0)

        # defaults: raw-space z-score
        x_off = f_mu.clone()
        x_std = f_sigma.clone()

        # apply asinh moments on masked columns
        if torch.any(x_mask):
            cols = torch.nonzero(x_mask, as_tuple=False).squeeze(1)
            mu_t, std_t = transformed_moments(f_mu[cols], f_sigma[cols], f_c[cols])
            x_off[cols] = mu_t
            x_std[cols] = std_t

        x_std = torch.clamp(x_std, min=eps)
        x_scale = 1.0 / x_std

        if not target_labels:
            y_off = y_scale = y_mask_tensor = y_c_tensor = None
        else:
            l_mu = torch.as_tensor(l_stats.mean, dtype=torch.float32, device=device)
            l_sigma = torch.as_tensor(l_stats.stddev(unbiased=False), dtype=torch.float32, device=device)
            l_sigma = torch.nan_to_num(l_sigma, nan=0.0)

            # defaults: raw-space z-score
            y_off = l_mu.clone()
            y_std = l_sigma.clone()

            if y_mask is not None and torch.any(y_mask):
                cols = torch.nonzero(y_mask, as_tuple=False).squeeze(1)
                mu_t, std_t = transformed_moments(l_mu[cols], l_sigma[cols], l_c[cols])
                y_off[cols] = mu_t
                y_std[cols] = std_t

            y_std = torch.clamp(y_std, min=eps)
            y_scale = 1.0 / y_std

            y_mask_tensor = y_mask
            y_c_tensor = l_c

        # stash everything
        self._params["_x_off"] = x_off
        self._params["_x_scale"] = x_scale
        self._params["_x_asinh_mask"] = x_mask
        self._params["_x_c"] = f_c

        if target_labels:
            self._params["_y_off"] = y_off
            self._params["_y_scale"] = y_scale
            self._params["_y_asinh_mask"] = y_mask_tensor
            self._params["_y_c"] = y_c_tensor
        else:
            self._params["_y_off"] = None
            self._params["_y_scale"] = None
            self._params["_y_asinh_mask"] = None
            self._params["_y_c"] = None
