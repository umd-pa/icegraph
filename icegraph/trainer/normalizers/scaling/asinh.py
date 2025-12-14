# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import math
from typing import Protocol, Any, Dict, Callable

import torch
from torch import Tensor

__all__ = ["AsinhScalingMixin"]

# cache log(10) because it is used frequently
LN10 = math.log(10.0)


class AsinhMixinProtocol(Protocol):
    """
    Protocol defining the required interface for any class that uses
    AsinhScalingMixin. This should hopefully satisfy the type checker.
    """
    _cache: Dict[str, Tensor]
    _attrs: Dict[str, Any]
    _x_stats: Any
    _y_stats: Any

    # Required methods to satisfy helper calls
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def _cached(self, build: Callable[[], Tensor], key: str, dtype: torch.dtype = torch.float32) -> Tensor: ...


class AsinhScalingMixin:

    def apply_raw_masked_scaling(self, tensor: Tensor, c: Tensor, asinh_mask: Tensor) -> Tensor:
        if asinh_mask.any():
            tensor[asinh_mask] = self._asinh10_torch(tensor, c)[asinh_mask]

        return tensor

    @staticmethod
    def _asinh10_torch(x: Tensor, c: Tensor) -> Tensor:
        # y = asinh(x/c) / ln(10)
        return torch.asinh(x / c) / LN10

    @staticmethod
    def _inv_asinh10_torch(y: Tensor, c: Tensor) -> Tensor:
        # x = c * sinh(y * ln(10))
        return c * torch.sinh(y * LN10)

    @property
    def _x_asinh_mask(self: AsinhMixinProtocol) -> Tensor:
        # use the function name for the cache key
        key = "_x_asinh_mask"

        def build() -> Tensor:
            return torch.as_tensor(
                [name in self._attrs["apply_log_scaling_x"] for name in self._x_stats.columns], dtype=torch.bool
            )

        return self._cached(build, key, dtype=torch.bool)

    @property
    def _y_asinh_mask(self: AsinhMixinProtocol) -> Tensor:
        # use the function name for the cache key
        key = "_y_asinh_mask"

        def build() -> Tensor:
            return torch.as_tensor(
                [name in self._attrs["apply_log_scaling_y"] for name in self._y_stats.columns], dtype=torch.bool
            )

        return self._cached(build, key, dtype=torch.bool)

    @property
    def _x_c(self: AsinhMixinProtocol) -> Tensor:
        # use the function name for the cache key
        key = "_x_c"

        def build() -> Tensor:
            return self._x_stats.asinh_cofactor()

        return self._cached(build, key)

    @property
    def _y_c(self: AsinhMixinProtocol) -> Tensor:
        # use the function name for the cache key
        key = "_y_c"

        def build() -> Tensor:
            return self._y_stats.asinh_cofactor()

        return self._cached(build, key)