# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Literal, Tuple, Any
import math

import torch
from torch import Tensor

from .normalizer import Normalizer, NormTarget
from .scaling import AsinhScalingMixin
from icegraph.utils import Statistics

__all__ = ["ZScoreNormalizer"]

# cache log(10) because it is used frequently
LN10 = math.log(10.0)


class ZScoreNormalizer(Normalizer, AsinhScalingMixin):
    """
    Z-score normalizer with optional asinh(base-10) pre-transform per column.

    For any column where the asinh mask is True, we apply:
        y = asinh(x / c) / ln(10)
    then standardize: (y - off) * scale where scale = 1/std_y
    """

    def _offset_constructor(self, stats: Statistics, c: Tensor, asinh_mask: Tensor) -> Tensor:
        mu = torch.as_tensor(stats.mean, dtype=torch.float32)
        std = torch.as_tensor(stats.stddev(unbiased=False), dtype=torch.float32)

        offset = mu.clone()

        # if any labels are asinh transformed, calculate offset in transformed space
        if asinh_mask.any():
            # calculate offset in transformed space
            xfrm_offset = (
                    torch.asinh(mu / c) / LN10
                    - (mu * std ** 2) / (2 * LN10 * (mu ** 2 + c ** 2) ** (3 / 2))
            )

            # overwrite masked indices with transformed space values
            offset[asinh_mask] = xfrm_offset[asinh_mask]

        return offset

    @property
    def _x_off(self) -> Tensor:
        # use the function name for the cache key
        key = "_x_off"

        def build():
            return self._offset_constructor(
                self._x_stats, self._x_c.cpu(), self._x_asinh_mask.cpu()
            )

        return self._cached(build, key)

    @property
    def _y_off(self) -> Tensor:
        # use the function name for the cache key
        key = "_y_off"

        return self._cached(lambda: self._offset_constructor(
            self._y_stats, self._y_c.cpu(), self._y_asinh_mask.cpu()
        ), key)

    def _scale_constructor(self, stats: Statistics, c: Tensor, asinh_mask: Tensor) -> Tensor:
        mu = torch.as_tensor(stats.mean, dtype=torch.float32)
        std = torch.as_tensor(stats.stddev(unbiased=False), dtype=torch.float32)

        inv_scale = std.clone()

        # if any labels are asinh transformed, calculate scaling factor in transformed space
        if asinh_mask.any():
            # calculate inverse scale factor in transformed space
            xfrm_inv_scale = std / (LN10 * (mu ** 2 + c ** 2).sqrt())

            # overwrite masked indices with transformed space values
            inv_scale[asinh_mask] = xfrm_inv_scale[asinh_mask]

        scale = inv_scale.clamp(min=self._eps).reciprocal()
        return scale

    @property
    def _x_scale(self) -> Tensor:
        # use the function name for the cache key
        key = "_x_scale"

        return self._cached(lambda: self._scale_constructor(
            self._x_stats, self._x_c.cpu(), self._x_asinh_mask.cpu()
        ), key)

    @property
    def _y_scale(self) -> Tensor:
        # use the function name for the cache key
        key = "_y_scale"

        return self._cached(lambda: self._scale_constructor(
            self._y_stats, self._y_c.cpu(), self._y_asinh_mask.cpu()
        ), key)
    
    def _select_norm_parameters(self, target: NormTarget) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # choose and stash norm parameters
        mask        = self._y_asinh_mask    if target == "labels" else self._x_asinh_mask
        cofactors   = self._y_c             if target == "labels" else self._x_c
        offset      = self._y_off           if target == "labels" else self._x_off
        scale       = self._y_scale         if target == "labels" else self._x_scale

        return mask, cofactors, offset, scale

    def _normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        mask, cofactors, offset, scale = self._select_norm_parameters(target)

        # asinh transform specified columns
        if mask.any():
            cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if cols.numel() > 0:
                tensor[:, cols] = self._asinh10_torch(tensor[:, cols], cofactors[cols])

        # apply z-score scaling
        tensor = tensor.add_(-offset).mul_(scale)

        return tensor

    def _inverse_normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        mask, cofactors, offset, scale = self._select_norm_parameters(target)

        # undo z-score scaling
        tensor = tensor.div_(scale).add_(offset)

        # undo asinh transform on specified columns
        if mask.any():
            cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if cols.numel() > 0:
                tensor[:, cols] = self._inv_asinh10_torch(tensor[:, cols], cofactors[cols])

        return tensor
