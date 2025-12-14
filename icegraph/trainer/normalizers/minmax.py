# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Tuple

import torch
from torch import Tensor

from .normalizer import Normalizer, NormTarget
from .scaling import AsinhScalingMixin
from icegraph.utils import Statistics

__all__ = ["MinMaxNormalizer"]


class MinMaxNormalizer(Normalizer, AsinhScalingMixin):
    """
    MinMax normalizer for scaling input features and target labels to [0, 1] range.
    """

    def _offset_constructor(self, stats: Statistics, c: Tensor, asinh_mask: Tensor) -> Tensor:
        minimum = torch.as_tensor(stats.minimum, dtype=torch.float32)

        # apply asinh scaling
        offset = self.apply_raw_masked_scaling(minimum, c, asinh_mask)

        return offset

    @property
    def _x_off(self) -> Tensor:
        # use the function name for the cache key
        key = "_x_off"

        return self._cached(lambda: self._offset_constructor(
            self._x_stats, self._x_c.cpu(), self._x_asinh_mask.cpu()
        ), key)

    @property
    def _y_off(self) -> Tensor:
        # use the function name for the cache key
        key = "_y_off"

        return self._cached(lambda: self._offset_constructor(
            self._y_stats, self._y_c.cpu(), self._y_asinh_mask.cpu()
        ), key)

    def _scale_constructor(self, stats: Statistics, c: Tensor, asinh_mask: Tensor) -> Tensor:
        minimum = torch.as_tensor(stats.minimum, dtype=torch.float32)
        maximum = torch.as_tensor(stats.maximum, dtype=torch.float32)

        # get the minimum (offset) and maximum with asinh scaling
        minimum = self.apply_raw_masked_scaling(minimum, c, asinh_mask)
        maximum = self.apply_raw_masked_scaling(maximum, c, asinh_mask)

        scale = (maximum - minimum).clamp_min(self._eps).reciprocal()

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

        # apply minmax scaling
        tensor = tensor.add_(-offset).mul_(scale)

        return tensor

    def _inverse_normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        mask, cofactors, offset, scale = self._select_norm_parameters(target)

        # undo minmax scaling
        tensor = tensor.div_(scale).add_(offset)

        # undo asinh transform on specified columns
        if mask.any():
            cols = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if cols.numel() > 0:
                tensor[:, cols] = self._inv_asinh10_torch(tensor[:, cols], cofactors[cols])

        return tensor