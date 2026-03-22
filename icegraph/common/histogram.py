# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass
from typing import Callable, Any
import functools

import numpy as np

from icegraph.types.common import ArrayF32, ArrayF64
from icegraph.types.transforms import TransformSpace

__all__ = ["Histogram"]


def require_bounds(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def inner(self, *args, **kwargs) -> Any:
        assert self.bounds is not None, f"Method {func.__name__} is not available if bounds is None."

        return func(self, *args, **kwargs)
    return inner


@dataclass(frozen=True, slots=True)
class Histogram:
    name:       str
    histogram:  ArrayF32
    space:      tuple[TransformSpace, ...]

    # optional fields
    bounds:     ArrayF64 | None    = None
    extended:   ArrayF32 | None    = None
    overflow:   ArrayF32 | None    = None

    def __post_init__(self) -> None:
        # ensure correct dtypes everywhere
        # need to use object.__setattr__ as we are working in a frozen dataclass
        histogram = np.asarray(self.histogram, dtype=np.float32)
        object.__setattr__(self, "histogram", histogram)

        if self.extended is not None:
            # set extended to ArrayF32
            object.__setattr__(self, "extended", np.asarray(self.extended, dtype=np.float32))

        if self.overflow is not None:
            # set overflow to ArrayF32
            object.__setattr__(self, "overflow", np.asarray(self.overflow, dtype=np.float32))

        if self.bounds is not None:
            # set bounds to ArrayF64 (want more precision here, and bounds is small)
            bounds = np.asarray(self.bounds, dtype=np.float64)

            # verify shape
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(f"Bounds must have shape (ndim, 2); got {bounds.shape}.")

            # verify data
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError("All bounds must satisfy min < max.")

            # verify one set of bounds for each histogram dim
            if histogram.ndim != bounds.shape[0]:
                raise ValueError(f"Bounds must be specified for each histogram dimension.")

            object.__setattr__(self, "bounds", bounds)

    @property
    def peak_value(self) -> np.floating[Any]:
        return self.histogram[self.peak_index]

    @property
    def peak_index(self) -> tuple[np.intp, ...]:
        return np.unravel_index(np.nanargmax(self.histogram), self.histogram.shape)

    @property
    @require_bounds
    def mins(self) -> ArrayF64:
        return self.bounds[:, 0]

    @property
    @require_bounds
    def maxs(self) -> ArrayF64:
        return self.bounds[:, 1]

    @property
    def bins(self) -> tuple[int, ...]:
        return self.histogram.shape

    @property
    @require_bounds
    def edges(self) -> tuple[ArrayF64, ...]:
        return tuple(
            np.linspace(low, high, n + 1) for low, high, n in zip(self.bounds[:, 0], self.bounds[:, 1], self.bins)
        )

    @property
    @require_bounds
    def widths(self) -> ArrayF64:
        return (self.bounds[:, 1] - self.bounds[:, 0]) / np.asarray(self.bins, dtype=np.float64)

    @property
    @require_bounds
    def centers(self) -> tuple[ArrayF64, ...]:
        return tuple((edge[:-1] + edge[1:]) / 2 for edge in self.edges)

    @require_bounds
    def count_quantile(self, threshold: float, axis: int = 0) -> ArrayF64:
        """Computes the count medians along specified axis of the histogram."""
        if not (-self.histogram.ndim <= axis < self.histogram.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for histogram.")

        # normalize axis to positive
        axis = axis % self.histogram.ndim

        if not (0 < threshold < 1):
            raise ValueError(f"Threshold must be a value between 0 and 1, got {threshold}.")

        # compute totals along the specified axis
        cumulative = np.cumsum(self.histogram, axis=axis)
        totals = cumulative.take(-1, axis=axis)

        # determine quantile indices
        targets = np.expand_dims(threshold * totals, axis=axis)
        median_indices = np.argmax(cumulative >= targets, axis=axis)

        # grab axis centers
        centers = self.centers[axis]

        # init nan array, fill with medians later
        medians = np.full(totals.shape, np.nan, dtype=centers.dtype)

        # mask for valid medians (in case totals = 0 anywhere)
        valid = totals > 0

        # apply mask and populate medians array, invalid entries will be nan
        medians[valid] = centers[median_indices[valid]]

        return medians

