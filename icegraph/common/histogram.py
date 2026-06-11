# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import functools

import numpy as np

from icegraph.typing.common import ArrayF32, ArrayF64, ArrayI64, ArrayG
from icegraph.common.transforms import TransformSpace

__all__ = ["Histogram"]


def require_bounds(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def inner(self, *args, **kwargs) -> Any:
        assert self.bounds is not None, f"Method {func.__name__} is not available if bounds is None."

        return func(self, *args, **kwargs)
    return inner


@dataclass(frozen=True, slots=True)
class Histogram:
    space:      tuple[TransformSpace, ...]
    histogram:  ArrayF32

    # optional fields
    bounds:     ArrayF64 | None    = None
    overflow:   ArrayF32 | None    = None

    def __post_init__(self) -> None:
        # ensure correct dtypes everywhere
        # need to use object.__setattr__ as we are working in a frozen dataclass
        histogram = np.asarray(self.histogram, dtype=np.float32)
        object.__setattr__(self, "histogram", histogram)

        if self.overflow is not None:
            # set overflow to ArrayF32
            object.__setattr__(self, "overflow", np.asarray(self.overflow, dtype=np.float32))

        if self.bounds is not None:
            # set bounds to ArrayF64 (want more precision here, and bounds is small)
            bounds = np.asarray(self.bounds, dtype=np.float64)

            # verify shape
            if bounds.ndim != 2 or bounds.shape[0] != 2:
                raise ValueError(f"Bounds must have shape (2, ndim); got {bounds.shape}.")

            # verify data
            if np.any(bounds[0] >= bounds[1]):
                raise ValueError("All bounds must satisfy min < max.")

            # verify one set of bounds for each histogram dim
            if histogram.ndim != bounds.shape[1]:
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
        return self.bounds[0]

    @property
    @require_bounds
    def maxs(self) -> ArrayF64:
        return self.bounds[1]

    @property
    def bins(self) -> tuple[int, ...]:
        return self.histogram.shape

    @property
    @require_bounds
    def edges(self) -> tuple[ArrayF64, ...]:
        return tuple(  # type: ignore
            np.linspace(low, high, n + 1) for low, high, n in zip(self.mins, self.maxs, self.bins)
        )

    @property
    @require_bounds
    def widths(self) -> ArrayF64:
        return (self.maxs - self.mins) / np.asarray(self.bins, dtype=np.float64)  # type: ignore

    @property
    @require_bounds
    def centers(self) -> tuple[ArrayF64, ...]:
        return tuple((edge[:-1] + edge[1:]) / 2 for edge in self.edges)  # type: ignore

    @require_bounds
    def count_quantile(self, threshold: float, axis: int = 0) -> ArrayI64:
        """
        Computes count-quantile bin indices along the specified axis.

        Returns an integer array with the reduced shape. Invalid entries
        (where the total count along the reduced axis is 0) are set to -1.
        """
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
        quantile_indices = np.argmax(cumulative >= targets, axis=axis)

        # init invalid output as -1
        indices = np.full(totals.shape, -1, dtype=np.int64)

        valid = totals > 0
        indices[valid] = quantile_indices[valid]

        return indices  # type: ignore

    def apply(self, fn: Callable[[ArrayG], ArrayG]) -> None:
        """Applies the given function in-place to histogram data."""
        object.__setattr__(
            self,
            "histogram",
            np.asarray(fn(self.histogram), dtype=np.float32),
        )

    def to_struct(self) -> dict[str, Any]:
        struct: dict[str, Any] = {
            "space": [space.name for space in self.space],
            "histogram": self.histogram.tolist(),
            "bounds": None if self.bounds is None else self.bounds.tolist(),
            "overflow": None if self.overflow is None else self.overflow.tolist(),
        }

        return struct

    @classmethod
    def from_struct(cls, struct: dict[str, Any]) -> Histogram:
        return cls(
            space=tuple(TransformSpace[name] for name in struct["space"]),
            histogram=np.asarray(struct["histogram"], dtype=np.float32),
            bounds=(
                None
                if struct.get("bounds") is None
                else np.asarray(struct["bounds"], dtype=np.float64)
            ),
            overflow=(
                None
                if struct.get("overflow") is None
                else np.asarray(struct["overflow"], dtype=np.float32)
            ),
        )

