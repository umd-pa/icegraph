# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable, Self, Iterable, Literal
from functools import reduce
import copy

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.typing.common import ArrayF, ArrayI, ArrayB

from .statistic import Statistic
from .factory import StatisticFactory
from .bundle import StatisticBundle
from .types import StatisticBundleStruct

__all__ = ["StatisticService"]


class StatisticService:

    def __init__(self, kinds: Iterable[str]) -> None:
        # build the stat bundle
        self.bundle: StatisticBundle = self._build_bundle(kinds)

    @staticmethod
    def _build_bundle(kinds: Iterable[str]) -> StatisticBundle:
        # initialize statistics dict
        stats: dict[str, Statistic] = {}

        for kind in kinds:
            # get an instance for each kind requested
            stats[kind] = StatisticFactory.create(kind)

        return StatisticBundle(stats)

    def __add__(self, other: StatisticService | Literal[0]) -> StatisticService:
        """Addition of two StatisticService objects yields a single merged instance."""
        if other == 0:
            return self

        return self.merge([self, other])

    def __radd__(self, other: StatisticService | Literal[0]) -> StatisticService:
        """Right-hand addition for compatibility with `sum`."""
        # delegate to __add__
        return self.__add__(other)

    def __reduce__(self) -> tuple[Callable[[StatisticBundleStruct], StatisticService], tuple[StatisticBundleStruct]]:
        # When unpickling, pickle will call:
        # .load_struct(struct)
        # which will fully rebuild the instance
        struct = self.to_struct()
        return self.__class__.from_struct, (struct,)

    @classmethod
    def merge(cls, services: Iterable[StatisticService]) -> StatisticService:
        """
        Merge two or more StatisticService instances into one.

        Args:
            services: Iterable of StatisticService objects to merge.
        """
        # verify types
        if not isinstance(services, Iterable):
            raise TypeError(f"Argument 'services' must be an iterable over {cls.__name__} objects.")

        # coerce to list
        services = list(services)

        # all items in collection need to be statistic service objects
        if any(not isinstance(s, cls) for s in services):
            raise TypeError(f"Argument 'services' must be an iterable over {cls.__name__} objects.")

        # break out if empty stats list
        if not services:
            raise ValueError("Cannot merge an empty list.")

        # break out if only one service passed
        if len(services) == 1:
            return services[0].copy()

        # merge bundles
        merged_bundle = reduce(StatisticBundle.merge, (service.bundle for service in services))

        return cls._from_bundle(merged_bundle)

    @classmethod
    def _from_bundle(cls, bundle: StatisticBundle) -> StatisticService:
        """
        Construct a StatisticService instance from a StatisticBundle and column names.

        Args:
            bundle: StatisticBundles to build the service with.
        """
        # instantiate
        instance = cls.__new__(cls)

        # populate
        instance.bundle = bundle

        # return
        return instance

    def compute_from_array(self, array: ArrayF) -> None:
        """
        Compute all statistics from a dense array. This method overrides all previously
        stored statistic information in the bundle, and is intended to be
        used only once per instance.

        The input array is treated as shape (N, F), where N is the number of
        samples and F is the number of columns.

        Args:
            array: 2D NumPy array of shape (N, F).
        """
        self.bundle.compute(array)

    def to_struct(self) -> StatisticBundleStruct:
        """Serialize the StatisticService instance into a packable dict structure."""
        return self.bundle.to_struct()

    @classmethod
    def from_struct(cls, struct: StatisticBundleStruct) -> StatisticService:
        """
        Construct a StatisticService instance from a serialized struct. Inverse method of `to_struct()`.

        Args:
            struct: A dictionary with same format as produced by `to_struct()`.
        """
        return cls._from_bundle(StatisticBundle.from_struct(struct))

    def num_columns(self) -> int:
        """Return the number of columns tracked by the internal stat bundle."""
        return self.bundle.num_columns()

    def align_to(self, indices: ArrayI) -> Self:
        """
        Reorder internal stat arrays in-place to match a new order.

        Args:
            indices: Desired index ordering.
        """
        self.bundle.align_to(indices)
        return self

    def filter_to(self, mask: ArrayB) -> Self:
        """
        Filter internal stat arrays in-place.

        This method only permits filtering and does not reorder.

        Args:
            mask: Mask to filter to.
        """
        self.bundle.filter_to(mask)
        return self

    def copy(self) -> Self:
        """Return a copy of this instance."""
        return copy.deepcopy(self)

    ### RAW AND DERIVED STATISTIC ACCESS

    def get(self, kind: str, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10) -> ArrayF:
        """
       Return a raw statistic from the underlying bundle.

       Args:
           kind: Statistic to retrieve (e.g. min, max, mean).
           space: Transform space in which the statistic was computed.
           base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
       """
        # get the statistic object and extract value
        statistic = self.bundle.get(kind)
        array = statistic.value(space)

        # translate to correct base for log and asinh (this should apply to asinh, changing the "base"
        # of asinh(x) correlates to changing the base of the log-like behavior of asinh at the tails)
        if space in (TransformSpace.LOG, TransformSpace.ASINH):
            if base <= 0 or base == 1:
                raise ValueError("Requested base must be > 0 and != 1 for LOG/ASINH spaces.")
            return array / (np.log(base) ** statistic.degree)

        # if not log or asinh (linear), return raw
        return array

    def valid_count(self, space: TransformSpace) -> ArrayF:
        """
        Derived statistic: requires FINITE_COUNT and POSITIVE_COUNT. Returns columnwise effective count of
        valid samples for the given transform space.

        Args:
            space: Transform space for which to compute the effective valid sample count.
        """
        if space in (TransformSpace.LINEAR, TransformSpace.ASINH):
            return self.get("finite_count", space=TransformSpace.LINEAR)
        elif space == TransformSpace.LOG:
            return self.get("positive_count", space=TransformSpace.LINEAR)
        else:
            raise TypeError(f"Parameter 'space' must be a TransformSpace, got {type(space).__name__}.")

    def geometric_mean(self, *, base: int = 10) -> ArrayF:
        """
        Derived statistic: requires MEAN. Returns columnwise geometric mean.

        Args:
            base: Log base used for computing the geometric mean.
        """
        # only defined in log space, so only provided in log space
        mean_log = self.get("mean", space=TransformSpace.LOG, base=base)

        # geo_mean = base ** mean(log_base(x))
        return base ** mean_log

    def variance(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires M2 and valid_count. Returns columnwise variance.

        Args:
            space: Transform space in which to compute the variance.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        m2 = self.get("m2", space=space, base=base)
        valid_count = self.valid_count(space)

        # for biased use n, for unbiased use n - 1
        denom = valid_count if biased else valid_count - 1

        # variance = m2 / n
        return np.where(denom > 0, m2 / denom, np.nan)

    def std(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires variance. Returns columnwise standard deviation.

        Args:
            space: Transform space in which to compute the standard deviation.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        variance = self.variance(space=space, base=base, biased=biased)

        # std = sqrt(variance)
        return np.sqrt(variance)

    def range(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10) -> ArrayF:
        """
        Derived statistic: requires MIN and MAX. Returns columnwise range.

        Args:
            space: Transform space in which to compute the range.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
        """
        maximum = self.get("max", space=space, base=base)
        minimum = self.get("min", space=space, base=base)

        # range = max - min
        return maximum - minimum

    def sem(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires std and valid_count. Returns columnwise standard error of the mean (SEM).

        Args:
            space: Transform space in which to compute the SEM.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        std = self.std(space=space, base=base, biased=biased)
        valid_count = self.valid_count(space)

        # SEM = std / sqrt(n)
        return np.where(valid_count > 0, std / np.sqrt(valid_count), np.nan)

    def cv(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires std and MEAN. Returns columnwise coefficient of variation (CV).

        Args:
            space: Transform space in which to compute the coefficient of variation.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        std = self.std(space=space, base=base, biased=biased)
        mean = self.get("mean", space=space, base=base)

        # CV = std / mean
        return np.where(mean != 0, std / mean, np.nan)

    def rms(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires variance and MEAN. Returns columnwise root-mean-square (RMS).

        Args:
            space: Transform space in which to compute the RMS.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        variance = self.variance(space=space, base=base, biased=biased)
        mean = self.get("mean", space=space, base=base)

        # RMS = sqrt(variance + mean ** 2)
        return np.sqrt(variance + np.square(mean))

    def snr(self, *, space: TransformSpace = TransformSpace.LINEAR, base: int = 10, biased: bool = False) -> ArrayF:
        """
        Derived statistic: requires std and MEAN. Returns columnwise signal-to-noise ratio (SNR).

        Args:
            space: Transform space in which to compute the SNR.
            base: Log base used for log- and asinh-transformed statistics. Ignored for linear space.
            biased: Whether to compute the biased estimator.
        """
        std = self.std(space=space, base=base, biased=biased)
        mean = self.get("mean", space=space, base=base)

        # SNR = mean / std
        return np.where(std != 0, mean / std, np.nan)
