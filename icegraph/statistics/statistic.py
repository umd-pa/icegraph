# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Self, ClassVar, Mapping, TYPE_CHECKING
import inspect
import time

# local package
from icegraph.types.transforms import TransformSpace
from icegraph.types.common import ArrayF, ArrayB, ArrayI
from icegraph.types.statistics import StatisticStruct

# local subpackage
from .transforms import linear_transform, log_transform, asinh_transform

if TYPE_CHECKING:
    from .bundle import StatisticBundle

__all__ = ["Statistic"]

# module logger
import logging
logger = logging.getLogger(__name__)


class Statistic(ABC):
    # name for registration
    name: ClassVar[str]

    # spaces the statistic is to be tracked in
    spaces: ClassVar[tuple[TransformSpace, ...]] = tuple(TransformSpace)

    # degree of the statistic under linear rescaling of transformed values
    degree: ClassVar[int]

    # for transforms to correct space
    transform: ClassVar[Mapping[TransformSpace, Callable[[ArrayF], ArrayF]]] = {
        TransformSpace.LINEAR:  linear_transform,
        TransformSpace.LOG:     log_transform,
        TransformSpace.ASINH:   asinh_transform
    }

    def __init__(self) -> None:
        self._values: dict[TransformSpace, ArrayF] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # break out on abstract subclasses (no enforcement)
        if inspect.isabstract(cls):
            return

        # ensure degree is an int
        degree = getattr(cls, "degree", None)
        if not isinstance(degree, int):
            raise TypeError("Class variable 'degree' must be of type int.")

        # ensure degree is non-negative
        if degree < 0:
            raise ValueError("Class variable 'degree' must be non-negative.")

        # ensure spaces is a tuple
        spaces = getattr(cls, "spaces", None)
        if not isinstance(spaces, tuple):
            raise TypeError("Class variable 'spaces' must be a tuple.")

        # ensure spaces is not empty
        if not spaces:
            raise ValueError("Class variable 'spaces' must be non-empty.")

        # ensure elements of spaces are of correct type
        if not all(isinstance(space, TransformSpace) for space in spaces):
            raise TypeError(f"Elements of class variable 'spaces' must be of type {TransformSpace.__name__}.")

        # ensure spaces has no duplicates
        if len(set(spaces)) != len(spaces):
            raise ValueError("Class variable 'spaces' must not contain duplicates.")


    def value(self, space: TransformSpace) -> ArrayF:
        """
        Return the statistic values.

        Args:
            space: Transform space in which to express the values.
        """
        if space not in self.spaces:
            raise KeyError(f"Statistic {type(self).__name__} does not track stats in space '{space}'")
        return self._values[space]

    def compute(self, array: ArrayF) -> None:
        """
        Compute the statistic from an input array.

        Args:
            array: Input data array used to compute the statistic.
        """
        # start time
        start = time.perf_counter()

        # reset stats first
        self._values.clear()

        # ensure correct shape
        array = self._ensure_shape(array)

        for space in self.spaces:
            # compute the stat on transformed data in each space
            self._values[space] = self._compute(self.transform[space](array))

        # log time elapsed
        logger.debug(
            f"stat={self.name}: compute completed in %.3fms",
            (time.perf_counter() - start) * 1000,
        )

    @abstractmethod
    def _compute(self, array: ArrayF) -> ArrayF:
        """
        Internal implementation of the statistic computation.

        Args:
            array: Input data array used to compute the raw statistic values.
        """
        ...

    @classmethod
    def merge(cls, a: StatisticBundle, b: StatisticBundle) -> Self:
        """
        Merge two statistic bundles into a single statistic instance.

        Args:
            a: First statistic bundle.
            b: Second statistic bundle.
        """
        # make a new instance
        instance = cls()

        # populate using downstream logic
        for space in cls.spaces:
            instance._values[space] = cls._merge(a, b, space)

        return instance

    @classmethod
    @abstractmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        """
        Internal implementation of statistic merging logic.

        Args:
            a: First statistic bundle.
            b: Second statistic bundle.
            space: Transform space in which to express the values.
        """
        ...

    def align_to(self, indices: ArrayI) -> None:
        """
        Reorder internal statistic data in-place to match a given index ordering.

        Args:
            indices: Index array defining the desired column reordering.
        """
        if indices.ndim != 1:
            raise ValueError(f"{type(self).__name__}.align_to: indices must be 1D, got shape {indices.shape}")

        for space, array in list(self._values.items()):
            # make sure shape matches
            if array.shape[-1] != len(indices):
                raise ValueError(f"{type(self).__name__}.align_to, {space}: last axis does not match indices")

            # apply reordering
            self._values[space] = array[..., indices]

    def filter_to(self, mask: ArrayB) -> None:
        """
        Filter internal statistic data in-place given a boolean mask.

        Args:
            mask: Boolean mask defining the desired columns to keep.
        """
        if mask.ndim != 1:
            raise ValueError(f"{type(self).__name__}.filter_to: mask must be 1D, got shape {mask.shape}")

        for space, array in list(self._values.items()):
            # make sure shape matches
            if array.shape[-1] != len(mask):
                raise ValueError(f"{type(self).__name__}.filter_to, {space}: last axis does not match mask")

            # apply filter
            self._values[space] = array[..., mask]

    def to_struct(self) -> StatisticStruct:
        """Serialize the statistic into a plain Python structure."""
        return {space.value: array for space, array in self._values.items()}

    @classmethod
    def from_struct(cls, struct: StatisticStruct) -> Self:
        """
        Reconstruct a statistic instance from a serialized structure.

        Args:
            struct: Dictionary representation of a statistic.
        """
        # build new instance
        instance = cls()

        # make sure array exists for each required space
        missing = [s for s in cls.spaces if s.value not in struct]
        if missing:
            raise ValueError(f"{cls.__name__}.from_struct: struct missing spaces {missing}")

        # rebuild from saved struct
        for space_value, data in struct.items():
            space = TransformSpace(space_value)
            if space not in cls.spaces:
                raise ValueError(f"{cls.__name__}.from_struct: struct contains unsupported space {space}")

            instance._values[space] = data

        return instance

    @staticmethod
    def _ensure_shape(array: ArrayF) -> ArrayF:
        """
        Ensure an array has a supported shape for statistic operations.

        Accepts 1D arrays (reshaped to column vectors) or 2D arrays, and
        raises otherwise.

        Args:
            array: Input array to validate and reshape if necessary.
        """
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 2:
            return array
        raise ValueError(f"Expected 1D or 2D array, got shape {array.shape}")
