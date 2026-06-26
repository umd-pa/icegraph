# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from functools import cached_property

import numpy as np

from icegraph.typing.common import ArrayI
from icegraph.common.data import DataRole

__all__ = ["LoaderSpec"]


@dataclass(frozen=True)
class LoaderSpec:
    _buffer_keys:   tuple[str, bytes]
    _exclude_roles: frozenset[DataRole]

    @classmethod
    def make(cls, keys: ArrayI, *, exclude_roles: Sequence[DataRole] | None = None) -> LoaderSpec:
        if keys.ndim != 1:
            raise ValueError(f"{cls.__name__}.make, 'keys' must have ndim 1, got ndim {keys.ndim}.")

        if not np.issubdtype(keys.dtype, np.integer):
            raise ValueError(f"{cls.__name__}.make, 'keys' must be an integer array, got dtype {keys.dtype.str}.")

        # normalize exclude_roles
        exclude_roles = exclude_roles if exclude_roles is not None else []

        # convert to contiguous array for consistency
        keys = np.ascontiguousarray(keys)
        return cls((keys.dtype.str, keys.tobytes()), frozenset(exclude_roles))

    @cached_property
    def exclude_roles(self) -> frozenset[DataRole]:
        return self._exclude_roles

    @cached_property
    def keys(self) -> ArrayI:
        dtype, buffer = self._buffer_keys
        return np.frombuffer(buffer, dtype=dtype).copy()