# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TypeAlias, Any

import numpy.typing as npt
import numpy as np

__all__ = [
    "ArrayG",
    "ArrayF",
    "ArrayB",
    "ArrayI",
    "ArrayUI",

    "ArrayF64",
    "ArrayF32",
    "ArrayF16",

    "ArrayI64",
    "ArrayI32",
    "ArrayI16",
    "ArrayI8",

    "ArrayUI64",
    "ArrayUI32",
    "ArrayUI16",
    "ArrayUI8",
]


# general arrays
ArrayG:     TypeAlias = npt.NDArray[np.generic]
ArrayF:     TypeAlias = npt.NDArray[np.floating[Any]]
ArrayB:     TypeAlias = npt.NDArray[np.bool_]
ArrayI:     TypeAlias = npt.NDArray[np.intp]
ArrayUI:    TypeAlias = npt.NDArray[np.uintp]

# dtype specific arrays
ArrayF64:   TypeAlias = npt.NDArray[np.float64]
ArrayF32:   TypeAlias = npt.NDArray[np.float32]
ArrayF16:   TypeAlias = npt.NDArray[np.float16]

ArrayI64:   TypeAlias = npt.NDArray[np.int64]
ArrayI32:   TypeAlias = npt.NDArray[np.int32]
ArrayI16:   TypeAlias = npt.NDArray[np.int16]
ArrayI8:    TypeAlias = npt.NDArray[np.int8]

ArrayUI64:  TypeAlias = npt.NDArray[np.uint64]
ArrayUI32:  TypeAlias = npt.NDArray[np.uint32]
ArrayUI16:  TypeAlias = npt.NDArray[np.uint16]
ArrayUI8:   TypeAlias = npt.NDArray[np.uint8]
