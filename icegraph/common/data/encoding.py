# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
import math

import numpy as np

__all__ = ["flatten", "restore"]


_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_NONFINITE_TAGS: dict[str, str] = {repr(value): tag for tag, value in _NONFINITE.items()}


# reserved key marking an encoded numpy array
_NDARRAY_TAG = "__ndarray__"


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with NONFINITE sentinels so the payload is valid JSON."""
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return _NONFINITE_TAGS[repr(obj)]
    return obj


def flatten(d: Any) -> dict[str, Any]:
    # recurse preserving nesting
    if isinstance(d, dict):
        return {key: flatten(item) for key, item in d.items()}

    # store normalized to numpy array, tagged for reconstruction on read
    array = np.asarray(d)

    return {
        _NDARRAY_TAG: {
            "data": _json_safe(array.tolist()),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    }


def _from_json(obj: Any) -> Any:
    """Restore non-finite floats from NONFINITE sentinels."""
    if isinstance(obj, list):
        return [_from_json(item) for item in obj]
    if isinstance(obj, str):
        return _NONFINITE[obj]
    return obj


def restore(d: Any) -> Any:
    # passthrough for anything already primitive
    if not isinstance(d, dict):
        return d

    # rebuild a tagged array
    if _NDARRAY_TAG in d:
        spec = d[_NDARRAY_TAG]
        dtype = np.dtype(spec["dtype"])
        data = spec["data"]

        # only float arrays carry sentinels; gating protects genuine string data
        if dtype.kind == "f":
            data = _from_json(data)

        return np.asarray(data, dtype=dtype).reshape(spec["shape"])

    # recurse preserving nesting
    return {key: restore(item) for key, item in d.items()}

