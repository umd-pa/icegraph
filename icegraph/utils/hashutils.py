# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import hashlib
from typing import Any, TypeAlias
import cbor2

import numpy as np

__all__ = ["stable_hash_blake2b"]


Primitive: TypeAlias = int | float | str | bool | None
NumpyScalar: TypeAlias = np.integer | np.floating

CBORHashable: TypeAlias = (
    Primitive
    | np.integer
    | np.floating
    | np.ndarray
    | list["CBORHashable"]
    | tuple["CBORHashable", ...]
    | set["CBORHashable"]
    | frozenset["CBORHashable"]
    | dict[str, "CBORHashable"]
)


def _to_cborable(x: Any) -> Any:
    """Normalize unsupported types for CBOR."""
    if isinstance(x, dict):
        return {_to_cborable(k): _to_cborable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_cborable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return tuple(sorted((_to_cborable(v) for v in x), key=repr))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return _to_cborable(x.tolist())
    return x


def stable_hash_blake2b(obj: CBORHashable) -> str:
    """
    Fast deterministic hash for nested Python containers using canonical CBOR + hashlib.
    Returns 256 bit digest.
    """
    norm = _to_cborable(obj)
    payload = cbor2.dumps(norm, canonical=True)  # stable bytes
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


stable_hash_blake2b.name = "cbor2(canonical)+blake2b-256"

