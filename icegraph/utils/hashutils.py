# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import hashlib
from typing import Any, Union, FrozenSet, Set, List, Dict, Tuple
import cbor2

import numpy as np

def _to_cborable(x: Any):
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


def stable_hash_cbor(
        obj: Union[Dict, List, Tuple, Set, FrozenSet, np.integer, np.floating, np.ndarray],
        digest_size: int = 32
) -> str:
    """
    Fast deterministic hash for nested Python containers using canonical CBOR + hashlib.
    """
    norm = _to_cborable(obj)
    payload = cbor2.dumps(norm, canonical=True)  # stable bytes
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
