# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import hashlib
from typing import Any, ClassVar
import cbor2

import numpy as np

__all__ = ["CBORBlake2B"]


class CBORBlake2B:
    name: ClassVar[str] = "cbor2(canonical)+blake2b-256"

    def __call__(self, obj: Any) -> str:
        """
        Fast deterministic hash for nested Python containers using canonical CBOR + hashlib.
        Returns 256 bit digest.
        """
        norm = self._to_cborable(obj)
        payload = cbor2.dumps(norm, canonical=True)  # stable bytes
        return hashlib.blake2b(payload, digest_size=32).hexdigest()

    def _to_cborable(self, x: Any) -> Any:
        """Normalize unsupported types for CBOR."""
        if isinstance(x, dict):
            return {self._to_cborable(k): self._to_cborable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._to_cborable(v) for v in x]
        if isinstance(x, (set, frozenset)):
            return tuple(sorted((self._to_cborable(v) for v in x), key=repr))
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return self._to_cborable(x.tolist())
        return x
