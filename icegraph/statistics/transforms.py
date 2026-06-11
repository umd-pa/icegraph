# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import numpy as np

from icegraph.typing.common import ArrayF

__all__ = ["linear_transform", "log_transform", "asinh_transform"]


def linear_transform(x: ArrayF) -> ArrayF:
    return x


def log_transform(x: ArrayF) -> ArrayF:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x, where=(x > 0), out=np.full_like(x, np.nan, dtype=float))


def asinh_transform(x: ArrayF) -> ArrayF:
    return np.arcsinh(x)
