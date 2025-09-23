# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .callbacks import (
    ExportCallback,
    ConsoleCallback,
    TensorBoardCallback,
    RegressionMetricsCallback,
    MulticlassMetricsCallback
)
from .normalizers import MinMaxNormalizer, ZScoreNormalizer, resolve_normalizer

# callbacks
ExportCallback.__module__ = __name__
ConsoleCallback.__module__ = __name__
TensorBoardCallback.__module__ = __name__
RegressionMetricsCallback.__module__ = __name__
MulticlassMetricsCallback.__module__ = __name__

# normalizers
MinMaxNormalizer.__module__ = __name__
ZScoreNormalizer.__module__ = __name__
resolve_normalizer.__module__ = __name__

__all__ = [
    "ExportCallback",
    "ConsoleCallback",
    "TensorBoardCallback",
    "RegressionMetricsCallback",
    "MulticlassMetricsCallback",
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "resolve_normalizer"
]
