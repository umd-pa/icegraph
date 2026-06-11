# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Callable
from dataclasses import dataclass

from icegraph.common.data import ColumnarRole
from icegraph.engine.components.transformer.types import TransformerSpec

from ..types import ContractComponentContext

__all__ = ["NormalizerContext"]


@dataclass(frozen=True)
class NormalizerContext(ContractComponentContext):
    transformer_spec_list: Callable[[ColumnarRole], list[TransformerSpec]]
