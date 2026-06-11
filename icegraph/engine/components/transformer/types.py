# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass

from icegraph.common.transforms import TransformSpace

from ..types import ContractComponentContext

__all__ = ["TransformerContext", "TransformerSpec"]


@dataclass(frozen=True)
class TransformerContext(ContractComponentContext):
    pass


@dataclass(frozen=True)
class TransformerSpec:
    space:  TransformSpace
    base:   int
