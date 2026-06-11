# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Literal, TypeAlias
from dataclasses import dataclass

from ..types import ContractComponentContext

__all__ = ["ReductionType", "LossContext"]


ReductionType: TypeAlias = Literal["mean", "sum"]


@dataclass(frozen=True)
class LossContext(ContractComponentContext):
    pass
