# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Iterable
from dataclasses import dataclass

from torch.nn import Parameter

from ..types import ComponentContext

__all__ = ["OptimizerContext"]


# optimizer needs access to model params
@dataclass(frozen=True)
class OptimizerContext(ComponentContext):
    model_params: Iterable[Parameter]
