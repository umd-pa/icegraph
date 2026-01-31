# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from torch.nn import Module

from ..module import TrainerModule
from ..params import ModuleParams

from .context import ModelContext

__all__ = ["Model"]


class Model(TrainerModule[ModelContext]):
    name: str

    def __init__(self, params: ModuleParams) -> None:
        super().__init__(params)
