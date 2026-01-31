# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Generic, TypeVar

from torch.nn import Module

from ..module import TrainerModule

__all__ = ["Component"]


T = TypeVar("T")

class Component(TrainerModule[T], Module, Generic[T]):
    pass
