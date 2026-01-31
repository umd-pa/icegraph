# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC
from typing import final, TYPE_CHECKING, TypeVar, Generic

from .types import AttachContext, Params

if TYPE_CHECKING:
    from torch.nn import Module

__all__ = ["TrainerModule"]


T = TypeVar("T", bound=AttachContext)

class TrainerModule(ABC, Generic[T]):

    def __init__(self, params: Params | None) -> None:
        # stash params
        self.params: Params = params if params is not None else Params.empty()

        # cache for context
        self._ctx: T | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # if torch Module, it already has state_dict/load_state_dict
        if issubclass(cls, Module):
            return

        # otherwise enforce both
        if "state_dict" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} must implement state_dict()")
        if "load_state_dict" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} must implement load_state_dict()")

    @final
    def attach(self, ctx: T) -> None:
        """Attach this module given a context object, should be run by the trainer or inference engine."""
        if self.is_attached:
            raise RuntimeError(f"{type(self).__name__} is already attached.")

        # stash context
        self._ctx = ctx

        # call hook for downstream post-attach logic
        self.on_attach(ctx)

    def on_attach(self, ctx: T) -> None:
        """Called just after module is attached."""
        return

    @property
    @final
    def is_attached(self) -> bool:
        """Whether this module has been attached."""
        return self._ctx is not None
