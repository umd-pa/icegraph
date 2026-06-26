# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Generic, Any

from .spec import CallbackSpec
from .callback import Callback
from . import context

if TYPE_CHECKING:
    from icegraph.engine import Engine

__all__ = ["CallbackManager"]

# module logger
import logging
logger = logging.getLogger(__name__)


E = TypeVar("E", bound="Engine[Any]")


class CallbackManager(Generic[E]):

    def __init__(self, engine: E) -> None:
        # cache engine
        self.engine:    E               = engine
        self.callbacks: list[Callback]  = []

    def register(self, spec: CallbackSpec) -> None:
        """Register a callback spec to the manager."""
        if not isinstance(spec, CallbackSpec):
            raise TypeError("'spec' must be an instance of CallbackSpec.")

        # initialize the callback
        callback = spec()

        # register
        callback.on_init(context.InitContext(engine=self.engine))
        self.callbacks.append(callback)

        logger.debug("registered callback %s", type(callback).__name__)

    def fire(self, hook_name: str, ctx: context.Context[E]) -> None:
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            ctx: Context to forward into each callback.
        """
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)

            # raise if a callback was registered without the hook
            if hook is None:
                raise RuntimeError(
                    f"Attempted to call {type(cb).__name__}.{hook_name}(), but could not resolve method."
                )

            hook(ctx)
