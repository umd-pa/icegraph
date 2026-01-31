# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Type, Any, TYPE_CHECKING

from .callback import Callback
from . import context

if TYPE_CHECKING:
    from icegraph.trainer import Trainer

__all__ = ["CallbackRegistry"]

# module logger
import logging
logger = logging.getLogger(__name__)


class CallbackRegistry:

    @dataclass
    class CallbackSpec:
        callback: Type[Callback]
        kwargs: dict[str, Any] = field(default_factory=dict)

        def __call__(self) -> Callback:
            if inspect.isclass(self.callback) and not issubclass(self.callback, Callback):
                raise TypeError("Expected callback to be a class inheriting from 'Callback'.")

            # initialize
            cb = self.callback(**self.kwargs)

            return cb

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()

        # cache trainer
        self.trainer = trainer

        self.callbacks: list[Callback] = []

    def register_spec(self, spec: CallbackRegistry.CallbackSpec) -> Callback:
        """Register a callback spec to the registry."""
        if not isinstance(spec, CallbackRegistry.CallbackSpec):
            raise TypeError("Parameter 'spec' must be an instance of CallbackSpec.")

        # initialize the callback
        callback: Callback = spec()

        # register the instance to the registry
        self.register_instance(callback)

        # return the callback instance
        return callback

    def register_instance(self, instance: Callback) -> None:
        """Register a callback instance to the registry."""
        # verify the callback is compatible with the current task (if a task exists)
        if hasattr(instance, "COMPATIBLE") and hasattr(self.trainer, "strategy"):
            if not self.trainer.strategy.name in instance.COMPATIBLE:
                raise TypeError(
                    f"Registered callback {instance.__class__.__name__} is not compatible "
                    f"with task {self.trainer.strategy.name}"
                )

        # initialize each instance and register
        instance.on_init(context.InitContext(self.trainer))
        self.callbacks.append(instance)

        logger.debug("registered callback instance %s", instance.__class__.__name__)

    def fire(self, hook_name: str, ctx: context.Context) -> None:
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            ctx: Context to forward into each callback.
        """
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)
