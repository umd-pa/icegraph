# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Dict, Any, List, Union, Sequence

from icegraph.trainer.callbacks import Callback
from icegraph.trainer.normalizers import Normalizer


class CallbackRegistryMixin:

    @dataclass
    class CallbackSpec:
        callback: Type[Callback]
        kwargs: Dict[str, Any] = field(default_factory=dict)

        def __call__(self, ctx: CallbackRegistryMixin) -> Callback:
            cb = self.callback(**self.kwargs)

            if not isinstance(cb, Callback):
                raise TypeError("callback must be a subclass of 'Callback'")

            if isinstance(cb, Normalizer):
                raise TypeError("Cannot register a Normalizer as a standard callback.")

            # run on_init for the callback
            cb.on_init(ctx)

            return cb

    def __init__(self) -> None:
        super().__init__()

        self.callbacks: List[Callback] = []

    def register_callback(
            self,
            callbacks: Union[
                CallbackRegistryMixin.CallbackSpec,
                Callback,
                Sequence[Union[CallbackRegistryMixin.CallbackSpec, Callback]]
            ]
    ) -> None:
        """Add one or more callback(s) to the registry."""
        if isinstance(callbacks, (list, tuple, set)):
            for cb in callbacks:
                self.register_callback(cb)
            return

        if isinstance(callbacks, CallbackRegistryMixin.CallbackSpec):
            cb: Callback = callbacks(self)

            # verify the callback is compatible with the current task (if a task exists)
            if hasattr(cb, "COMPATIBLE") and hasattr(self, "strategy"):
                if not self.strategy.task in cb.COMPATIBLE:
                    raise RuntimeError(
                        f"Registered callback {callbacks.__name__} is not compatible with task {self.strategy.task}"
                    )
        elif isinstance(callbacks, Callback):
            cb: Callback = callbacks
        else:
            raise TypeError("Callback must be a Callback instance or a factory returning one.")

        self.callbacks.append(cb)

    def _fire(self, hook_name: str, *args, **kwargs) -> None:
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            *args: Positional arguments to forward into the callback.
            **kwargs: Keyword arguments to forward into the callback.
        """
        for cb in self.callbacks:
            getattr(cb, hook_name)(self, *args, **kwargs)
