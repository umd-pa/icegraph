# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, overload, Literal, Any, Callable, Self
from abc import ABC
from functools import cached_property

import torch
from torch import Tensor
from torch.nn import Module

from icegraph.common.plugins import Plugin

from .types import ComponentContext

__all__ = ["Component"]

import logging
logger = logging.getLogger(__name__)


C = TypeVar("C")
X = TypeVar("X", bound=ComponentContext)


class Component(Plugin[C, X], Module, ABC):
    _device: torch.device

    @cached_property
    def _dynamic_buffers(self) -> list[str]:
        return []

    def register_dynamic_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        # save as a dynamic buffer
        self._dynamic_buffers.append(name)

        # register as normal
        self.register_buffer(name, tensor, persistent)

    def _internal_build_before_user_build(self) -> None:
        # module starts on cpu
        self._device = torch.device("cpu")

    def _internal_build_after_user_build(self) -> None:
        self.register_load_state_dict_pre_hook(self._preload_dynamic_buffers)

    @property
    def device(self) -> torch.device:
        """The device currently associated with this component."""
        return self._device

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        probe = torch.empty(0, device=self._device)
        moved_probe = fn(probe)
        self._device = moved_probe.device

        return super()._apply(fn, recurse=recurse)

    @overload
    def load_buffer(
            self,
            name: str,
            /,
            *,
            allow_empty: bool = True,
            allow_none: Literal[False] = False,
    ) -> Tensor:
        ...

    @overload
    def load_buffer(
            self,
            name: str,
            /,
            *,
            allow_empty: bool = True,
            allow_none: Literal[True],
    ) -> Tensor | None:
        ...

    @overload
    def load_buffer(
            self,
            name: str,
            /,
            *,
            allow_empty: bool = True,
            allow_none: bool,
    ) -> Tensor | None:
        ...

    def load_buffer(
            self,
            name: str,
            /,
            *,
            allow_empty: bool = True,
            allow_none: bool = False
    ) -> Tensor | None:
        """
        Returns a registered buffer by name with optional validation.

        Args:
            name (str): The name of the registered buffer to load.
            allow_empty (bool): Whether to allow tensors with zero elements.
            allow_none (bool): Whether to allow buffers whose value is None.
        """

        # load buffer
        try:
            buffer: Tensor | None = self._buffers[name]
        except KeyError:
            raise RuntimeError(
                f"Buffer '{name}' does not exist for component '{type(self).__name__}'"
            )

        # check if empty
        if buffer is None:
            if allow_none:
                return None

            raise RuntimeError(
                f"Buffer '{name}' is None for component '{type(self).__name__}'."
            )

        # check if initialized
        if buffer.numel() == 0:
            if allow_empty:
                return buffer

            raise RuntimeError(
                f"Buffer '{name}' is empty for component '{type(self).__name__}'."
            )

        return buffer

    @staticmethod
    def _preload_dynamic_buffers(
        module: Component[Any, Any],
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # resize each buffer placeholder to match for state dict reload
        for name in module._dynamic_buffers:
            key = prefix + name

            # skip if not being loaded
            if key not in state_dict:
                continue

            loaded = state_dict[key]
            current = module._buffers.get(name)

            # if nothing present or not correct shape, resize
            if current is None or current.shape != loaded.shape:
                # maintain same device if possible
                device = loaded.device if current is None else current.device

                module.register_buffer(
                    name,
                    torch.empty_like(loaded, device=device),
                )
