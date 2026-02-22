# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar
from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

__all__ = ["Transform"]


class Transform(Module, ABC):
    name: ClassVar[str]

    def __init_subclass__(cls) -> None:
        if getattr(cls, "name", None) is None:
            raise ValueError(f"All subclasses of {cls.__name__} must define the 'name' attribute.")

    def _check(self, t: Tensor, buffer_name: str) -> None:
        # get the buffer
        buffer: Tensor = getattr(self, buffer_name)

        # verify shapes are broadcastable
        if t.size(-1) != buffer.numel():
            raise RuntimeError(
                f"{type(self).__name__}: invalid size for '{buffer}'; "
                f"expected {t.size(-1)}, got {buffer.numel()}."
            )

    def forward(self, t: Tensor, *, inverse: bool = False) -> Tensor:
        if inverse:
            return self.inverse_transform(t)
        return self.transform(t)

    @abstractmethod
    def configure(self, **kwargs) -> None:
        ...

    @abstractmethod
    def inverse_transform(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def transform(self, t: Tensor) -> Tensor:
        ...
