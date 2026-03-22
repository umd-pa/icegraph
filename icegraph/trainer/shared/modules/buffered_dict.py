# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Sequence
from collections.abc import Mapping, Iterator

import torch
from torch import Tensor
from torch.nn import Module


class BufferedDict(Module, Mapping[str, Tensor]):

    __hash__ = Module.__hash__

    def __init__(self) -> None:
        super().__init__()

        # store key list
        self._keys: list[str] = []

    def __setitem__(self, key: str, value: Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError("BufferedDict values must be torch.Tensor")

        name = self._buffer_name(key)

        if key not in self._keys:
            # register new buffer
            self._keys.append(key)
            self.register_buffer(name, value)
        else:
            # overwrite existing buffer value
            setattr(self, name, value)

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, self._buffer_name(key))

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    @staticmethod
    def _buffer_name(key: str) -> str:
        # prefix to avoid collisions with other attributes/methods
        return f"buffer__{key.replace('.', '_')}"

    @classmethod
    def from_dict(cls, d: dict[str, Tensor]) -> Self:
        """Build a buffered dict from a raw python dict."""
        # init instance
        instance = cls()

        # register each key value pair
        for key, value in d.items():
            instance[key] = value

        return instance

    @classmethod
    def from_keys(
            cls, keys: Sequence[str], /, *,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu")
    ) -> Self:
        """Build a buffered dict from a sequence of keys, each value initialized to empty tensor."""
        # init instance
        instance = cls()

        # register each key to None
        for key in keys:
            instance[key] = torch.empty(0, dtype=dtype, device=device)

        return instance
