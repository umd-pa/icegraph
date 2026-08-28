# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterator, Self, overload
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["SegmentedTensor", "SegmentLayout"]


@dataclass(frozen=True, eq=False)
class _DeviceTensors:
    """Device-resident tensors that move with data."""
    ids:    Tensor
    widths: Tensor

    @property
    def device(self) -> torch.device:
        return self.ids.device  # all tenors on same device, just get one of them

    def to(self, device: torch.device | str | int, *, non_blocking: bool = False) -> Self:
        # ids/widths are index tensors, never dtype-cast
        return dataclasses.replace(
            self,
            ids=self.ids.to(device, non_blocking=non_blocking),
            widths=self.widths.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, eq=False)
class SegmentLayout:
    offsets:            Tensor  # [L + 1], CPU tensor
    names:              list[str]  # [L]
    _device_tensors:    _DeviceTensors

    def __post_init__(self) -> None:
        # offsets must be CPU
        if self.offsets.device.type != "cpu":
            raise ValueError(
                f"offsets must be a CPU tensor, got device {self.offsets.device}"
            )

        if len(self.names) != len(self.offsets) - 1:
            raise ValueError(
                f"len(names) ({len(self.names)}) must equal len(offsets) - 1 ({len(self.offsets) - 1})"
            )

    @property
    def device(self) -> torch.device:
        return self._device_tensors.device

    @property
    def ids(self) -> Tensor:
        """Returns a logical index for each physical index as a Tensor on device."""
        return self._device_tensors.ids

    @property
    def widths(self) -> Tensor:
        """Returns the width of each segment as a Tensor on device."""
        return self._device_tensors.widths

    @property
    def full_width(self) -> Tensor:
        """Returns the full layout width as a scalar tensor on CPU."""
        return self.offsets[-1]

    @classmethod
    def build(cls, names: list[str], offsets: Tensor) -> Self:
        """Build the layout from offsets and names."""
        # build widths
        widths = (offsets[1:] - offsets[:-1])  # [L]

        # build ids
        ids = torch.repeat_interleave(  # [V], e.g. [2,3] -> [0,0,1,1,1]
            torch.arange(widths.numel()), widths,
        )

        # package and build
        _device_tensors = _DeviceTensors(widths=widths, ids=ids)
        return cls(offsets=offsets, names=names, _device_tensors=_device_tensors)

    def to(self, device: torch.device | str | int, *, non_blocking: bool = False) -> Self:
        return dataclasses.replace(
            self, _device_tensors=self._device_tensors.to(device, non_blocking=non_blocking)
        )


@dataclass(frozen=True, eq=False)
class SegmentedTensor(Sequence[Tensor]):
    data:   Tensor
    layout: SegmentLayout

    def __post_init__(self) -> None:
        if self.layout.full_width != self.data.shape[1]:
            raise ValueError(
                f"offsets[-1] ({int(self.layout.full_width)}) must equal data.shape[1] ({self.data.shape[1]})"
            )

        if self.layout.device != self.data.device:
            raise ValueError(
                f"Device mismatch: {type(self.layout).__name__} on {self.layout.device}, data on {self.device}"
            )

    def __len__(self) -> int:
        # number of logical indices
        return len(self.offsets) - 1

    @overload
    def __getitem__(self, index: int) -> Tensor: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Tensor]: ...

    def __getitem__(self, index: int | slice) -> Tensor | Sequence[Tensor]:
        # dont allow slicing
        if not isinstance(index, int):
            raise TypeError(f"{type(self).__name__} does not support slicing.")

        start = int(self.offsets[index])
        stop  = int(self.offsets[index + 1])
        return self.data[:, start:stop]

    def __iter__(self) -> Iterator[Tensor]:
        # iterate over logical indices along dim 1
        for i in range(len(self)):
            yield self[i]

    def block(self, names: Sequence[str], *, contiguous: bool = False) -> Tensor:
        """Return the columns of the named logical segments, concatenated along dim 1.

        A single segment is one span of ``data``, so it comes back as a view; pass
        ``contiguous`` when the consumer rejects a strided tensor. Concatenating
        several segments always materializes, and the flag is redundant there.
        """
        if not names:
            raise ValueError(f"{type(self).__name__} requires at least one segment name.")

        missing = [name for name in names if name not in self.names]
        if missing:
            raise KeyError(
                f"{type(self).__name__}: unknown segment(s) {missing}; "
                f"available: {self.names}."
            )

        if len(names) == 1:
            block = self[self.names.index(names[0])]
            return block.contiguous() if contiguous else block

        return torch.cat([self[self.names.index(name)] for name in names], dim=1)

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def offsets(self) -> Tensor:
        return self.layout.offsets

    @property
    def widths(self) -> Tensor:
        return self.layout.widths

    @property
    def ids(self) -> Tensor:
        return self.layout.ids

    @property
    def names(self) -> list[str]:
        return self.layout.names

    def detach(self) -> Self:
        return dataclasses.replace(self, data=self.data.detach())

    def to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None, *,
        non_blocking: bool = False,
    ) -> Self:
        layout = self.layout if device is None else self.layout.to(device, non_blocking=non_blocking)

        data = self.data
        if device is not None:
            data = data.to(device=device, non_blocking=non_blocking)
        if dtype is not None:
            data = data.to(dtype=dtype)

        return dataclasses.replace(self, data=data, layout=layout)
