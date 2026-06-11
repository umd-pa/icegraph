# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cached_property
from typing import TypeVar, final

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor, SegmentLayout
from icegraph.common.data import DataRole

from ..component import Component

from .types import ModelContext

__all__ = ["Model"]


C = TypeVar("C")


class Model(Component[C, ModelContext], ABC):

    def build(self) -> None:
        self.register_dynamic_buffer("_in_channels", None)

    @cached_property
    def in_channels(self) -> int:
        cached = self.load_buffer("_in_channels", allow_none=True)
        if cached is not None:
            return int(cached)

        try:
            decoder = self._ctx.services.require("decode", required_by=type(self))
            width = decoder.get_segment_layout(DataRole.FEATURES, torch.device("cpu")).full_width
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve {type(self).__name__}.in_channels"
            ) from e

        self.register_buffer("_in_channels", width)  # scalar tensor, valid buffer
        return int(width)

    @cached_property
    def out_channels(self) -> int:
        # out channels are pulled from output layout
        return int(self._output_layout.full_width)

    @cached_property
    def _output_layout(self) -> SegmentLayout:
        layout = self._ctx.contract.kwargs.get("out_layout")

        # ensure layout was provided
        if layout is None:
            raise ValueError(
                "Model requires an output segment layout provided in the contract under 'kwargs.out_layout'."
            )

        # ensure correct type
        if not isinstance(layout, SegmentLayout):
            raise TypeError(
                f"Layout must be of type {SegmentLayout.__name__}, got {type(layout).__name__}."
            )

        return layout

    @final
    def forward(self, t: SegmentedTensor, /, batch: Tensor | None = None) -> SegmentedTensor:
        out = self.forward_pass(t, batch)

        # run contract validator
        self._ctx.contract.forward_validator(out, self._ctx.debug)

        # internal validation
        if out.shape[1] != self.out_channels:
            raise ValueError(
                f"Model output must satisfy shape[1] == out_channels, "
                f"got shape[1]={out.shape[1]} and out_channels={self.out_channels}"
            )

        # build segmented output
        return SegmentedTensor(data=out, layout=self._output_layout)

    @abstractmethod
    def forward_pass(self, t: SegmentedTensor, /, batch: Tensor | None) -> Tensor:
        ...

