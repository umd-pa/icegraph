# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cached_property
from typing import TypeVar, final, Any
from collections.abc import Mapping

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor, SegmentLayout
from icegraph.common.data import DataRole

from ..component import Component

__all__ = ["Model"]


C = TypeVar("C")


class Model(Component[C], ABC):

    _head_names: list[str] | None

    def build(self) -> None:
        self.register_dynamic_buffer("_in_channels", None)
        self.register_dynamic_buffer("_out_offsets", None)

        # cache for head names, needs to travel with the model
        self._head_names = None

    def get_extra_state(self) -> dict[str, Any]:
        return {"heads": self._head_names}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._head_names = state.get("heads")

    def on_preload(self, state_dict: Mapping[str, Any]) -> None:
        # grab offsets from the checkpoint so out_layout resolves at
        # construction time without a contract
        offsets = state_dict.get("_out_offsets")
        if offsets is not None:
            self.register_buffer("_out_offsets", offsets.cpu())

        extra = state_dict.get("_extra_state")
        if extra is not None:
            self._head_names = extra.get("heads")

    @cached_property
    def head_names(self) -> list[str]:
        if self._head_names is not None:
            return self._head_names

        try:
            decoder = self._ctx.services.require("decode", required_by=type(self))
        except KeyError as e:
            raise RuntimeError(
                f"{type(self).__name__} could not resolve head names: the 'decode' "
                f"service is unavailable and no 'heads' were loaded from the state dict. "
                f"Either load a checkpoint whose extra state contains head names, "
                f"or make the decode service available."
            ) from e

        self._head_names = decoder.get_columns(DataRole.TARGETS)
        if self._head_names is None:
            raise RuntimeError(
                f"{type(self).__name__} could not resolve head names: "
                f"{type(decoder).__name__}.get_columns(DataRole.TARGETS) returned None."
            )

        return self._head_names

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
        return int(self.out_layout.full_width)

    @cached_property
    def out_offsets(self) -> Tensor:
        try:
            out_offsets = self.load_buffer("_out_offsets", allow_none=True)

            if out_offsets is None:
                contract = self._ctx.contract
                if contract is None:
                    raise RuntimeError(
                        f"{type(self).__name__} could not resolve 'out_offsets': a contract "
                        f"has not been provided and no 'out_offsets' was loaded from the state dict. "
                        f"Either load a checkpoint that contains 'out_offsets', "
                        f"or provide a contract."
                    )

                out_offsets = contract.kwargs.get("out_offsets")

                # ensure offset was provided
                if out_offsets is None:
                    raise ValueError(
                        "Model requires output offsets provided in the contract under 'kwargs.out_offsets'."
                    )

                # ensure correct type
                if not isinstance(out_offsets, Tensor):
                    raise TypeError(
                        f"Offsets must be of type {Tensor.__name__}, got {type(out_offsets).__name__}."
                    )

                self.register_buffer("_out_offsets", out_offsets)

            return out_offsets.cpu()  # offsets always on cpu

        # need to catch and reraise any exception here, or torch throws some cryptic error that explains nothing
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve {type(self).__name__}.out_offsets"
            ) from e

    @cached_property
    def out_layout(self) -> SegmentLayout:
        offsets = self.out_offsets

        # get services
        state = self._ctx.services.require("state", required_by=type(self))

        # built once on init and move to accelerator
        layout = SegmentLayout.build(
            names=self.head_names,
            offsets=offsets
        ).to(state.device)

        return layout

    @final
    def forward(
            self,
            t: SegmentedTensor,
            /,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor | None = None
    ) -> SegmentedTensor:
        out = self.forward_pass(t, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        # internal validation
        if out.shape[1] != self.out_channels:
            raise ValueError(
                f"Model output must satisfy shape[1] == out_channels, "
                f"got shape[1]={out.shape[1]} and out_channels={self.out_channels}"
            )

        # run contract validator
        self._run_forward_validator(out)

        # build segmented output
        return SegmentedTensor(data=out, layout=self.out_layout)

    @abstractmethod
    def forward_pass(
            self,
            t: SegmentedTensor,
            /,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor | None
    ) -> Tensor:
        ...
