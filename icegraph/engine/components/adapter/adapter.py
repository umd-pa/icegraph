# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, TypeVar, Any
from functools import cached_property

from icegraph.common.data import GraphBatch, DataRole
from icegraph.common.tensors import SegmentLayout

from ..component import Component
from ..types import ComponentContract

from .types import AdapterContext

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Adapter"]


C = TypeVar("C")


class Adapter(Component[C, AdapterContext], ABC):
    """Base class for task-specific training strategies."""

    _head_names: list[str] | None

    def build(self) -> None:
        # initialize empty buffered tensors
        self.register_dynamic_buffer("_out_offsets", None)

        # cache for head names, needs to travel with the model
        self._head_names = None

    def get_extra_state(self) -> dict[str, Any]:
        return {"heads": self._head_names}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._head_names = state.get("heads")

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

    def model_contract(self) -> ComponentContract:
        # get services
        state = self._ctx.services.require("state", required_by=type(self))

        # get offsets from subclass
        out_offsets = self.get_out_offsets()

        # built once on init and move to accelerator
        layout = SegmentLayout.build(
            names=self.head_names,
            offsets=out_offsets
        ).to(state.device)

        # model requires output layout
        contract = ComponentContract(
            kwargs=dict(
                out_layout=layout
            ),
            forward_validator=self.model_forward_validator
        )

        return contract

    def model_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def loss_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(),
            forward_validator=self.loss_forward_validator
        )

        return contract

    def loss_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def normalizer_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(),
            forward_validator=self.normalizer_forward_validator
        )

        return contract

    def normalizer_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def transformer_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(),
            forward_validator=self.transformer_forward_validator
        )

        return contract

    def transformer_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def get_out_offsets(self) -> Tensor:
        try:
            out_offsets = self.load_buffer("_out_offsets", allow_none=True)

            if out_offsets is None:
                out_offsets = self._compute_out_offsets()
                self.register_buffer("_out_offsets", out_offsets)

            return out_offsets.cpu()

        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve {type(self).__name__}.compute_out_offsets()"
            ) from e

    @abstractmethod
    def _compute_out_offsets(self) -> Tensor:
        ...

    @abstractmethod
    def preprocess_batch(self, batch: GraphBatch) -> GraphBatch:
        ...

    @property
    @abstractmethod
    def use_normalized_targets(self) -> bool:
        ...
