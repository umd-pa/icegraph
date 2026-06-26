# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cached_property
from typing import TYPE_CHECKING, TypeVar, Any

import torch

from icegraph.engine.components.types import ComponentContract
from icegraph.common.plugins import Plugin
from icegraph.common.engine import ComponentKind

from .types import PolicyContext, TaskSpec

if TYPE_CHECKING:
    from torch import Tensor

    from icegraph.engine.components import Component

    from icegraph.engine.components.loss import LossFunction
    from icegraph.engine.components.model import Model
    from icegraph.engine.components.transformer import Transformer
    from icegraph.engine.components.optimizer import Optimizer
    from icegraph.engine.components.normalizer import Normalizer

__all__ = ["Policy"]


C = TypeVar("C")


class Policy(Plugin[C, PolicyContext], ABC):
    """Base class for task-specific training strategies."""

    def build(self) -> None:
        return

    def get_contract_for(self, kind: ComponentKind) -> ComponentContract[Component[Any]]:
        return {
            ComponentKind.MODEL: self.model_contract,
            ComponentKind.NORMALIZER: self.normalizer_contract,
            ComponentKind.TRANSFORMER: self.transformer_contract,
            ComponentKind.OPTIMIZER: self.optimizer_contract,
            ComponentKind.LOSS: self.loss_contract
        }[kind]()

    def model_contract(self) -> ComponentContract[Model[Any]]:
        # model requires output offsets
        contract = ComponentContract(
            kwargs=dict(
                out_offsets=self.task_spec.out_offsets
            ),
            validator=self.model_validator,
            forward_validator=self.model_forward_validator
        )

        return contract

    def model_validator(self, model: Model[Any]) -> None:
        return None

    def model_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        expected_c = int(self.task_spec.out_offsets[-1])

        # structural checks, metadata only, no GPU sync
        if t.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}: expected model output of rank 2 "
                f"[B, C], got rank {t.ndim} (shape {tuple(t.shape)})."
            )
        if t.shape[-1] != expected_c:
            raise ValueError(
                f"{type(self).__name__}: expected {expected_c} output channels "
                f"(out_channels), got {t.shape[-1]} (shape {tuple(t.shape)})."
            )

        # forces a sync, debug only
        if debug:
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"{type(self).__name__}: model produced non-finite logits "
                    f"(inf/nan)."
                )

    def loss_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(),
            validator=self.loss_validator,
            forward_validator=self.loss_forward_validator
        )

        return contract

    def loss_validator(self, loss: LossFunction[Any]) -> None:
        return None

    def loss_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def normalizer_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(
                norm_targets=self.task_spec.norm_targets
            ),
            validator=self.normalizer_validator,
            forward_validator=self.normalizer_forward_validator
        )

        return contract

    def normalizer_validator(self, normalizer: Normalizer[Any]) -> None:
        return None

    def normalizer_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def transformer_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(
                norm_targets=self.task_spec.norm_targets
            ),
            validator=self.transformer_validator,
            forward_validator=self.transformer_forward_validator
        )

        return contract

    def transformer_validator(self, transformer: Transformer[Any]) -> None:
        return None

    def transformer_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        return None

    def optimizer_contract(self) -> ComponentContract:
        contract = ComponentContract(
            kwargs=dict(),
            validator=self.transformer_validator,
            forward_validator=None
        )

        return contract

    def optimizer_validator(self, optimizer: Optimizer[Any]) -> None:
        return None

    @cached_property
    def task_spec(self) -> TaskSpec:
        return self._build_task_spec()

    @abstractmethod
    def _build_task_spec(self) -> TaskSpec:
        ...
