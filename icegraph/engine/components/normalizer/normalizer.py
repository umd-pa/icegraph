# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final, Any
from abc import abstractmethod, ABC
import dataclasses
from functools import cached_property

import torch
from torch import Tensor

from icegraph.common.data import ColumnarRole, DataRole
from icegraph.common.tensors import SegmentedTensor

from ..component import Component

__all__ = ["Normalizer"]


C = TypeVar("C")


class Normalizer(Component[C], ABC):

    def _internal_build_after_user(self) -> None:
        super()._internal_build_after_user()

        # -1 = unresolved, 0 = False, 1 = True
        self.register_buffer("_norm_targets", torch.tensor(-1, dtype=torch.long))
        self._norm_targets: Tensor

    @cached_property
    def norm_targets(self) -> bool:
        value = int(self._norm_targets.item())
        if value in (0, 1):
            return bool(value)

        contract = self._ctx.contract
        if contract is None:
            raise RuntimeError(
                f"{type(self).__name__} could not resolve 'norm_targets' flag: a contract "
                f"has not been provided and no 'norm_targets' was loaded from the state dict. "
                f"Either load a checkpoint that contains 'norm_targets', "
                f"or provide a contract."
            )

        norm_targets = contract.kwargs.get("norm_targets")
        if not isinstance(norm_targets, bool):
            raise RuntimeError(
                f"{type(self).__name__} could not resolve 'norm_targets': "
                f"contract.kwargs.get('norm_targets') should return bool, got {type(norm_targets).__name__}."
            )

        # write back so the resolved value gets serialized
        self._norm_targets.fill_(int(norm_targets))
        return norm_targets

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        # the buffer may change on load, drop any cached values beforehand
        vars(self).pop("norm_targets", None)
        super()._load_from_state_dict(*args, **kwargs)

    @final
    @torch.no_grad()
    def forward(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool = False) -> SegmentedTensor:
        """Forward pass through the normalizer."""
        # skip if norm is not required for targets
        if role == DataRole.TARGETS:
            if not self.norm_targets:
                return t

        out = self.normalize(t, role, inverse=inverse)

        # internal validation
        if out.shape != t.data.shape:
            raise ValueError(
                f"Normalizer is a value map only, tensors cannot be reshaped. "
                f"Expected shape {t.data.shape}, got {out.shape}"
            )

        # only check in debug mode, since this forces a sync
        if self._ctx.debug:
            if not torch.isfinite(out).all():
                raise ValueError(
                    f"Transformer produced non-finite values (inf/nan); "
                    f"check domain of inputs to the {type(self).__name__} mapping "
                    f"(e.g. log of non-positive values)."
                )

        # run contract validator
        self._run_forward_validator(out)

        return dataclasses.replace(t, data=out)

    @abstractmethod
    def normalize(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
        ...
