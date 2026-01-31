# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from icegraph.trainer.callbacks.base.accumulator import Accumulator, AccumulatorStore
from ..reducer import Reducer

__all__ = ["HistogramReducer"]


class HistogramReducer(Reducer, ABC):

    def _init_accumulator(self, device: torch.device) -> AccumulatorStore:
        # initialize each accumulator, construct an accumulator store and return
        return AccumulatorStore(
            accumulators=self._build_accumulators(device)
        )

    @abstractmethod
    def _build_accumulators(self, device: torch.device) -> dict[str, Accumulator]:
        ...
