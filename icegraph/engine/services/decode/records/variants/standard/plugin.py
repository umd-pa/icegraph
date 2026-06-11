# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.record import Record

from ...decoder import RecordDecoder

from .config import StandardRecordDecoderConfig

__all__ = ["StandardRecordDecoder"]


class StandardRecordDecoder(RecordDecoder[StandardRecordDecoderConfig]):
    name: ClassVar[str] = "standard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StandardRecordDecoderConfig:
        return StandardRecordDecoderConfig(**config)

    def extract(self, record: Record, key: str) -> Tensor:
        array = record.data.get(key)

        if array is None:
            return torch.empty((0,))
        
        return torch.tensor(array)
