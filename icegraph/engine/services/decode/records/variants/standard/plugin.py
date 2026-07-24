# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar
import torch
from torch import Tensor

from jaxtyping import Int

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

    def extract(self, record: Record, key: str) -> Tensor | None:
        data = record.data.get(key)

        # if key not present in the record, return None
        if data is None:
            return None

        # try to convert it to a tensor, if it raises, try to be helpful
        try:
            tensor = torch.tensor(data)
        except Exception as e:
            raise RuntimeError(
                f"{type(self).__name__}.extract: could not convert loaded "
                f"data (type={type(data).__name__}) to a torch.Tensor"
            ) from e
        
        return tensor

    def _extract_edge_index(self, record: Record, key: str) -> Int[Tensor, "2 E"] | None:
        tensor = self.extract(record, key)

        if tensor is None:
            return None

        # need to ensure contiguity after transpose or downstream will explode
        return tensor.t().contiguous()
