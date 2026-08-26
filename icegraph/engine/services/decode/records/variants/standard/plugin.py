# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from icegraph.common.record import Column, RecordBlock

from ...decoder import RecordDecoder

from .config import StandardRecordDecoderConfig

__all__ = ["StandardRecordDecoder"]


class StandardRecordDecoder(RecordDecoder[StandardRecordDecoderConfig]):
    """Reads the standard layout: each key is one column of the block."""

    name: ClassVar[str] = "standard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StandardRecordDecoderConfig:
        return StandardRecordDecoderConfig(**config)

    def extract(self, block: RecordBlock, key: str) -> Column | None:
        return block.columns.get(key)
