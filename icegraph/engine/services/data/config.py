# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, model_validator

import logging
logger = logging.getLogger(__name__)

__all__ = ["DataConfig"]


class DataConfig(BaseModel):
    batch_size:                 int
    chunk_size:                 int
    shuffle_chunks:             bool = False
    buffer_size:                int
    num_workers:                int
    prefetch_factor:            int
    mp_context:                 Literal["spawn", "forkserver"]
    persistent_workers:         int
    buffer_refill_threshold:    float = 0.25

    @model_validator(mode="after")
    def validate_buffer_size(self) -> Self:
        if self.buffer_size < 1:
            raise ValueError("Buffer size must be equal to or greater than 1.")

        if (self.buffer_size * self.chunk_size) < self.batch_size:
            logger.warning(
                f"Buffer size ({self.buffer_size * self.chunk_size} samples) is smaller than batch size ({self.batch_size}), "
                f"there will be no fine mixing."
            )

        elif 1 < (self.buffer_size * self.chunk_size) < self.batch_size * 10:
            logger.warning(
                f"Buffer size ({self.buffer_size * self.chunk_size} samples) is less than 10x batch size "
                + f"({self.batch_size} samples); buffer spans only "
                + f"{(self.buffer_size * self.chunk_size) / self.batch_size:.1f} batches, which may result in poor "
                + "non-local mixing. Aim for a buffer:batch ratio above 10:1."
            )

        if 1 < self.buffer_size < 10:
            logger.warning(
                f"Buffer size ({self.buffer_size} chunks) is less than 10 which may result in poor "
                + "non-local mixing. Aim for a buffer size >10 if memory permits."
            )

        return self

    @model_validator(mode="after")
    def validate_chunk_size(self) -> Self:
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be equal to or greater than 1.")

        return self

    @model_validator(mode="after")
    def validate_buffer_refill_threshold(self) -> Self:
        if not (0 <= self.buffer_refill_threshold < 1):
            raise ValueError("Buffer refill threshold must be a float in the range [0, 1).")

        return self
