from __future__ import annotations

from typing import Iterator, ClassVar, Any
from functools import cached_property
import math

import torch

from ...sampler import Sampler
from .config import Config

__all__ = ["BlockwiseSampler"]


class BlockwiseSampler(Sampler[Config]):
    """Blockwise sampler for DDP."""

    name: ClassVar[str] = "blockwise"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    @cached_property
    def _n(self) -> int:
        return len(self._ctx.dataset)

    @cached_property
    def _block_count(self) -> int:
        return math.ceil(self._n / self.config.block_size)

    def _block_order(self) -> list[int]:
        if not self._ctx.shuffle:
            return list(range(self._block_count))

        g = torch.Generator()
        g.manual_seed(self.seed())
        return torch.randperm(self._block_count, generator=g).tolist()

    def _block_len(self, block: int) -> int:
        start = block * self.config.block_size
        end = min(start + self.config.block_size, self._n)
        return max(0, end - start)

    def _target_num_samples(self, blocks: list[int]) -> int:
        lengths = [0] * self._ctx.num_replicas
        for pos, block in enumerate(blocks):
            lengths[pos % self._ctx.num_replicas] += self._block_len(block)
        return max(lengths, default=0)

    def __len__(self) -> int:
        if self._n == 0:
            return 0

        blocks = self._block_order()
        return self._target_num_samples(blocks)

    def __iter__(self) -> Iterator[int]:
        if self._n == 0:
            return iter(())

        g = torch.Generator()
        g.manual_seed(self.seed())

        # Global block order
        if self._ctx.shuffle:
            blocks = torch.randperm(self._block_count, generator=g).tolist()
        else:
            blocks = list(range(self._block_count))

        target_num_samples = self._target_num_samples(blocks)

        # Assign blocks to this rank
        assigned_blocks = blocks[self._ctx.rank :: self._ctx.num_replicas]

        indices: list[int] = []

        for block in assigned_blocks:
            start = block * self.config.block_size
            end = min(start + self.config.block_size, self._n)
            valid = end - start

            if valid <= 0:
                continue

            # Shuffle within block if requested
            if self._ctx.shuffle:
                offsets = torch.randperm(valid, generator=g).tolist()
            else:
                offsets = list(range(valid))

            for off in offsets:
                indices.append(start + off)

        # Pad shorter ranks so every rank has the same number of samples.
        if len(indices) < target_num_samples:
            pad_source = indices

            # Rare case: this rank received no blocks at all (e.g. world size > block count).
            # Fall back to the global blockwise order for deterministic padding.
            if not pad_source:
                g_pad = torch.Generator()
                g_pad.manual_seed(self.seed())

                if self._ctx.shuffle:
                    pad_blocks = torch.randperm(self._block_count, generator=g_pad).tolist()
                else:
                    pad_blocks = list(range(self._block_count))

                pad_source = []
                for block in pad_blocks:
                    start = block * self.config.block_size
                    end = min(start + self.config.block_size, self._n)
                    valid = end - start

                    if valid <= 0:
                        continue

                    if self._ctx.shuffle:
                        offsets = torch.randperm(valid, generator=g_pad).tolist()
                    else:
                        offsets = list(range(valid))

                    for off in offsets:
                        pad_source.append(start + off)

            needed = target_num_samples - len(indices)
            if needed > 0:
                repeats = math.ceil(needed / len(pad_source))
                indices.extend((pad_source * repeats)[:needed])

        return iter(indices)