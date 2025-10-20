# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import math

from torch.utils.data import Sampler, get_worker_info
import torch

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


class DistributedBlockShuffleSampler(Sampler[int]):
    """
    Shuffles block order globally, assigns whole blocks to ranks and shuffles indices within blocks.
    """

    def __init__(self, n: int, block_size: int, *, seed: int = 0, drop_last: bool = True):
        super().__init__(None)
        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        # world/rank
        if dist is not None and getattr(dist, "is_available", lambda: False)() and dist.is_initialized():
            self.world = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world = 1
            self.rank = 0

        self.n_total = int(n)
        self.block = int(block_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        # if dropping ignore the tail samples that dont form a full block
        if self.drop_last:
            self.n = (self.n_total // self.block) * self.block
        else:
            self.n = self.n_total

        self.num_blocks = math.ceil(self.n / self.block)
        self._len = self._compute_len()  # per rank sample count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def _global_rng(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + 1_000_003 * self.epoch)  # no rank
        return g

    def _rng(self, extra: int = 0) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + 1_000_003 * self.epoch + 271 * self.rank + extra)
        return g

    def _block_range(self, b: int):
        lo = b * self.block
        hi = min(lo + self.block, self.n)  # self.n may exclude tail
        return lo, hi

    def _pad_or_trim(self, blocks):
        if self.world == 1:
            return blocks
        r = len(blocks) % self.world
        if r == 0:
            return blocks
        # Equalize number of blocks per rank
        if self.drop_last:
            return blocks[: len(blocks) - r]
        need = self.world - r
        return blocks + blocks[:need]

    def _compute_len(self) -> int:
        blocks = list(range(self.num_blocks))
        blocks = self._pad_or_trim(blocks)
        my_blocks = blocks[self.rank :: self.world]
        return sum(self._block_range(b)[1] - self._block_range(b)[0] for b in my_blocks)

    def __iter__(self):
        # Global block shuffle
        blocks = torch.randperm(self.num_blocks, generator=self._global_rng()).tolist()
        blocks = self._pad_or_trim(blocks)
        my_blocks = blocks[self.rank:: self.world]

        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rg = self._rng(extra=97 * wid)

        for b in my_blocks:
            lo, hi = self._block_range(b)
            k = hi - lo
            if k <= 1:
                if k == 1:
                    yield lo
                continue
            perm = torch.randperm(k, generator=rg).tolist()
            for o in perm:
                yield lo + o