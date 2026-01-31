# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os

import torch.distributed as dist
from torch.distributed import Work

from icegraph.trainer.types import Params

from ..service import Service
from ..types import ServiceContext

from .view import StateView

__all__ = ["StateService"]


class StateService(Service):
    name = "state"
    deps = []
    view = StateView

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        self._procinfo: dict[str, int] = {}

    def __bool__(self) -> bool:
        return bool(self._procinfo)

    def on_attach(self, ctx: ServiceContext) -> None:
        self._procinfo["rank"]       = 0 if not self._env_ready() else int(os.environ["RANK"])
        self._procinfo["world"]      = 1 if not self._env_ready() else int(os.environ["WORLD_SIZE"])
        self._procinfo["local_rank"] = 0 if not self._env_ready() else int(os.environ["LOCAL_RANK"])

    @property
    def rank(self) -> int | None:
        return self._procinfo.get("rank")

    @property
    def world(self) -> int | None:
        return self._procinfo.get("world")

    @property
    def local_rank(self) -> int | None:
        return self._procinfo.get("local_rank")

    def close(self):
        """Destroy the default process group. Safe to call multiple times."""
        if self.is_ddp():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def is_main_process(self) -> bool:
        """Returns True if the current process is the main process. Checks if RANK is 0."""
        return self._procinfo["rank"] == 0

    def barrier(self) -> Work | None:
        """Wait for all processes to reach this barrier."""
        if self.is_ddp():
            dist.barrier()

    def is_ddp(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.world > 1

    @staticmethod
    def _env_ready() -> bool:
        """Checks if DDP env vars are set (RANK, WORLD_SIZE, LOCAL_RANK)."""
        return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
