# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar
import os

import torch
import torch.distributed as dist

from ..service import Service

from .view import StateView
from .config import StateConfig
from .types import ProcInfo

__all__ = ["StateService"]


class StateService(Service[StateView, StateConfig]):
    name: ClassVar[str] = "state"

    interface = StateView

    # make the type checker happy
    _procinfo:  ProcInfo
    _device:    torch.device

    def build(self) -> None:
        # init to default world 1 on cpu
        self._procinfo = ProcInfo()
        self._device = torch.device("cpu")

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StateConfig:
        return StateConfig(**config)

    def on_attach(self) -> None:
        if self._env_ready():
            self._procinfo = ProcInfo(
                rank=int(os.environ["RANK"]),
                world=int(os.environ["WORLD_SIZE"]),
                local_rank=int(os.environ["LOCAL_RANK"])
            )

        if torch.cuda.is_available():
            # get local rank from env
            local_rank = self._procinfo.local_rank

            # set device
            torch.cuda.set_device(local_rank)
            self._device = torch.device("cuda", local_rank)

    @property
    def rank(self) -> int:
        return self._procinfo.rank

    @property
    def world(self) -> int:
        return self._procinfo.world

    @property
    def local_rank(self) -> int:
        return self._procinfo.local_rank

    @property
    def device(self) -> torch.device:
        return self._device

    def is_main_process(self) -> bool:
        """Returns True if the current process is the main process. Checks if RANK is 0."""
        return self._procinfo.rank == 0

    def barrier(self) -> None:
        """Wait for all processes to reach this barrier."""
        if self.is_ddp():
            dist.barrier()

    def is_ddp(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.world > 1

    @staticmethod
    def _env_ready() -> bool:
        """Checks if DDP env vars are set (RANK, WORLD_SIZE, LOCAL_RANK)."""
        return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))

    def state_dict(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(mode="json")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = type(self).validate_config(state["config"])
