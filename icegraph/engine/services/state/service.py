# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING
from typing_extensions import override
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..service import Service

from .config import StateConfig
from .types import BoundModel
from ._procinfo import ProcInfo
from ._noddp import _NoDDP

if TYPE_CHECKING:
    from icegraph.engine.components.model import Model

__all__ = ["StateService"]


class StateService(Service[StateConfig]):
    name: ClassVar[str] = "state"
    version: ClassVar[int] = 1

    # make the type checker happy
    _procinfo:  ProcInfo
    _device:    torch.device

    @override
    def build(self) -> None:
        # init to default world 1 on cpu
        self._procinfo = ProcInfo()
        self._device = torch.device("cpu")

    @classmethod
    @override
    def validate_config(cls, config: dict[str, Any]) -> StateConfig:
        return StateConfig(**config)

    @override
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

    @property
    def seed(self) -> int:
        return self.config.seed

    def bind_model(self, model: Model[Any]) -> BoundModel:
        """Bind a model to the current execution context."""
        if self.is_ddp():
            return DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,  # type: ignore
                output_device=(self.device.index if self.device.type == "cuda" else None),  # type: ignore
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False
            )

        return _NoDDP(model)

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
