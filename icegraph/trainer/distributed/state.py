# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, Dict
import os

import torch
import torch.distributed as dist


class DDPProcessState:
    procinfo: Optional[Dict[str, int]] = None

    def __bool__(self) -> bool:
        return bool(self.procinfo)

    @staticmethod
    def env_ready() -> bool:
        """Checks if DDP env vars are set (RANK, WORLD_SIZE, LOCAL_RANK)."""
        return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))

    def cleanup(self, interrupt: bool = False):
        """Destroy the default process group. Safe to call multiple times."""
        if dist.is_available() and dist.is_initialized() and self:
            try:
                if not interrupt:
                    backend = dist.get_backend()
                    if backend == "nccl" and torch.cuda.is_available():
                        dist.barrier()
                    else:
                        dist.barrier()
            except Exception:
                pass
            finally:
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass

    def is_main_process(self) -> bool:
        """Returns True if the current process is the main process. Checks if RANK is 0."""
        return self.procinfo["rank"] == 0

    def barrier(self):
        """Wait for all processes to reach this barrier."""
        if dist.is_available() and dist.is_initialized() and self:
                dist.barrier()

    def init(self) -> None:
        """
        Initialize (or attach to) the default process group based on env vars.
        Safe to call multiple times. Returns rank/world/local_rank dict, or None if not in DDP mode.
        """
        if not self.env_ready():
            return None

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        self.procinfo = {"rank": rank, "world": world, "local_rank": local_rank}
        return None