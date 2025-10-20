# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def env_ready() -> bool:
    """Checks if DDP env vars are set (RANK, WORLD_SIZE, LOCAL_RANK)."""
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def init(backend: str = "nccl"):
    """
    Initialize (or attach to) the default process group based on env vars.
    Safe to call multiple times. Returns rank/world/local_rank dict, or None if not in DDP mode.
    """
    if not env_ready():
        return None

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if dist.is_available() and not dist.is_initialized():
        be = backend
        if be == "nccl" and not torch.cuda.is_available():
            be = "gloo"
        dist.init_process_group(backend=be, init_method="env://", rank=rank, world_size=world)

    return {"rank": rank, "world": world, "local_rank": local_rank}


def cleanup():
    """Destroy the default process group. Safe to call multiple times."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Returns True if the current process is the main process. Checks if RANK is 0."""
    return int(os.environ.get("RANK", "0")) == 0


def barrier():
    """Wait for all processes to reach this barrier."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()