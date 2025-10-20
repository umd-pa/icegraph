# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os
import torch
from typing import Callable, Any, Optional

import torch.multiprocessing as mp
import torch.distributed as dist


def _child_entry(local_rank: int, world_size: int, main_fn: Callable[[], Any]):
    # Minimal env for single node DDP
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"]  = str(world_size)
    os.environ["LOCAL_RANK"]  = str(local_rank)
    os.environ["RANK"]        = str(local_rank)

    # pin to GPU (this will change, will add ability to select GPUs from config)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # initialize default process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=local_rank,
            world_size=world_size,
        )

    # Run user entrypoint
    return main_fn()

def autorun(main_fn: Callable[[], Any], world_size: Optional[int] = None):
    """
    If already under DDP env -> run main_fn().
    If >= 2 GPUs -> spawn one process per GPU.
    Else -> run single process.
    """
    ddp_env = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if ddp_env:
        # if external launcher, ensure device is set
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        return main_fn()

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return main_fn()

    ws = int(world_size or n_gpus)
    mp.set_start_method("spawn", force=True)
    mp.spawn(_child_entry, args=(ws, main_fn), nprocs=ws, join=True)