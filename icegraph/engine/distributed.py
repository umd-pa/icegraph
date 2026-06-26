# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable, Generic, TypeVar, Self
import inspect
from dataclasses import dataclass
from datetime import timedelta
import traceback
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from .callbacks import CallbackSpec
from .engine import Engine

__all__ = ["Distributed"]


E = TypeVar("E", bound=Engine)


@dataclass
class EngineInitContext:
    callbacks:  list[CallbackSpec]
    engine_cls: type[Engine]
    ctor:       str
    args:       tuple
    kwargs:     dict


def _execute(ctx: EngineInitContext) -> None:
    """Build a plain engine in this rank via the recorded constructor."""
    factory = getattr(ctx.engine_cls, ctx.ctor)
    with factory(*ctx.args, **ctx.kwargs) as engine:
        for callback in ctx.callbacks:
            engine.callbacks.register(callback)
        engine.execute()

def _child_entry(local_rank: int, world_size: int, ctx: EngineInitContext) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=local_rank,
                world_size=world_size,
                timeout=timedelta(seconds=30),
            )
        _execute(ctx)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class _DistributedRun:
    """A configured distributed launch. Mirrors an engine's
    context-manager + execute surface so it drops in for a plain engine."""

    MASTER_ADDR: str = "127.0.0.1"
    MASTER_PORT: str = "29500"

    def __init__(self, ctx: EngineInitContext) -> None:
        self._init_ctx = ctx

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def register_callback(self, callback: CallbackSpec) -> None:
        self._init_ctx.callbacks.append(callback)

    def execute(self) -> None:
        world = torch.cuda.device_count()
        if world <= 1:
            _execute(self._init_ctx)
            return

        os.environ.setdefault("MASTER_ADDR", self.MASTER_ADDR)
        os.environ.setdefault("MASTER_PORT", self.MASTER_PORT)
        os.environ["WORLD_SIZE"] = str(world)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

        ctx: mp.ProcessContext | None = mp.start_processes(
            _child_entry,
            args=(world, self._init_ctx),
            nprocs=world,
            join=False,
            start_method="spawn",
        )

        if ctx is not None:
            try:
                ctx.join()
            except KeyboardInterrupt:
                for p in ctx.processes:
                    if p.is_alive():
                        p.terminate()
                for p in ctx.processes:
                    p.join()
                raise


class Distributed(Generic[E]):
    """Launch any engine under single-node DDP."""

    def __init__(self, engine_cls: type[E]) -> None:
        self._engine_cls = engine_cls


    def __getattr__(self, name: str) -> Callable[..., _DistributedRun]:
        # Runtime delegation
        if name.startswith("_"):
            raise AttributeError(name)

        engine_cls = self._engine_cls

        raw = inspect.getattr_static(engine_cls, name, None)
        if not isinstance(raw, (classmethod, staticmethod)):
            raise AttributeError(
                f"{engine_cls.__name__!r} has no constructor {name!r} "
                f"(Distributed only forwards classmethods/staticmethods)"
            )

        def deferred_constructor(*args, **kwargs) -> _DistributedRun:
            return _DistributedRun(EngineInitContext(
                callbacks=[],
                engine_cls=engine_cls,
                ctor=name,
                args=args,
                kwargs=kwargs,
            ))

        return deferred_constructor

    def __repr__(self) -> str:
        return f"Distributed({self._engine_cls.__name__})"