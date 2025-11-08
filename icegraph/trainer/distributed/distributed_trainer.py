# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os
import sys
import faulthandler
import signal
from dataclasses import dataclass
import torch
from typing import TYPE_CHECKING, Optional, Self, Type, Callable, List
from contextlib import suppress
from datetime import timedelta

import torch.multiprocessing as mp
import torch.distributed as dist

from icegraph.trainer import Trainer
from icegraph.config import IGConfig
from icegraph.console import Console
from .state import DDPProcessState

if TYPE_CHECKING:
    from icegraph.data import DatasetRegistry


@dataclass
class _TrainerInitContext:
    registry: DatasetRegistry
    callbacks: List[Trainer.CallbackSpec]
    config: IGConfig
    process_state_factory: Optional[Type[DDPProcessState]]
    debug: bool

    def no_ddp(self) -> Self:
        self.process_state_factory = None
        return self


class DistributedTrainer:

    MASTER_ADDR: str = "127.0.0.1"

    ENV_VAR: str = "CUDA_VISIBLE_DEVICES"

    CallbackSpec = Trainer.CallbackSpec

    def __init__(self, registry: DatasetRegistry, *, debug: bool = False) -> None:
        """
        Initialize the Distributed Trainer. Sets up the DDP environment and launches trainers.

        Args:
            registry (DatasetRegistry): Dataset registry containing dataloaders.
        """
        # port cache
        self._port: str

        self.trainer_ctx: _TrainerInitContext = _TrainerInitContext(
            registry=registry,
            callbacks=[],
            config=IGConfig.get(),
            process_state_factory=DDPProcessState,
            debug=debug
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def _get_port(self) -> str:
        if getattr(self, "_port", None) is None:
            import socket

            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            self._port = str(s.getsockname()[1])
            s.close()

        return self._port

    def register_callback(self, callback: Trainer.CallbackSpec) -> None:
        """Register a callback to the trainer init context."""
        self.trainer_ctx.callbacks.append(callback)

    def execute(self) -> None:
        # if already launched via external launcher, break out
        if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
            return self._execute(self.trainer_ctx)

        # respects CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            world = torch.cuda.device_count()
            if world > 1:
                os.environ.setdefault("MASTER_ADDR", self.MASTER_ADDR)
                os.environ["MASTER_PORT"] = self._get_port()
                os.environ["WORLD_SIZE"] = str(world)

                os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
                os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

                if self.trainer_ctx.debug:
                    Console.out(f"Set MASTER_ADDR={self.MASTER_ADDR}, MASTER_PORT={self._get_port()}, WORLD_SIZE={world!s}.")

                try:
                    mp.set_start_method("spawn", force=True)
                except RuntimeError:
                    pass  # already set elsewhere

                try:
                    ctx = mp.start_processes(
                        self._child_entry,
                        args=(world, self._execute, self.trainer_ctx),
                        nprocs=world,
                        join=False,
                        start_method="spawn"
                    )
                    ctx.join()
                except KeyboardInterrupt:
                    # terminate any stuck ranks
                    for p in ctx.processes:
                        if p.is_alive():
                            p.terminate()
                    for p in ctx.processes:
                        p.join()
                return None


        # single GPU or CPU
        return self._execute(self.trainer_ctx.no_ddp())

    @staticmethod  # static to avoid pickling self on spawn
    def _child_entry(
            local_rank: int, world_size: int,
            _exec: Callable[[_TrainerInitContext], None],
            trainer_ctx: _TrainerInitContext
    ) -> None:
        # neutralize any pre-installed handlers and env
        os.environ.pop("PYTHONFAULTHANDLER", None)

        for sig in (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM):
            with suppress(Exception):
                faulthandler.unregister(sig)  # if someone registered it

        with suppress(Exception):
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()

        # make sure fault handler pipes to a file not to stdout/stderr
        fh = open(f"fault_rank{local_rank}.log", "a", buffering=1)

        # Route all faulthandler output to the file
        faulthandler.enable(fh, all_threads=True)

        # on demand dumps to the same file
        faulthandler.register(signal.SIGUSR2, file=fh, all_threads=True)

        # silence ctrl+c tracebacks
        def _quiet_kbi_excepthook(t, e, tb) -> None:
            """Suppress traceback output on KeyboardInterrupt."""
            if t is KeyboardInterrupt:
                return None  # suppress console traceback
            return sys.__excepthook__(t, e, tb)

        sys.excepthook = _quiet_kbi_excepthook

        # single node DDP
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)

        # pin to GPU
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
                timeout=timedelta(seconds=30)
            )

        # assert MASTER consistency in debug mode
        if trainer_ctx.debug:
            master = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, master)

            if len(set(gathered)) != 1:
                raise RuntimeError(f"MASTER mismatch across ranks: {gathered}")

        # Run entrypoint
        try:
            return _exec(trainer_ctx)
        except KeyboardInterrupt:
            # allow graceful unwind
            return None
        finally:
            with suppress(Exception):
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            with suppress(Exception):
                dist.destroy_process_group()

            # turn off faulthandler before closing the file
            # each must attempt to run even if another fails, thus
            # each call gets put in its own exception suppression block
            with suppress(Exception):
                faulthandler.cancel_dump_traceback_later()
            with suppress(Exception):
                faulthandler.unregister(signal.SIGUSR2)
            with suppress(Exception):
                faulthandler.disable()
            with suppress(Exception):
                fh.close()

    @staticmethod  # static to avoid pickling self on spawn
    def _execute(trainer_ctx: _TrainerInitContext) -> None:
        # register for global access within each child
        IGConfig.register(trainer_ctx.config)

        process_state = trainer_ctx.process_state_factory() if trainer_ctx.process_state_factory is not None else None

        with Trainer(
                trainer_ctx.registry,
                process_state=process_state,
                debug=trainer_ctx.debug
        ) as trainer:

            # register callbacks for each trainer
            for cb in trainer_ctx.callbacks:
                trainer.register_callback(cb)

            trainer.execute()
