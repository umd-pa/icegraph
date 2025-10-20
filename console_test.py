from __future__ import annotations
from icegraph.trainer.callbacks import ConsoleCallback
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional, List
import time

# --- minimal Console stub so callback's Console.out works ---
class Console:
    @staticmethod
    def out(msg: str, severity: int = 1):
        print(msg)

# --- stubbed "optimizer" with param_groups so LR logging works ---
class DummyOptimizer:
    def __init__(self, lr: float = 1e-3):
        # matches torch optimizer interface used by your callback
        self.param_groups: List[Dict[str, Any]] = [{"lr": lr}]

# --- tiny trainer_config with just max_epochs ---
@dataclass
class TrainerConfigStub:
    max_epochs: int = 3

# --- absolute bare-minimum trainer for your callback ---
class MiniTrainer:
    def __init__(self, *, epochs: int = 20, batches: int = 2000, lr: float = 1e-3, outdir: str | Path = "./_dummy_out"):
        # attrs the callback reads
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.device = SimpleNamespace(type="cpu")   # behaves like torch.device('cpu')
        self.trainer_config = TrainerConfigStub(max_epochs=epochs)
        self.optimizer = DummyOptimizer(lr=lr)
        self.train_batch_count = int(batches)

    # fire the exact hooks your callback expects, in a realistic order
    def run_with(self, callback) -> None:
        # on_train_begin
        callback.on_train_begin(self)

        for epoch in range(self.trainer_config.max_epochs):
            # on_epoch_begin
            callback.on_epoch_begin(self, epoch)

            # simulate batches
            for b in range(self.train_batch_count):
                time.sleep(0.001)
                # your callback only uses on_batch_end to advance the bar
                callback.on_batch_end(self, batch=None, out=None, target=None, loss=0.1234, metrics={"loss": 0.1234})

            # on_epoch_end (aliased to display_metrics in your code)
            callback.on_epoch_end(self, epoch, {"loss": 0.1234 - epoch * 0.01, "rmse": 0.35 - math.sqrt(epoch * 0.01)})

        # on_teardown
        callback.on_teardown(self)


if __name__ == "__main__":
    MiniTrainer().run_with(ConsoleCallback())
