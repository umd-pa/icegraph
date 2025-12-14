# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from icegraph.config import IGConfig


@dataclass
class TrainerConfig:
    max_epochs: int
    val_interval: int
    hidden_channels: int
    hidden_layers: int
    num_nbrs: int
    save_interval: int
    seed: int
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    scheduler: Optional[str]
    scheduler_kwargs: Dict[str, Any]
    scheduler_step_mode: str

    @classmethod
    def from_config(cls, config: IGConfig) -> "TrainerConfig":
        p = config.user_config.training.trainer_params
        opt = config.user_config.training.optimizer
        sch = config.user_config.training.scheduler
        return cls(
            max_epochs=p.max_epochs,
            val_interval=p.val_interval_epochs,
            hidden_channels=p.hidden_channels,
            hidden_layers=p.hidden_layers,
            num_nbrs=p.num_nbrs,
            save_interval=p.save_interval,
            seed=config.user_config.training.seed,
            optimizer=opt.task,
            optimizer_kwargs=opt.kwargs.toDict(),
            scheduler=sch.task,
            scheduler_kwargs=sch.kwargs.toDict(),
            scheduler_step_mode=sch.step_mode
        )

    def to_dict(self) -> Dict:
        return asdict(self)
