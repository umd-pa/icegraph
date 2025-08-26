# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

from icegraph.config import IGConfig


@dataclass
class TrainerConfig:
    max_epochs: int
    test_interval: int
    hidden_channels: int
    hidden_layers: int
    seed: int
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool

    @classmethod
    def from_config(cls, config: IGConfig) -> "TrainerConfig":
        p = config.user_config.training.trainer_params
        opt = config.user_config.training.optimizer.toDict()
        return cls(
            max_epochs=p.max_epochs,
            test_interval=p.test_interval_epochs,
            hidden_channels=p.hidden_channels,
            hidden_layers=p.hidden_layers,
            seed=config.user_config.training.seed,
            lr=float(opt["learning_rate"]),
            betas=tuple(map(float, opt["betas"])),
            eps=float(opt["eps"]),
            weight_decay=float(opt["weight_decay"]),
            amsgrad=bool(opt["amsgrad"])
        )

    def to_dict(self) -> Dict:
        return asdict(self)
