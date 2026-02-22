# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope

from .config import GroupConfig

__all__ = ["Grouper"]


class Grouper(Processor[GroupConfig]):
    """Group columns by key."""
    name: ClassVar[str] = "group"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GroupConfig:
        return GroupConfig(**config)

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        # only need to check keys, values are already either cols or valid groups
        # need to check against every frame
        for frame in env.tmp.values():
            if any(name in frame.columns for name in self.config.map_):
                raise RuntimeError("Groups and columns cannot have identical names.")

        for group, columns in self.config.map_.items():
            self.config.map_[group] = columns

        # add to state as new group
        env.state["groups"].update(self.config.map_)
        
        return env
