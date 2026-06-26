# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import Config

__all__ = ["Aliaser"]


class Aliaser(Processor[Config]):
    """Alias columns by key."""
    name: ClassVar[str] = "alias"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        # only need to check keys, values are already either cols or valid aliases
        # need to check against every frame
        for frame in item.tmp.values():
            if any(name in frame.columns for name in self.config.map_):
                raise RuntimeError("Aliases and columns cannot have identical names.")

        # add to state as new group
        item.state["alias"].update(self.config.map_)
        
        return item
