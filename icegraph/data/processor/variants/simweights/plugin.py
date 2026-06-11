# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Callable

from numpy.typing import ArrayLike
import simweights

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import SimweighterConfig
from .model import FluxModel, FluxModelFactory, FluxModelContext

__all__ = ["Simweighter"]


class Simweighter(Processor[SimweighterConfig]):
    """Load a frame into tmp for processing."""
    name: ClassVar[str] = "simweights"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SimweighterConfig:
        return SimweighterConfig(**config)

    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # resolve flux model class and weighter class
        weighter_cls = self._get_weighter(self.config.weighter)

        # instantiate flux model
        flux_model = FluxModelFactory.create(self.config.flux.name, **self.config.flux.kwargs)
        flux_model.attach(FluxModelContext())

        # construct the weighter over env.data
        weighter = weighter_cls(env.data, nfiles=1)

        # compute weights
        weights = weighter.get_weights(flux_model)

        # weights must align with the active frame row count
        if len(weights) != len(main):
            raise RuntimeError(
                f"simweights returned {len(weights)} weights but active frame "
                f"{env.active!r} has {len(main)} rows. The active frame likely "
                f"diverged from env.data ordering after some upstream processor."
            )

        # write to active frame
        main[self.config.out] = weights

        # cache weight_group to attrs
        env.set_local_attr("weight_group", self.config.weight_group)

        return env

    @staticmethod
    def _get_weighter(name: str) -> Callable[[...], simweights.Weighter]:
        weighter = getattr(simweights, name, None)

        # ensure the user requested a valid weighter
        if weighter is None:
            raise ValueError(f"Requested weighter '{name}' does not exist.")

        if not callable(weighter):
            raise TypeError(
                f"{name!r} must be callable (a Weighter factory or subclass); "
                f"got {type(weighter).__name__}"
            )

        return weighter
