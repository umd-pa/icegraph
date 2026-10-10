# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from icegraph.utils.stdout import suppress_output

from .config import SelectionConfig, MultiplicityConfig, RebuildMCTreeConfig

__all__ = ["is_sub_event_stream", "DropCounter", "event_selector", "bundle_multiplicity", "rebuild_mctree"]


def is_sub_event_stream(frame: Any, sub_event_stream: str) -> bool:
    """Whether a P frame belongs to the sub event stream events are extracted from."""
    return frame.Has("I3EventHeader") and frame["I3EventHeader"].sub_event_stream == sub_event_stream


@dataclass
class DropCounter:
    dropped:    dict[str, dict[str, int]]

    @classmethod
    def from_config(cls, config: SelectionConfig) -> DropCounter:
        # one entry per enabled rule, a rule never set is left out rather than counted as zero
        rules = {"coincident": config.drop_coincident, "oversplit": config.drop_oversplit}
        return cls(dropped={rule: {"daq": 0, "physics": 0} for rule, on in rules.items() if on})


def event_selector(config: SelectionConfig, tally: DropCounter, sub_event_stream: str) -> type:
    """
    Build a module dropping each simulated Q frame, along with every P frame split from it,
    that meet certain criteria.
    """
    with suppress_output():
        from icecube import icetray  # pyright: ignore[reportMissingImports]

    class EventSelector(icetray.I3PacketModule):
        def __init__(self, context):
            # a packet is a Q frame and every P frame that follows it
            icetray.I3PacketModule.__init__(self, context, icetray.I3Frame.DAQ)

        def _reason(self, daq: Any, events: int) -> str | None:
            # a packet holding no event produces nothing
            # if it doesnt start with a Q frame we don't know how to handle
            if not events or daq.Stop != icetray.I3Frame.DAQ:
                return None

            if config.mctree not in daq:
                raise KeyError(
                    f"selection counts primaries in '{config.mctree}', which the Q frame does not hold. "
                    f"The selection only applies to simulation."
                )

            primaries = len(daq[config.mctree].get_primaries())
            if config.drop_coincident and primaries > 1:
                return "coincident"
            if config.drop_oversplit and primaries == 1 and events > 1:
                return "oversplit"

            return None

        def FramePacket(self, frames):
            daq, *rest = frames
            events = sum(1 for frame in rest if is_sub_event_stream(frame, sub_event_stream))

            reason = self._reason(daq, events)
            if reason is None:
                for frame in frames:
                    self.PushFrame(frame)
                return

            # nothing is pushed, so the Q frame and every P frame split from it are dropped together
            tally.dropped[reason]["daq"] += 1
            tally.dropped[reason]["physics"] += events

    return EventSelector


def bundle_multiplicity(config: MultiplicityConfig) -> type:
    """
    Build a module writing the bundle multiplicity, defined as the number of muons that enter the detector
    hull, to each Q frame.
    """
    with suppress_output():
        from icecube import icetray, dataclasses, simclasses, MuonGun  # noqa: F401  # pyright: ignore[reportMissingImports]

    # only muons belong to a bundle
    muons = (dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.MuPlus)

    class BundleMultiplicity(icetray.I3Module):
        def __init__(self, context):
            icetray.I3Module.__init__(self, context)
            self._surface = None

        def Geometry(self, frame):
            self._surface = MuonGun.ExtrudedPolygon.from_I3Geometry(frame["I3Geometry"], config.padding)
            self.PushFrame(frame)

        def DAQ(self, frame):
            if self._surface is None:
                raise RuntimeError("bundle multiplicity needs the geometry, which no frame before the Q frame held.")

            count = sum(1 for p in MuonGun.muons_at_surface(frame, self._surface) if p.type in muons)
            frame[config.key] = icetray.I3Int(count)
            self.PushFrame(frame)

    return BundleMultiplicity


def rebuild_mctree(tray: Any, config: RebuildMCTreeConfig) -> None:
    """
    Add a module rebuilding the propagated tree of each Q frame that does not hold it using PROPOSAL/CMC.
    """
    with suppress_output():
        from icecube import phys_services, sim_services  # noqa: F401  # pyright: ignore[reportMissingImports]
        from icecube.simprod.segments.PropagateMuons import make_standard_propagators  # pyright: ignore[reportMissingImports]

    # the seeds do not matter, the generator is re-initialized from the saved state
    random_service = {
        "SPRNG":    lambda: phys_services.I3SPRNGRandomService(seed=2, nstreams=10000, streamnum=1),
        "GSL":      lambda: phys_services.I3GSLRandomService(42),
        "MT":       lambda: phys_services.I3MTRandomService(42),
    }[config.random_service]()

    # without the saved state the tree cannot be reproduced
    def missing(frame: Any) -> bool:
        return config.rng_state in frame and config.mctree not in frame

    # the propagator writes the track list along with the tree and fails if the frame kept the original,
    # which is replaced by the one rebuilt along with the tree
    tray.Add("Delete", Keys=["MMCTrackList"], If=missing)

    tray.Add(
        "I3PropagatorModule",
        PropagatorServices=make_standard_propagators(),
        RandomService=random_service,
        RNGStateName=config.rng_state,
        InputMCTreeName=config.raw_mctree,
        OutputMCTreeName=config.mctree,
        If=missing,
    )
