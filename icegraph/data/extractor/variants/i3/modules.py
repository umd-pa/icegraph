# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from icegraph.utils.stdout import suppress_output

from .config import SelectionConfig, MuonGunConfig, RebuildMCTreeConfig, ChargeCutConfig

__all__ = ["is_sub_event_stream", "DropCounter", "event_selector", "muongun_labels", "rebuild_mctree"]


def readable(frame: Any, key: str) -> bool:
    """
    Whether the frame holds an object under the key that deserializes. A corrupt object is still listed under its key,
    but reading it fails, and readers such as MuonGun see it as absent.
    """
    if key not in frame:
        return False

    try:
        return frame[key] is not None
    except KeyError:
        return False


def is_sub_event_stream(frame: Any, sub_event_stream: str) -> bool:
    """Whether a P frame belongs to the sub event stream events are extracted from."""
    return frame.Has("I3EventHeader") and frame["I3EventHeader"].sub_event_stream == sub_event_stream


@dataclass
class DropCounter:
    dropped: dict[str, dict[str, int]] = field(default_factory=dict)


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
            # make sure tally is set
            for reason in ("coincident", "oversplit"):
                d = tally.dropped.setdefault(reason, {})
                for frame_type in ("daq", "physics"):
                    d.setdefault(frame_type, 0)

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


def charge_cut(config: ChargeCutConfig, tally: DropCounter, sub_event_stream: str) -> type:
    with suppress_output():
        from icecube import icetray, dataclasses  # pyright: ignore[reportMissingImports]

    class ChargeCutter(icetray.I3Module):
        def __init__(self, context):
            icetray.I3Module.__init__(self, context)

        def _get_total_charge(self, frame: Any) -> float:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, config.pulse_series)
            return float(sum(pulse.charge for series in pulses.values() for pulse in series))

        def Physics(self, frame: Any):
            if not is_sub_event_stream(frame, sub_event_stream):
                self.PushFrame(frame)
                return
            
            # make sure tally is set
            tally.dropped.setdefault("charge", {}).setdefault("physics", 0)

            total_charge = self._get_total_charge(frame)

            # check if within set bounds
            if config.min_pe <= total_charge <= config.max_pe:
                self.PushFrame(frame)
                return

            tally.dropped["charge"]["physics"] += 1

    return ChargeCutter


def muongun_labels(config: MuonGunConfig) -> type:
    """
    Build a module writing labels of the muon bundle that enters the detector hull, computed with MuonGun, to each
    Q frame: the multiplicity, the number of muons that enter, and the prominence, the share of the bundle energy
    at the hull carried by the leading muon.
    """
    with suppress_output():
        from icecube import icetray, dataclasses, simclasses, MuonGun  # noqa: F401  # pyright: ignore[reportMissingImports]

    # only muons belong to a bundle
    muons = (dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.MuPlus)

    class MuonGunLabels(icetray.I3Module):
        def __init__(self, context):
            icetray.I3Module.__init__(self, context)
            self._surface = None

        def Geometry(self, frame):
            self._surface = MuonGun.ExtrudedPolygon.from_I3Geometry(frame["I3Geometry"], config.padding)
            self.PushFrame(frame)

        def DAQ(self, frame):
            if self._surface is None:
                raise RuntimeError("MuonGun labels need the geometry, which no frame before the Q frame held.")

            # MuonGun reads the propagated tree under this name only
            # name the event and the trees the frame does hold, since MuonGun's own error names neither
            if not readable(frame, "I3MCTree"):
                header = frame["I3EventHeader"] if "I3EventHeader" in frame else None
                event = f"run {header.run_id}, event {header.event_id}" if header is not None else "with no header"
                state = "holds it, but it does not deserialize" if "I3MCTree" in frame else "does not hold it"
                trees = sorted(key for key in frame.keys() if "MCTree" in key)
                raise RuntimeError(
                    f"MuonGun labels need the propagated tree 'I3MCTree', and the Q frame ({event}) {state}. "
                    f"It holds {trees}. Set rebuild_missing_mctree to rebuild it, which needs the frame to have kept "
                    f"the RNG state of the propagation."
                )

            # MuonGun only counts muons with an entry in the track list, so a tree without one has no muon to count
            # the propagator stores no list for an event it propagated no muon or tau in
            if "MMCTrackList" not in frame:
                energies = np.empty(0)
            else:
                # energy of each muon as it enters the hull
                entering = MuonGun.muons_at_surface(frame, self._surface)
                energies = np.array([p.energy for p in entering if p.type in muons])

            if config.multiplicity is not None:
                frame[config.multiplicity] = icetray.I3Int(energies.size)

            # leading muon energy fraction
            if config.prominence is not None:
                prominence = energies.max() / energies.sum() if energies.size > 0 else np.nan
                frame[config.prominence] = dataclasses.I3Double(prominence)

            self.PushFrame(frame)

    return MuonGunLabels


@cache
def _propagators() -> Any:
    """
    The PROPOSAL/CMC propagators, built once per process and shared by every tray. Building them is slow.
    """
    with suppress_output():
        from icecube.simprod.segments.PropagateMuons import make_standard_propagators  # pyright: ignore[reportMissingImports]

    return make_standard_propagators()


def rebuild_mctree(tray: Any, config: RebuildMCTreeConfig) -> None:
    """
    Add a module rebuilding the propagated tree of each Q frame that does not hold it using PROPOSAL/CMC.
    """
    with suppress_output():
        from icecube import phys_services, sim_services  # noqa: F401  # pyright: ignore[reportMissingImports]

    # the seeds do not matter, the generator is re-initialized from the saved state
    # the module hands its generator to the shared propagators, so each tray still propagates with its own
    random_service = {
        "SPRNG":    lambda: phys_services.I3SPRNGRandomService(seed=2, nstreams=10000, streamnum=1),
        "GSL":      lambda: phys_services.I3GSLRandomService(42),
        "MT":       lambda: phys_services.I3MTRandomService(42),
    }[config.random_service]()

    # without the saved state the tree cannot be reproduced
    # a tree that does not deserialize is rebuilt as if it was discarded
    def missing(frame: Any) -> bool:
        return config.rng_state in frame and not readable(frame, config.mctree)

    # the propagator writes the track list along with the tree and fails if the frame kept the original,
    # which is replaced by the one rebuilt along with the tree
    tray.Add("Delete", Keys=[config.mctree, "MMCTrackList"], If=missing)

    tray.Add(
        "I3PropagatorModule",
        PropagatorServices=_propagators(),
        RandomService=random_service,
        RNGStateName=config.rng_state,
        InputMCTreeName=config.raw_mctree,
        OutputMCTreeName=config.mctree,
        If=missing,
    )
