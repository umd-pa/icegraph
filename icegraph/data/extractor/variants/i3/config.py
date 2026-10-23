# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, FilePath, model_validator

__all__ = ["I3ExtractorConfig", "SelectionConfig", "MuonGunConfig", "RebuildMCTreeConfig", "ChargeCutConfig"]


class SelectionConfig(BaseModel):
    drop_coincident:    bool            = False  # drop Q frames holding more than one primary
    drop_oversplit:     bool            = False  # drop Q frames holding one primary split into several events
    mctree:             str             = "I3MCTree_preMuonProp"  # tree the primaries are counted in

    @model_validator(mode="after")
    def any_rule(self) -> Self:
        if not (self.drop_coincident or self.drop_oversplit):
            raise ValueError("Selection was enabled but no rules were set.")

        return self


class ChargeCutConfig(BaseModel):
    pulse_series:   str     = "SplitInIcePulses"
    min_pe:         float   = Field(default=0., ge=0)
    max_pe:         float   = Field(default=float("inf"), ge=0)


class MuonGunConfig(BaseModel):
    padding:            float           = 0.0  # metres added around the detector hull built from the GCD

    # frame keys each label is written to, a label is only computed if its key is set
    multiplicity:       str | None      = None  # number of muons entering the hull
    prominence:         str | None      = None  # leading muon energy fraction

    @model_validator(mode="after")
    def any_label(self) -> Self:
        if self.multiplicity is None and self.prominence is None:
            raise ValueError("MuonGun was enabled but no labels were set.")

        return self


class RebuildMCTreeConfig(BaseModel):
    mctree:             str             = "I3MCTree"  # propagated tree, rebuilt when the Q frame does not hold it
    raw_mctree:         str             = "I3MCTree_preMuonProp"  # un-propagated tree it is rebuilt from
    rng_state:          str             = "I3MCTree_preMuonProp_RNGState"  # state of the random number generator it was propagated with
    random_service:     Literal["SPRNG", "GSL", "MT"] = "SPRNG"  # must match the type the state was saved from


class I3ExtractorConfig(BaseModel):
    gcd_path:               FilePath
    include:                list[str]
    ml_suite:               dict[str, Any]  # validation is up to ml_suite
    mclabeler:              dict[str, Any]      | None  = None  # validation is up to MCLabeler, only run if set
    selection:              SelectionConfig     | None  = None  # only run if set
    muongun:                MuonGunConfig       | None  = None  # only run if set
    charge_cut:             ChargeCutConfig     | None  = None  # only run if set
    rebuild_missing_mctree: RebuildMCTreeConfig | None  = None  # only run if set
    sub_event_stream:       str                         = "InIceSplit"
    event_header:           str                         = "I3EventHeader"
    skip_missing:           bool                        = False  # skip files with events that lack an included key
    suppress_icetray_output: bool                       = True

    # columns identifying an event, the index columns the table writer books with every table
    ids:            list[str]       = Field(default=["Run", "Event", "SubEvent", "SubEventStream"], min_length=1)

    @model_validator(mode="after")
    def muongun_selection(self) -> Self:
        if self.muongun is None:
            return self

        # the labels cover every muon in the Q frame, which is only the truth of an event
        # when the frame holds a single primary that was split into a single event
        selection = self.selection
        if selection is None or not (selection.drop_coincident and selection.drop_oversplit):
            raise ValueError(
                "'muongun' requires 'selection' with both 'drop_coincident' and 'drop_oversplit' set, "
                "otherwise the labels will be incorrect."
            )

        return self

    @model_validator(mode="after")
    def rebuild_selection(self) -> Self:
        rebuild, selection = self.rebuild_missing_mctree, self.selection
        if rebuild is None or selection is None:
            return self

        # the tree is rebuilt after the selection, so the selection never sees a rebuilt tree
        if selection.mctree == rebuild.mctree:
            raise ValueError(
                f"'selection' counts primaries in '{selection.mctree}', which is only rebuilt after the selection. "
                f"Count them in '{rebuild.raw_mctree}' instead (the primaries are the same)."
            )

        return self
