# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Self
from pathlib import Path

from pydantic import BaseModel, FilePath, model_validator

__all__ = ["I3ExtractorConfig", "SelectionConfig", "MultiplicityConfig"]


class SelectionConfig(BaseModel):
    drop_coincident:    bool            = False  # drop Q frames holding more than one primary
    drop_oversplit:     bool            = False  # drop Q frames holding one primary split into several events
    mctree:             str             = "I3MCTree"  # tree the primaries are counted in

    @model_validator(mode="after")
    def any_rule(self) -> Self:
        if not (self.drop_coincident or self.drop_oversplit):
            raise ValueError("Selection was enabled but no rules were set.")

        return self


class MultiplicityConfig(BaseModel):
    key:                str             = "bundle_multiplicity"  # frame key the count is written to
    padding:            float           = 0.0  # metres added around the detector hull built from the GCD


class I3ExtractorConfig(BaseModel):
    gcd_path:       FilePath
    include:        list[str]
    ml_suite:       dict[str, Any]  # validation is up to ml_suite
    mclabeler:      dict[str, Any] | None = None  # validation is up to MCLabeler, only run if set
    selection:      SelectionConfig | None = None  # only run if set
    multiplicity:   MultiplicityConfig | None = None  # only run if set
    sub_event_stream: str           = "InIceSplit"
    skip_missing:   bool            = False  # skip any files with no frames
    suppress_icetray_output: bool   = True

    @model_validator(mode="after")
    def multiplicity_selection(self) -> Self:
        if self.multiplicity is None:
            return self

        # the multiplicity counts every muon in the Q frame, which is only the truth of an event
        # when the frame holds a single primary that was split into a single event
        selection = self.selection
        if selection is None or not (selection.drop_coincident and selection.drop_oversplit):
            raise ValueError(
                "'multiplicity' requires 'selection' with both 'drop_coincident' and 'drop_oversplit' set, "
                "otherwise the count will be incorrect."
            )

        return self
