# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import uuid

from icegraph.exceptions import IceCubeImportError

import warnings

# Silence Boost.Python converter warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

try:
    from icecube import dataclasses as _dataclasses
    from icecube import icetray as _icetray
    from icecube.icetray import I3Module as _I3Module
except ImportError:
    _dataclasses = IceCubeImportError()
    _icetray = IceCubeImportError()
    _I3Module = IceCubeImportError.IceCubeMissingBase

dataclasses = _dataclasses
icetray = _icetray
I3Module = _I3Module


class UniqueID(I3Module):

    def DAQ(self, frame) -> None:
        self._apply_tag(frame)

    Physics = DAQ

    def _apply_tag(self, frame):
        header = frame["I3EventHeader"]
        u = uuid.uuid4().int
        header.run_id = u & ((1 << 32) - 1)
        self.PushFrame(frame)