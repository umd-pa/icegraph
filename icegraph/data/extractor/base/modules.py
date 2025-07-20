# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import uuid

try:
    from icecube import dataclasses
    from icecube.icetray import I3Module
except ImportError:
    icetray = None
    dataclasses = None

    class I3Module:
        def PushFrame(self, frame): pass


class UniqueID(I3Module):

    def DAQ(self, frame) -> None:
        self._apply_tag(frame)

    Physics = DAQ

    def _apply_tag(self, frame):
        header = frame["I3EventHeader"]
        u = uuid.uuid4().int
        header.run_id = u & ((1 << 32) - 1)
        self.PushFrame(frame)