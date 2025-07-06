# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import uuid

from icecube import icetray, dataclasses


class UniqueID(icetray.I3Module):

    def DAQ(self, frame) -> None:
        self._apply_tag(frame)

    Physics = DAQ

    def _apply_tag(self, frame):
        header = frame["I3EventHeader"]
        u = uuid.uuid4().int
        header.run_id = u & ((1 << 32) - 1)
        self.PushFrame(frame)