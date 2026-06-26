# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations


def main():
    from icecube import dataio

    def get_i3_livetime_seconds(path: str) -> float:
        """
        Return the wall-clock livetime/span of an I3 file in seconds.

        This uses the first and last I3EventHeader.start_time found in the file.
        """
        first_time = None
        last_time = None

        i3_file = dataio.I3File(path)

        try:
            while i3_file.more():
                frame = i3_file.pop_frame()

                if "I3EventHeader" not in frame:
                    continue

                t = frame["I3EventHeader"].start_time

                if first_time is None:
                    first_time = t

                last_time = t

        finally:
            i3_file.close()

        if first_time is None or last_time is None:
            return 0.0

        return (
                last_time.mod_julian_day_double
                - first_time.mod_julian_day_double
        ) * 86400.0

    path = "/data/i3store/users/tstjean/data/runs/00131335/Level2_IC86.2018_data_Run00131335_Subrun00000000_00000001.i3.zst"
    print(get_i3_livetime_seconds(path))


if __name__ == "__main__":
    main()
