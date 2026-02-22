# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icecube.icetray import I3Tray
from icecube import hdfwriter, ml_suite

import pandas as pd


def main():

    tray = I3Tray()
    tray.Add("I3Reader", Filenamelist=[
        "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass2.i3.gz",
        "/data/i3store/users/umd-ml/sim/cascade/22646/0000000-0000999/DNNCascadeL4_NuGen_22646_00000100.i3.zst"
    ])

    tray.Add(ml_suite.EventFeatureExtractorModule, cfg_file="./test_mlsuite.yaml")

    tray.AddSegment(
        hdfwriter.I3HDFWriter,
        Output="/data/i3store/users/tstjean/test_mlsuite_output.hdf5",
        Keys=[
            "ml_suite_features",
            "I3MCWeightDict"
        ],
        SubEventStreams=["InIceSplit"],
    )

    tray.Execute()


def read(visit):
    import h5py
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}")
            print(f"  shape: {obj.shape}")
            print(f"  dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")


    if visit:
        with h5py.File("/data/i3store/users/tstjean/test_mlsuite_output.hdf5", "r") as f:
            print("\nFile: /data/i3store/users/tstjean/test_mlsuite_output.hdf5")
            f.visititems(visitor)

    else:
        with h5py.File("/data/i3store/users/tstjean/test_mlsuite_output.hdf5", "r") as f:
            print("\nFile: /data/i3store/users/tstjean/test_mlsuite_output.hdf5")
            df = pd.DataFrame(f["ml_suite_features"][:])
            print(df.head(20))


if __name__ == "__main__":
    main()
    read(False)
