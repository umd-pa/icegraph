import warnings


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.converter import HDF5ToParquet, I3ToHDF5
from icegraph import config


converter = I3ToHDF5(config.TEST_I3_DIR)
converter.convert()

# The following is temporary for testing, I will be implementing a full testing script after a bit more dev

import h5py
with h5py.File("/data/i3store/users/tstjean/i3_10_test/hdf5/data.hdf5", "r") as f:
    print(list(f.keys()))
    ds = f["ml_suite_features"]
    dtype = ds.dtype

    if dtype.names:
        print("Columns:")
        print(dtype.names)

    print(ds[:20])

    ds = f["classification"]
    dtype = ds.dtype

    if dtype.names:
        print("Columns:")
        print(dtype.names)

    print(ds[:10])



