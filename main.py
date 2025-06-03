import warnings


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.converter import HDF5ToParquet, I3ToHDF5
from icegraph import config

import os


converter = I3ToHDF5(config.TEST_I3_DIR)
converter.convert()
