import warnings


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.converter import H5ToParquet
from icegraph import config

import os


converter = H5ToParquet(config.TEST_H5_FILE)
converter.convert()
