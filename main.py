import warnings
import os


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.converter import HDF5ToParquet
from icegraph.extractor import FeatureExtractor
from icegraph import config


extractor = FeatureExtractor(config.TEST_I3_DIR)
extractor.extract()

converter = HDF5ToParquet(os.path.join(config.TEST_I3_DIR, "extraction/data.hdf5"))
converter.convert()
