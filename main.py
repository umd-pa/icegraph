import warnings
import os

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.data import Data
from icegraph import config

input_dir = config.TEST_I3_DIR
config_path = os.path.join(config.CONFIG_DIR, "extraction/feature_extraction.yaml")

data = Data.from_i3(input_dir, config_path)
