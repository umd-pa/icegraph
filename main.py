import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.dataset import Data
from icegraph.config import Config

config_path = "./config/config.yaml"
config = Config(config_path)

data = Data.from_config(config)
