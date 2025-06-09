import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.dataset import IGData
from icegraph.config import IGConfig

config_path = "./config/config.yaml"
config = IGConfig(config_path)

data = IGData.from_config(config)
