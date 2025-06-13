import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)


from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.render import FeaturePlot


def main():

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    dataset_registry = DatasetRegistry.from_config(config)

    plot = FeaturePlot(dataset_registry.validation_dataset, config)
    plot.plot_feature("total_charge", 1)

if __name__ == "__main__":
    main()
