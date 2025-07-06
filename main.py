import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from pathlib import Path

import pandas as pd

from icegraph.data.transform import TransformToDataset
from icegraph.data.extract import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.split import DatasetSplitter
from icegraph.train import Trainer
from icegraph.train.module import GravNetModel


def main():
    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # specify the data input directory, usually the i3 file set
    resource = [f"/data/i3store/users/tstjean/i3_100_test"]

    # define the processing chain and run each process sequentially left to right
    for stage in [FeatureExtractor, TransformToDataset, DatasetSplitter]:
        processor = stage(resource)
        resource = processor()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(*resource)

    # use dataset_registry to pass data to training system
    in_channels = dataset_registry.train_dataset.num_node_features
    hidden_channels = 64

    model = GravNetModel(in_channels, hidden_channels, 1)
    trainer = Trainer(dataset_registry, model)
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main()
