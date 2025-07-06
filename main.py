import warnings

from torch import split
from torch.fx import Graph

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import pandas as pd

from icegraph.data.preprocess import GraphSamplePreprocessor
from icegraph.data.extract import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.render import FeaturePlot
from icegraph.data import TrainingDataset
from icegraph.data.split import SplitFactory


def main():

    # pandas formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # specify the data input directory, usually the i3 file set
    active_file = "/data/i3store/users/tstjean/i3_100_test/extraction/graphs.lmdb"

    # define the processing chain and run each process sequentially left to right
    for stage in [FeatureExtractor, GraphSamplePreprocessor]:
        processor = stage(active_file)
        active_file = processor()

    # generate train/test/val splits
    split_factory = SplitFactory(active_file)
    lmdb_files = split_factory.generate_splits()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(*lmdb_files)

    # use dataset_registry to pass data to training system

if __name__ == "__main__":
    main()
