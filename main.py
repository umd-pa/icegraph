
from pathlib import Path
import time

from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.transformer import FeatureExtractor, FeatureProcessor
from icegraph.data.splitter import SplitMapBuilder
from icegraph.trainer import Trainer


def main():
    # specify the data input directory, usually the i3 file set
    source = Path("/data/i3store/users/tstjean/i3_10_test")

    # define the processing chain and run each process sequentially left to right
    for stage in [FeatureExtractor, FeatureProcessor]:
        processor = stage(source)
        source = processor()

    map_file = SplitMapBuilder(source).build_map()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(source, map_file)

    trainer = Trainer(dataset_registry)
    trainer.run()


if __name__ == "__main__":
    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register for global access
    IGConfig.register(config)
    main()
