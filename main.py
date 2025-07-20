import warnings
from backcall import callback_prototype

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from pathlib import Path

from icegraph.data.processor import FeatureProcessor
from icegraph.data.extractor import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.splitter import DatasetSplitter
from icegraph.trainer import Trainer
from icegraph.trainer.callbacks import CheckpointCallback, ConsoleCallback, TensorBoardCallback


def main():
    config_path = Path("./config/config.yaml")
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # specify the data input directory, usually the i3 file set
    resource = [Path(f"/data/i3store/users/tstjean/i3_100_test/extraction/splits/{split}.graphs.lmdb") for split in ["train", "val", "test"]]

    # define the processing chain and run each process sequentially left to right
    for stage in []:
        processor = stage(resource)
        resource = processor()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(*resource)

    trainer = Trainer(dataset_registry)
    trainer.run()

if __name__ == "__main__":
    main()
