
from pathlib import Path
import time

from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.extractor import FeatureExtractor
from icegraph.data.processor import FeatureProcessor, TruthProcessor, EdgeProcessor, StandardSplitAllocator
from icegraph.data.pipeline import Pipeline
from icegraph.data.readers import LMDBDatasetShardReader
from icegraph.data.writers import LMDBWriter
from icegraph.renderer import ParityPlot
from icegraph.trainer import Trainer
from icegraph.trainer.callbacks import RegressionMetricsCallback


def main():
    source = Path("/data/i3store/users/tstjean/i3_100_test")
    outdir = Path("/data/i3store/users/tstjean/output")

    # define the processing pipeline
    #with Pipeline() as pipeline:
    #    pipeline.build(
    #        extractor=FeatureExtractor,
    #        processors=[FeatureProcessor, TruthProcessor, EdgeProcessor, StandardSplitAllocator],
    #        writer=LMDBWriter
    #    )
    #    pipeline.configure(source, outdir=outdir)
    #    pipeline.execute()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(outdir)

    with Trainer(dataset_registry) as trainer:
        trainer.register_callback(RegressionMetricsCallback)
        trainer.execute()


def test_plotter():
    plotter = ParityPlot()
    plotter.plot([0, 1, 2], [1, 4, 3], "bruh", "/data/i3store/users/tstjean")


if __name__ == "__main__":
    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register for global access
    IGConfig.register(config)
    main()
