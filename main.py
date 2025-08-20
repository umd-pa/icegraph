
from pathlib import Path
import time

from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.extractor import FeatureExtractor
from icegraph.data.processor import FeatureProcessor, TruthProcessor, EdgeProcessor
from icegraph.data.pipeline import Pipeline
from icegraph.data.readers import LMDBConfiguredShardReader
from icegraph.data.splitter import SplitMapBuilder
from icegraph.data.writers import LMDBWriter
from icegraph.trainer import Trainer


def main():
    source = Path("/data/i3store/users/tstjean/i3_10_test")
    outdir = Path("/data/i3store/users/tstjean/temp")

    # define the processing pipeline
    #with Pipeline() as pipeline:
    #    pipeline.build(
    #        extractor=FeatureExtractor,
    #        processors=[FeatureProcessor, TruthProcessor, EdgeProcessor],
    #        writer=LMDBWriter
    #    )
    #    pipeline.configure(source, outdir=outdir)
    #    pipeline.execute()

    map_file = SplitMapBuilder(outdir).build_map()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(outdir, map_file)

    trainer = Trainer(dataset_registry)
    trainer.run()


if __name__ == "__main__":
    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register for global access
    IGConfig.register(config)
    main()
