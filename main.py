import warnings
import time

from icegraph.data.readers import LMDBConfiguredShardReader

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

from pathlib import Path
import pprint

import pandas as pd

from icegraph.data.processor import FeatureProcessor
from icegraph.data.extractor import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.splitter import DatasetSplitter
from icegraph.data.mergers import LMDBMerger
from icegraph.trainer import Trainer
from icegraph.trainer.callbacks import CheckpointCallback, ConsoleCallback, TensorBoardCallback
from icegraph.data.pulses import Pulses
from icegraph.renderer import CDFPlot, ChargeDistributionPlot, PDFPlot


def main():
    # specify the data input directory, usually the i3 file set
    source = Path("/data/i3store/users/tstjean/i3_100_test/processor")

    # define the processing chain and run each process sequentially left to right
    for stage in []:
        processor = stage(source)
        source = processor()

    map_file = DatasetSplitter(source).build_map()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(source, map_file)

    trainer = Trainer(dataset_registry)
    trainer.run()


def measure_throughput():
    resource = Path("/data/i3store/users/tstjean/i3_10_test/processor")
    map_file = Path("/data/i3store/users/tstjean/i3_10_test/splits/split_map.lmdb")

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(resource, map_file)

    loader = dataset_registry.train_dataloader

    count = 0
    start = time.perf_counter()
    for batch in loader:
        count += batch[0].size(0)  # or however many samples this batch holds
        if count >= 50000:
            break
    elapsed = time.perf_counter() - start
    print(f"Effective loader throughput: {count / elapsed:.1f} samples/s")


def run_parallel():
    resource = Path("/data/i3store/users/tstjean/21220_processed")

    merger = LMDBMerger(resource)
    lmdb_file = merger.merge("/data/i3store/users/tstjean/21220_merged/graphs.lmdb")


def test_cdf():
    resource = Path("/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_21002_alerts/Alertv2_IC86.2016_NuMu.021002.000002.i3.zst")
    pulses = Pulses(resource)

    plot = CDFPlot(pulses)
    plot.plot()

    plot = ChargeDistributionPlot(pulses)
    plot.plot()

    plot = PDFPlot(pulses)
    plot.plot()


if __name__ == "__main__":
    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register for global access
    IGConfig.register(config)
    main()
