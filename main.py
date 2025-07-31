import warnings

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
from icegraph.data.mergers import LMDBMerger
from icegraph.trainer import Trainer
from icegraph.trainer.callbacks import CheckpointCallback, ConsoleCallback, TensorBoardCallback
from icegraph.data.pulses import Pulses
from icegraph.renderer import CDFPlot, ChargeDistributionPlot, PDFPlot


def main():
    # specify the data input directory, usually the i3 file set
    resource = Path("/data/i3store/users/blaufuss/data/alert_catalog_v2/sim_21220_alerts")

    # define the processing chain and run each process sequentially left to right
    for stage in [FeatureExtractor, FeatureProcessor, DatasetSplitter]:
        processor = stage(resource)
        resource = processor()

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(*resource)

    trainer = Trainer(dataset_registry)
    trainer.run()


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
    config_path = Path("./config/config.yaml")
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)
    test_cdf()
