import argparse

from icegraph.trainer import Trainer
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input LMDB dataset. Can be a file, list of files, or a directory."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output directory. All generated files will be saved here."
    )

    args = parser.parse_args()

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(args.source)

    # use dataset_registry to pass data to training system
    # optionally define specific callbacks to pass to trainer:
    #
    # callbacks = [ConsoleCallback(), TensorBoardCallback(), ExportCallback()]
    # with Trainer(dataset_registry, outdir=args.output, callbacks=callbacks) as ...:
    #
    # or, if you don't want to overwrite existing default callbacks
    #
    # with Trainer(...) as trainer:
    #     trainer.register_callback(RegressionMetricsCallback)

    with Trainer(dataset_registry, outdir=args.output) as trainer:
        trainer.run()

if __name__ == "__main__":
    main()