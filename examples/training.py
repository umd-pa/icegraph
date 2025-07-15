import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import argparse

from icegraph.trainer import Trainer
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to the input training LMDB"
    )
    parser.add_argument(
        "--val-file",
        required=True,
        help="Path to the input validation LMDB"
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to the input testing LMDB"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output file"
    )
    parser.add_argument(
        '--tensorboard',
        action=argparse.BooleanOptionalAction
    )
    parser.set_defaults(
        tensorboard=False
    )

    args = parser.parse_args()

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # load all data
    dataset_registry = DatasetRegistry.load_from_lmdb(args.train_file, args.val_file, args.test_file)

    # use dataset_registry to pass data to training system
    trainer = Trainer(dataset_registry, args.output)
    trainer.run(tensorboard=args.tensorboard)

if __name__ == "__main__":
    main()