import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import argparse

from icegraph.data.split import DatasetSplitter
from icegraph.config import IGConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output file"
    )

    args = parser.parse_args()

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # run the dataset splitting
    splitter = DatasetSplitter(args.input)
    splitter.generate_splits(args.output)


if __name__ == "__main__":
    main()