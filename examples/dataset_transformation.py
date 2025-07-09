import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import os

# Disable the fatal HDF5 version‐mismatch check
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import argparse

from icegraph.data.transform import TransformToDataset
from icegraph.config import IGConfig


def main() -> None:
    # argument parsing
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

    # run the dataset transformation
    transformer = TransformToDataset(args.input)
    transformer.process(args.output)


if __name__ == "__main__":
    main()