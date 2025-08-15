import os

# Disable the fatal HDF5 version‐mismatch check
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import argparse
from pathlib import Path

from icegraph.data.transformer import FeatureProcessor
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
        help="Path to the output file"
    )

    args = parser.parse_args()

    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # run the dataset transformation
    transformer = FeatureProcessor(args.input)
    transformer.process(args.output)


if __name__ == "__main__":
    main()