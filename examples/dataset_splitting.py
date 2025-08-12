import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import argparse

from icegraph.data.splitter import SplitMapBuilder
from icegraph.config import IGConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the LMDB dataset. Can be a file, list of files, or a directory."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output file. This is where the generated map file will be saved."
    )

    args = parser.parse_args()

    config_path = "./config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # run the dataset splitting
    splitter = SplitMapBuilder(args.input)
    splitter.build_map(args.output)


if __name__ == "__main__":
    main()