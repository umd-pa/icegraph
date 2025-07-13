import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

import argparse

from icegraph.data.extractor import FeatureExtractor
from icegraph.config import IGConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output file"
    )

    args = parser.parse_args()

    config_path = "/data/i3home/tstjean/icegraph/config/config.yaml"
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    # run the feature extraction
    extractor = FeatureExtractor(args.input)
    extractor.extract(args.output)

if __name__ == "__main__":
    main()