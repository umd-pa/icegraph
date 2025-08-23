import os

# Disable the fatal HDF5 version‐mismatch check
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import argparse
from pathlib import Path

from icegraph.data.pipeline import Pipeline
from icegraph.data.processor import FeatureProcessor, EdgeProcessor, TruthProcessor, StandardSplitAllocator
from icegraph.data.writers import LMDBWriter
from icegraph.config import IGConfig


def main() -> None:
    # argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input source"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output directory"
    )

    args = parser.parse_args()

    config_path = Path("/data/i3home/tstjean/icegraph/config/config.yaml")
    config = IGConfig(config_path)

    # register the config instance
    IGConfig.register(config)

    with Pipeline() as pipeline:
        pipeline.build(
            extractor=FeatureExtractor,
            processors=[FeatureProcessor, TruthProcessor, EdgeProcessor, StandardSplitAllocator],
            writer=LMDBWriter
        )
        pipeline.configure(args.input, outdir=args.output)
        pipeline.execute()


if __name__ == "__main__":
    main()