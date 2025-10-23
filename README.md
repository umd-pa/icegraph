![image](https://github.com/umd-pa/icegraph/tree/main/img/icegraph light.png)

[![CodeQL](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql)

IceGraph is a work-in-progress pipeline for training Graph Neural Networks for reconstruction/classification work on IceCube data using PyTorch.

Documentation: https://umd-pa.github.io/icegraph

## Installation (Ubuntu 22.04)

Install the package via git clone:

```
git clone git@github.com:umd-pa/icegraph.git
cd icegraph
```

Make the install script executable and run it:

```
chmod +x install.sh
./install.sh
```

WARNING: You may have to modify the OS version and architecture within the install.sh script depending on your system configuration.

This software must be run within the IceTray environment.

Example scripts are located under [examples](https://github.com/umd-pa/icegraph/tree/main/examples). Before running these, tailor the config file at [config/config.yaml](https://github.com/umd-pa/icegraph/blob/main/config/config.yaml), and update the IGConfig config path to point to this file within each example file you intend to run.

## Usage

This program has two primary functions; loading and processing data from I3 files into an ML friendly format (in this case Lightning Memory-Mapped Database, or LMDB), and training GNN's using the PyTorch framework.

To get from a set of I3 files to a trained model, the pipeline is as follows:

Required imports:
```
from pathlib import Path

from icegraph.data.processor import FeatureProcessor, TruthProcessor, EdgeProcessor, StandardSplitAllocator
from icegraph.data.extractor import FeatureExtractor
from icegraph.data.writers import LMDBWriter
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.pipeline import Pipeline
from icegraph.trainer import Trainer
```

Define and register project configurations:
```
# define an IGConfig instance from a config.yaml
config_path = Path("path/to/config.yaml")
config = IGConfig(config_path)

# register it for global access
IGConfig.register(config)
```

Set the path to your I3 file(s). This can be either a path to one I3 file, a list of paths to multiple I3 files, or a directory containing one or more I3 files.
```
source = Path("path/to/i3_file(s)")
```

Extract data from I3 files and process it. This is done by first running the `FeatureExtractor` module, then running:
```
with Pipeline() as pipeline:
    pipeline.build(
        extractor=FeatureExtractor,
        processors=[FeatureProcessor, TruthProcessor, EdgeProcessor, StandardSplitAllocator],
        writer=LMDBWriter
    )
    pipeline.configure(args.input, outdir=args.output)
    pipeline.execute()
```
The pipeline can be run in parallel as jobs. This is computationally intensive, thus for larger datasets it is highly recommended to parallelize.


Load the data to a registry. The `DatasetRegistry` class acts as an interface between the training system and the formatted data.
```
dataset_registry = DatasetRegistry.load_from_lmdb(source)
```

Pass the dataset registry instance to a `Trainer`, then run the training. Training configuration and hyperparameter selection is all done via config.yaml.
```
outdir = Path("path/to/trainer/outdir")
with Trainer(dataset_registry, outdir=outdir) as trainer:
    trainer.run()
```

You can pass in custom callbacks if desired:

```
outdir = Path("path/to/trainer/outdir")
with Trainer(dataset_registry, outdir=outdir) as trainer:
    trainer.register_callback(CustomCallback)
    trainer.run()
```

Or override defaults:

```
outdir = Path("path/to/trainer/outdir")
callbacks = [List(), Of(), Callbacks()]
with Trainer(dataset_registry, outdir=outdir, callbacks=callbacks) as trainer:
    trainer.run()
```
