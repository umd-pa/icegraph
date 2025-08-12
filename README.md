# IceGraph

[![CodeQL](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql)

IceGraph is a work-in-progress pipeline for training Graph Neural Networks for reconstruction/classification work on IceCube data using PyTorch.

Documentation: https://umd-pa.github.io/icegraph

## Installation (Ubuntu)

Install the package via git clone:

```
git clone git@github.com:umd-pa/icegraph.git
cd icegraph
```

First step after cloning the repository is to create the virtual environment using the version of python packaged with CVMFS and sourcing from it:

```
/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/bin/python -m venv venv
source venv/bin/activate
```

Next, install dependencies:

```
pip install -r requirements.txt
```

Finally, install IceGraph:

```
pip install .
```

This software must be run within the IceTray environment.

Example scripts are located under icegraph/examples. Before running these, tailor the config file at icegraph/config/config.yaml, and update the IGConfig config path to point to this file within each example file you intend to run.

## Usage

This program has two primary functions; loading and processing data from I3 files into an ML friendly format (in this case Lightning Memory-Mapped Database, or LMDB), and training GNN's using the PyTorch framework.

To get from I3 --> trained model, the pipeline is as follows:

### Non-parallelized (small datasets, testing configurations, etc)

Required imports:
```
from pathlib import Path

from icegraph.data.processor import FeatureProcessor
from icegraph.data.extractor import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.splitter import DatasetSplitter
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

Set the path to your I3 file(s). This can be either a path to one I3 file, or to a directory containing multiple I3 files.
```
resource = Path("path/to/i3_file(s)")
```

Extract data from I3 files and process it. This is done by first running the `FeatureExtractor` module (which accepts a , then running 
```
for stage in [FeatureExtractor, FeatureProcessor]:
    processor = stage(resource)
    resource = processor()
```

Generate the split mapping file using the `SplitMapBuilder`:

```
split_map_file = SplitMapBuilder(resource)
```

Load the split data and register it. The DatasetRegistry class acts as an interface between the training system and the formatted data.
```
dataset_registry = DatasetRegistry.load_from_lmdb(*resource)
```

Pass the dataset registry instance to a Trainer, then run the training. Training configuration and hyperparameter selection is all done via config.yaml. The trained model is automatically saved on each epoch.
```
outfile = Path("path/to/model.pt")
trainer = Trainer(dataset_registry, outfile=outfile)
trainer.run()
```

Or, you can optionally specify callbacks to use during training. You can also define custom callbacks if necessary.
```
from icegraph.trainer.callbacks import ConsoleCallback, CheckpointCallback, TensorBoardCallback

# these are the default callbacks used in Trainer
# if you only need these callbacks, there is no need to pass them manually
callbacks = [ConsoleCallback(), TensorBoardCallback(), CheckpointCallback()]

outfile = Path("path/to/model.pt")
trainer = Trainer(dataset_registry, outfile=outfile, callbacks=callbacks)
trainer.run()
```

### Parallelized (large datasets)
