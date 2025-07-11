# IceGraph

IceGraph is a work-in-progress pipeline for training Graph Neural Networks for reconstruction/classification work on IceCube data using PyTorch.

Documentation: https://umd-pa.github.io/icegraph

## Installation (Ubuntu)

---
Install the package via git clone:

```
git clone github@github.com:umd-pa/icegraph
cd icegraph
```

First step after cloning the repository is to create the virtual environment using the version of python packaged with CVMFS and sourcing from it:

```
/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/bin/python -m venv venv
source venv/bin/activate
```

Next, install dependencies. IceGraph needs very specific versions of torch, so we need to tell pip where to find the correct wheels:

```
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --find-links https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

Finally, install IceGraph:

```
pip install .
```

This software must be run within the IceTray environment.

Example scripts are located under icegraph/examples. Before running these, tailor the config file at icegraph/config/config.yaml, and update the IGConfig config path to point to this file within each example file you intend to run.

## Usage

---
This program has two primary functions; loading and processing data from I3 files into an ML friendly format (in this case Lightning Memory-Mapped Database, or LMDB), and training GNN's using the PyTorch framework.

To get from I3 --> trained model, the pipeline is as follows:

### Non-parallelized (small datasets, testing configurations, etc)

Required imports:
```
from pathlib import Path

from icegraph.data.transform import TransformToDataset
from icegraph.data.extract import FeatureExtractor
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.data.split import DatasetSplitter
from icegraph.train import Trainer
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

Define the processing chain. Each process in the list is run sequentially from left to right.
```
for stage in [FeatureExtractor, TransformToDataset, DatasetSplitter]:
    processor = stage(resource)
    resource = processor()
```

Load the split data and register it. The DatasetRegistry class acts as an interface between the training system and the formatted data.
```
dataset_registry = DatasetRegistry.load_from_lmdb(*resource)
```

Pass the dataset registry instance to a Trainer, then run the training. Training configuration and hyperparameter selection is all done via config.yaml. The trained model can be saved.
```
trainer = Trainer(dataset_registry)
trainer.run()

# save the model 
save_path = Path("path/to/model.pth")
trainer.save(save_path)
```

### Parallelized (large datasets)