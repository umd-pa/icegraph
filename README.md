# IceGraph

IceGraph is a work-in-progress pipeline for training Graph Neural Networks for reconstruction/classification work on IceCube data using PyTorch.

Documentation: https://umd-pa.github.io/icegraph

## Installation

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

Next, install dependencies:

```
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --find-links https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

Install IceGraph:

```
pip install .
```

Example scripts are located under icegraph/examples. Before running, modify the config file at icegraph/config/config.yaml, and update the config path to point to this file within each example file you intend to run.