![image](<img/logo-dark.png>)

[![CodeQL](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/umd-pa/icegraph/actions/workflows/github-code-scanning/codeql)

IceGraph is an end-to-end pipeline for building and deploying Graph Neural Networks for reconstruction/classification work in IceCube.

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
./install.sh /path/to/cvmfs/py3-vX/<OS_arch>/bin/python
```

This software must be run via the provided python shim when using functionality that requires access to IceCube's [IceTray](https://github.com/icecube/icetray) (i.e. for feature extraction during data processing, if using the standard I3 extractor which uses [ml_suite](https://github.com/icecube/icetray/tree/main/ml_suite)). This shim is provided as `initicetray.sh`, which is used in-place of `python3` (`python3 main.py` becomes `initicetray.sh main.py`).

## Quick Start

This framework has three primary functions; loading and processing data from I3 files into an ML friendly format, training GNN's using the PyTorch framework, and running inference.

The workflow is as follows:

### Logging

IceGraph provides a default logging setup function you can use, or you could write your own.
```
from icegraph.logging import configure_logging

configure_logging(
    level="debug"
)
```

### Processing

Required imports:
```
from pathlib import Path
from icegraph.data import Pipeline
```

Set path to configuration file.
```
config_path: str | Path = Path("path/to/config")
```

Set the path to your I3 file(s). This can be either a path to one I3 file, a list of paths to multiple I3 files, or a directory containing one or more I3 files.
```
source_files: str | Path | list[str | Path] = Path("path/to/dir(s)_or_file(s)")
```

Initialize the pipeline from config and source files, then execute to process.
```
with Pipeline.from_yaml(source_files, config_path) as pipeline:
    pipeline.execute()
```
The pipeline can be run in parallel as separate processes.

### Training

Required imports:
```
from pathlib import Path
from icegraph.training import Trainer
```

Set path to configuration file.
```
config_path: str | Path = Path("path/to/config")
```

Initialize the training engine from config, then execute to begin training.
```
with Trainer.from_yaml(config_path) as trainer:
    trainer.execute()
```

Engines can be run in distributed mode using the `Distributed` wrapper:

```
from icegraph.engine import Distributed

with Distributed(Trainer).from_yaml(config_path) as trainer:
    trainer.execute()
```

You can register built-in or custom callbacks before execution:

```
from icegraph.trainer.callbacks import ConsoleCallback, CallbackSpec

with Trainer.from_yaml(config_path) as trainer:
    # register the callback, this can also be done in distributed mode
    trainer.register_callback(
        CallbackSpec(
            callback=ConsoleCallback,
            kwargs: {}
        )
    ) 

    trainer.execute()
```

### Inference

Required imports:
```
from pathlib import Path
from icegraph.inference import BatchInference
```

Set path to configuration file.
```
config_path: str | Path = Path("path/to/config")
```

Initialize the inference engine from config, then execute to begin inference.
```
with BatchInference.from_yaml(config_path) as inference:
    inference.execute()
```

Engines can be run in distributed mode using the `Distributed` wrapper:

```
from icegraph.engine import Distributed

with Distributed(BatchInference).from_yaml(config_path) as inference:
    inference.execute()
```

>WARNING: Running inference in distributed mode may currently result in data loss; chunks may be dropped so sample count matches across ranks. This is intended for training, but is planned to be fixed for inference in a future update.

You can register built-in or custom callbacks before execution:

```
from icegraph.inference.callbacks import ConsoleCallback, CallbackSpec

with BatchInference.from_yaml(config_path) as inference:
    # register the callback, this can also be done in distributed mode
    inference.register_callback(
        CallbackSpec(
            callback=ConsoleCallback,
            kwargs: {}
        )
    ) 

    inference.execute()
```
