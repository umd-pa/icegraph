#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# change this to the path to your virtual environment if necessary
venv_path="venv"
export HDF5_DISABLE_VERSION_CHECK=1

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
if [[ -d "$ROOT/src" ]]; then
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
fi

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
source "$venv_path/bin/activate"

# Force Python to prioritize virtual environment
export PYTHONPATH="$venv_path/lib/python3.11/site-packages:$PYTHONPATH"

"$SROOT"/metaprojects/icetray/v1.8.2/env-shell.sh "$venv_path/bin/python3.11" -m pytest "$@"