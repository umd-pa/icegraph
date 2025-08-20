#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# change this to the path to your virtual environment if necessary
venv_path="venv"

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
source "$venv_path/bin/activate"

# Force Python to prioritize virtual environment
export PYTHONPATH="$venv_path/lib/python3.11/site-packages:$PYTHONPATH"

"$SROOT"/metaprojects/icetray/v1.12.1/env-shell.sh "$venv_path/bin/python3.11" "$@"