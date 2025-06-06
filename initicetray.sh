#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

export HDF5_DISABLE_VERSION_CHECK=1

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
source /data/i3home/tstjean/icegraph/venv/bin/activate

# Force Python to prioritize virtual environment
export PYTHONPATH=/data/i3home/tstjean/icegraph/venv/lib/python3.11/site-packages:$PYTHONPATH

"$SROOT"/metaprojects/icetray/v1.8.2/env-shell.sh /data/i3home/tstjean/icegraph/venv/bin/python3.11 "$@"