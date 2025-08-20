#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# Fail fast + trace commands with timestamps
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND exited with status $?"; exit 1' ERR

PY_BIN="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/bin/python"

# build virtual environment
echo "Building virtual environment..."
"$PY_BIN" -m venv venv

# activate icetray (sets SROOT, etc.)
echo "Activating IceTray and sourcing from virtual environment..."
eval "$(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)"
: "${SROOT:?SROOT should be set by IceTray setup}"

# source venv
source venv/bin/activate
export PYTHONPATH="venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

# sanity info
which python
python -V
pip -V
echo "SROOT=$SROOT"

# install dependencies (verbose pip)
echo "Installing dependencies, this may take a while..."
"$SROOT/metaprojects/icetray/v1.12.1/env-shell.sh" \
  venv/bin/python3.11 -m pip install -v --progress-bar on -r requirements.txt

echo "Install complete."
