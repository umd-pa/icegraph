#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# Fail fast + trace commands with timestamps
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND exited with status $?"; exit 1' ERR

os_arch="Ubuntu_20.04_x86_64"
venv="venv_$os_arch"

PY_BIN="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/$os_arch/bin/python"

# build virtual environment
echo "Building virtual environment..."
"$PY_BIN" -m venv $venv

# activate icetray (sets SROOT, etc.)
echo "Activating IceTray and sourcing from virtual environment..."
eval "$(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)"
: "${SROOT:?SROOT should be set by IceTray setup}"

# source venv
source $venv/bin/activate
export PYTHONPATH="$venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

# sanity info
which python
python -V
pip -V
echo "SROOT=$SROOT"

# install dependencies (verbose pip)
echo "Installing dependencies, this may take a while..."
"$SROOT/metaprojects/icetray/v1.12.1/env-shell.sh" \
  $venv/bin/python3.11 -m pip install -v --progress-bar on -r requirements.txt

echo "Generating initicetray.sh..."
cat <<EOF > initicetray.sh
#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

venv=$venv

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
source "\$venv/bin/activate"

# Force Python to prioritize virtual environment
export PYTHONPATH="\$venv/lib/python3.11/site-packages:$PYTHONPATH"

"$SROOT"/metaprojects/icetray/v1.12.1/env-shell.sh "\$venv/bin/python3.11" "$@"
EOF

echo "Install complete."
