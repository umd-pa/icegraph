#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# Fail fast + trace commands with timestamps
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND exited with status $?"; exit 1' ERR

# CVMFS python interpreter, passed in by the user, and optionally the IceTray release, for example:
#   /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/bin/python3 v1.12.1
PY_BIN="${1:?Usage: $0 /path/to/cvmfs/py3-vX/<OS_arch>/bin/python [icetray version, default v1.12.1]}"
ICETRAY="${2:-v1.12.1}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "[ERROR] '$PY_BIN' is not an executable file" >&2
  exit 1
fi

# derive layout from the interpreter path:
#   .../<CVMFS_ROOT>/<os_arch>/bin/python
os_dir="$(dirname "$(dirname "$PY_BIN")")"
os_arch="$(basename "$os_dir")"
CVMFS_ROOT="$(dirname "$os_dir")"
venv="venv_$os_arch"

if [[ ! -f "$CVMFS_ROOT/setup.sh" ]]; then
  echo "[ERROR] No setup.sh found at '$CVMFS_ROOT/setup.sh'" >&2
  echo "Is '$PY_BIN' really a CVMFS .../<OS_arch>/bin/python path?" >&2
  exit 1
fi

# derive the python version
pyver="$("$PY_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"

# build virtual environment
echo "Building virtual environment..."
"$PY_BIN" -m venv "$venv"

# activate icetray (sets SROOT, etc.)
echo "Activating IceTray and sourcing from virtual environment..."
eval "$("$CVMFS_ROOT/setup.sh")"
: "${SROOT:?SROOT should be set by IceTray setup}"

# newer IceTray releases launch through bin/icetray-shell, older ones through env-shell.sh
metaproject="metaprojects/icetray/$ICETRAY"
if [[ -x "$SROOT/$metaproject/bin/icetray-shell" ]]; then
  launcher="$metaproject/bin/icetray-shell"
elif [[ -x "$SROOT/$metaproject/env-shell.sh" ]]; then
  launcher="$metaproject/env-shell.sh"
else
  echo "[ERROR] No IceTray $ICETRAY launcher found under '$SROOT/$metaproject'" >&2
  exit 1
fi

# source venv
source "$venv/bin/activate"
export PYTHONPATH="$venv/lib/$pyver/site-packages:${PYTHONPATH:-}"

# sanity info
which python
python -V
pip -V
echo "SROOT=$SROOT"
echo "IceTray launcher: $SROOT/$launcher"

# install dependencies (verbose pip)
echo "Installing dependencies, this may take a while..."
"$SROOT/$launcher" \
  "$venv/bin/$pyver" -m pip install -v --progress-bar on -r requirements.txt

echo "Generating initicetray.sh..."
cat <<EOF > initicetray.sh
#!/bin/bash
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

venv="$venv"

eval "\$($CVMFS_ROOT/setup.sh)"
source "\$venv/bin/activate"

# Force Python to prioritize virtual environment
export PYTHONPATH="\$venv/lib/$pyver/site-packages:\${PYTHONPATH:-}"

"\$SROOT/$launcher" "\$venv/bin/$pyver" "\$@"
EOF

chmod +x initicetray.sh

echo "Install complete."
