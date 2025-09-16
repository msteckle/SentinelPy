#!/usr/bin/env bash
set -euo pipefail

# Project root
ROOT="/mnt/poseidon/remotesensing/arctic/data/rasters/esa_sentinel"
cd "$ROOT"

# Paths to your tools
SNAP_BIN="/mnt/poseidon/remotesensing/6ru/apps/snap-9.0.0/bin/gpt"
OPENMPI_PREFIX="/home/6ru/openmpi-5.0.5"

# 0) Sanity checks
[ -x "$SNAP_BIN" ] || { echo "ERROR: SNAP gpt not found at $SNAP_BIN"; exit 1; }
[ -d "$OPENMPI_PREFIX" ] || { echo "ERROR: OpenMPI not found at $OPENMPI_PREFIX"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not on PATH. Install from https://docs.astral.sh/uv/"; exit 1; }
command -v gdal-config >/dev/null 2>&1 || { echo "ERROR: gdal-config not found. Install system GDAL devel package."; exit 1; }

# 1) Recreate the venv with uv (choose 3.11 or 3.10/3.9 if you prefer)
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate

echo "Using Python: $(python -V)  @ $(which python)"

# 2) Make OpenMPI visible so mpi4py builds against it
export PATH="$OPENMPI_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OPAL_PREFIX="$OPENMPI_PREFIX"

# 3) Pin GDAL Python bindings to the system GDAL version
GDAL_VER="$(gdal-config --version)"
GDAL_CFLAGS="$(gdal-config --cflags || true)"
GDAL_LDFLAGS="$(gdal-config --libs || true)"
echo "System GDAL version: $GDAL_VER"

# Some distros need these so building from source can find headers/libs if no wheel:
export CFLAGS="${CFLAGS:-} ${GDAL_CFLAGS}"
export LDFLAGS="${LDFLAGS:-} ${GDAL_LDFLAGS}"

# Optional (helps pyproj/GDAL find data if your system install needs it)
if gdal-config --datadir >/dev/null 2>&1; then
  export GDAL_DATA="$(gdal-config --datadir)"
fi
# If you know your PROJ share dir, set it (otherwise pyproj wheels bundle data)
# export PROJ_LIB="/usr/share/proj"

# 4) Install Python deps with uv (fast)
uv pip install -U pip wheel setuptools
uv pip install numpy pandas requests
uv pip install shapely pyproj geopandas
uv pip install "GDAL==${GDAL_VER}"            # will use wheel if available; else builds
uv pip install mpi4py                         # builds/binds against OpenMPI in PATH
uv pip install asf-search s1_orbits
uv pip install lxml