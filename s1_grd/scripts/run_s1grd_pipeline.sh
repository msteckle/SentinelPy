#!/bin/bash
set -euo pipefail

# Point to your venv and OpenMPI install here
VENV_DIR="/mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr/.venv"
OPENMPI_PREFIX="$HOME/opt/openmpi-5.0.5" 

# Activate venv
source "${VENV_DIR}/bin/activate"

# Ensure mpirun is on PATH and libmpi is visible
if ! command -v mpirun >/dev/null 2>&1; then
  if [ -x "${OPENMPI_PREFIX}/bin/mpirun" ]; then
    export PATH="${OPENMPI_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OPENMPI_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export OPAL_PREFIX="${OPENMPI_PREFIX}"
  else
    echo "ERROR: mpirun not found. Set OPENMPI_PREFIX or adjust OPENMPI_PREFIX above." >&2
    exit 1
  fi
fi

export EARTHDATA_USER="msteckler98" # supply your own username
export EARTHDATA_PASS="$(< $HOME/.earthdata/earthdata_pw)" # supply your own password
export GDAL_VRT_ENABLE_PYTHON=YES

# --bbox here is a 3x3-degree box centered over Toolik station
mpirun -x EARTHDATA_USER -x EARTHDATA_PASS -n 10 python s1grd_pipeline.py \
  --bbox -150 67 -148 69 \
  --work_dir /mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd \
  --grid_csv /mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr/panarctic_grid/panarctic_0p25_gridcells.csv \
  --start 2019-06-01T00:00:00Z --end 2019-08-31T00:00:00Z \
  --snap-xml /mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd/scripts/GEEPreprocessing.xml \
  --snap-props /mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd/scripts/GEEPreprocessing.properties \
  --gpt-bin /mnt/poseidon/remotesensing/6ru/apps/snap-9.0.0/bin/gpt \
  --snap-outdir /mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd/tmp/snap_outputs \
  --snap-prefix Orb_NR_Cal_TC \
  --snap-format GeoTIFF-BigTIFF \
  --tr 0.000179663056824 0.000179663056824 \
  --tap \