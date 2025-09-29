#!/bin/bash

# Point to your venv and OpenMPI install here
VENV_DIR="/mnt/poseidon/remotesensing/arctic/data/rasters/esa_sentinel/.venv"
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

export CDSE_USERNAME="morganrsteckler@gmail.com" # supply your own username
export CDSE_PASSWORD_FILE="$HOME/.cdse/cdse_pw" # create your own secure password file
export GDAL_VRT_ENABLE_PYTHON=YES

# --bbox here is a 3x3-degree box centered over Toolik station
mpirun -x CDSE_USERNAME -x CDSE_PASSWORD_FILE -n 10 python s2sr_pipeline.py \
  --bbox -150 67 -148 69 \
  --work_dir /mnt/poseidon/remotesensing/arctic/data/rasters/esa_sentinel/s2_sr \
  --grid_csv /mnt/poseidon/remotesensing/arctic/data/rasters/esa_sentinel/s2_sr/panarctic_grid/panarctic_0p25_gridcells.csv \
  --start 2019-06-01T00:00:00Z --end 2019-08-31T00:00:00Z \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --bands_res 20 \
  --mask-classes 0 1 2 3 7 8 9 10 11 \
  --tr 0.000179663056824 0.000179663056824 \
  --tap \
  --keep-tmp  # if keeping tmp files, you must manually delete them from tmp/ and logs/ before running again