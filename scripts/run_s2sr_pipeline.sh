#!/bin/bash

export CDSE_USERNAME="morganrsteckler@gmail.com" # supply your own username
export CDSE_PASSWORD_FILE="$HOME/.cdse/cdse_pw" # create your own secure password file
export GDAL_VRT_ENABLE_PYTHON=YES

# determine if mpirun or mpiexec is available to launch script
LAUNCH=$(command -v mpirun || command -v mpiexec)
if [ -z "$LAUNCH" ]; then
  echo "No MPI launcher found. Install OpenMPI/MPICH or load an HPC module." >&2
  exit 1
fi

# --bbox here is a 3x3-degree box centered over Toolik station
$LAUNCH -x CDSE_USERNAME -x CDSE_PASSWORD_FILE -n 10 python s2sr_pipeline.py \
  --bbox -150 67 -148 69 \
  --grid_csv /mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr/panarctic_grid/panarctic_0p25_gridcells.csv \
  --tr 0.000179663056824 0.000179663056824 \
  --tap \
  --start 2019-06-01T00:00:00Z --end 2019-08-31T00:00:00Z \
  --bands B02 B03 B04 B05 B06 B07 B8A B11 B12 \
  --work_dir /mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr \
  --keep-tmp  # if keeping tmp files, you must manually delete them from tmp/ and logs/ before running again