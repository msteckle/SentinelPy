# Sentinel-1 GRD Pipeline — README

This repository provides a command-line workflow to **discover, download, and pre-process Sentinel-1 GRD scenes** with SNAP, preparing them for tiling/mosaicking and (optionally) dB packing. The current script performs:

1) **ASF discovery** over an AOI & date window  
2) **Bulk download** of matching ZIPs (with Earthdata credentials)  
3) **SNAP GPT preprocessing** per scene (orbit files + EGM96 handling)  
4) **Discovery of processed products** for downstream mosaics (staging)

> Note: grid-based VRT stacking/median compositing (Stages III–IV) are scaffolded in the code and can be enabled/extended as needed. The script already prepares all prerequisites and lists processed products to feed that step.

---

## Contents

- [Overview & Workflow](#overview--workflow)  
- [Requirements](#requirements)  
- [Environment Variables](#environment-variables)  
- [Input Data & Files](#input-data--files)  
- [Directory Layout](#directory-layout)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Examples](#examples)  
- [How It Works](#how-it-works)  
- [Performance Tips](#performance-tips)  
- [Troubleshooting](#troubleshooting)  
- [Extending to VRT/Median/UInt16 dB](#extending-to-vrtmedianuint16-db)

---

## Overview & Workflow

**Goal:** Given a bounding box and date range, find Sentinel-1 GRD scenes, download them, run a SNAP GPT graph for calibration/terrain correction (or your custom graph), and stage outputs for mosaicking.

**High-level stages**

- **I. Discover + Download**  
  - Query ASF/CMR by AOI, beam mode, product levels, flight direction.  
  - Select the predominant flight direction (or force ASC/DESC).  
  - Save a `s1_manifest.csv` and download all ZIPs.

- **II. SNAP Preprocessing**  
  - Ensure **POEORB** orbits are available in the SNAP userdir.  
  - Ensure **EGM96** geoid grid is present.  
  - Run **SNAP GPT** with your XML + properties for each ZIP, producing BEAM-DIMAP outputs (`.dim` + `*.img`).

- **III–IV. (Scaffolded)**  
  - Build per-cell VRT stacks by polarization, add PixelFunction for **median**, reproject/resample as needed, and pack to **UInt16 dB** using `--db-min/--db-max`.  
  - These blocks are present but commented; you can enable/finish them for end-to-end tiling.

---

## Requirements

- **Python 3.10+**
- Python packages:
  - `pandas`, `geopandas`, `shapely`
  - `mpi4py`
  - `GDAL` Python bindings (`osgeo.gdal`)
- **SNAP** (ESA Sentinel Application Platform) with **gpt** on PATH  
  - Tested with SNAP 9.x
- **SNAP userdir** with write access (for auxdata: POEORB, EGM96)
- **Earthdata Login** credentials (for ASF downloads)
- Command-line access to `gpt` (SNAP) and networking to ASF endpoints

---

## Environment Variables

The script accepts credentials/paths via flags or environment variables.

- `EARTHDATA_USER` / `EARTHDATA_PASS` — for ASF downloads  
- `SNAP_USER_DIR` — persistent SNAP user directory (auxdata cache)  
- (Set automatically in-process) `GDAL_VRT_ENABLE_PYTHON=YES` during SNAP runs

You can also provide `--earthdata-user`, `--earthdata-pass`, and `--snap-userdir` flags instead of env vars.

---

## Input Data & Files

- **AOI**: Provided as `--bbox xmin ymin xmax ymax` (EPSG:4326).
- **Pan-Arctic grid CSV**: `--grid_csv path.csv`  
  Required columns: `xmin, ymin, xmax, ymax` (optional `tile_id_rc` if you want persistent IDs).  
- **SNAP graph XML**: `--snap-xml path/to/graph.xml`  
- **SNAP properties**: `--snap-props path/to/graph.properties`  
  (Typical GPT pattern: use XML for the graph, a `.properties` file for parameter substitution.)
- **Working & output dirs**: `--work_dir` and `--snap-outdir`

---

## Directory Layout

At runtime, the script creates:
```
<work_dir>/
    imagery_raw/ # downloaded S1A/S1B *.zip
    tmp/ # temp work area
    indices/ # s1_manifest.csv and other indices
    logs/ # one log per MPI rank
    <snap_outdir>/ # SNAP BEAM-DIMAP outputs (.dim + .img)
<snap_userdir>/ # SNAP auxdata cache (POEORB, EGM96, etc.)
```

---

## Installation

Using `uv` (recommended) or `pip`.

### Create environment (uv)
```bash
uv venv
uv pip install pandas geopandas shapely mpi4py GDAL
```

---

## Usage

```bash
mpirun -x EARTHDATA_USER -x EARTHDATA_PASS -n 10 python s1grd_pipeline.py \
  --bbox -150 67 -148 69 \
  --grid_csv /path/to/panarctic_grid.csv \
  --work_dir /path/to/workdir \
  --start 2019-06-01 --end 2019-06-30 \
  --beam IW \
  --levels GRD_HD GRD_FD GRD_HS GRD_MD GRD_MS \
  --flightdir predominant \
  --max-download-workers 6 \
  --snap-xml /path/to/graph.xml \
  --snap-props /path/to/graph.properties \
  --gpt-bin gpt \
  --snap-userdir /path/to/.snap \
  --snap-outdir /path/to/snap_outputs \
  --snap-prefix Orb_NR_Cal_TC \
  --dst_srs EPSG:4326 \
  --tr 0.00025 0.00025 \
  --tap \
  --pols VV VH \
  --db-min -50 --db-max 1
```
> Note: `mpirun` is optional but recommended for parallel downloads and SNAP processing. You can run without it, but performance will be slower.
> Note: Securely store your Earthdata credentials by using environment variables or a credentials manager.

### Key flags
- **AOI & grid:** --bbox, --grid_csv
- **ASF query:** --start, --end, --beam, --levels, --flightdir
- **Downloads:** --max-download-workers, --earthdata-user, --earthdata-pass
- **SNAP:** --snap-xml, --snap-props, --gpt-bin, --snap-userdir, --snap-outdir, --snap-prefix, --snap-overwrite
- **Reprojection / mosaic grid:** --dst_srs, --tr xres yres, --tap, --pols
- **dB packing:** --db-min, --db-max (used in Stage IV when enabled)

---

## How It Works

1. ASF Discovery (asf_search_aoi)
- Builds a manifest of URLs matching AOI/time/beam/level.
- If --flightdir predominant, computes the majority direction and filters to it.

2. Downloading (download_asf_urls)
- Concurrent workers fetch ZIPs into work_dir/imagery_raw.
- Requires Earthdata credentials.

3. SNAP Preparation
- Ensures POEORB orbits exist in SNAP_USER_DIR (ensure_poeorb_via_s1_orbits).
- Ensures EGM96 grid present (ensure_egm96_present).

4. SNAP GPT Execution (run_snap_gpt)
- Executes your graph.xml with graph.properties.
- Outputs to BEAM-DIMAP (.dim + one or more *.img) in --snap-outdir.
- Existing outputs are skipped unless --snap-overwrite is provided (in which case stale BEAM-DIMAP is removed first via remove_beam_dimap).

5. Staging for Composites
- Discovers processed *.img files for downstream mosaicking.
- (Commented code shows how to index footprints via gdal.Info, intersect with grid cells, build per-cell/per-pol VRT stacks, and add PixelFunctions for medians.)