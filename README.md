# S2SR Pipeline — download -> process -> composite (Sentinel-2 L2A)

This pipeline pulls Sentinel-2 L2A scenes from Copernicus Data Space (CDSE), applies cloud/snow masking and PB harmonization, warps to WGS84, and builds per-gridcell median composites.

---

## What you’ll need

* **Python 3.10+** with: `numpy`, `pandas`, `requests`, `shapely`, `mpi4py`, `osgeo` (GDAL Python bindings)
* **GDAL** (command-line tools available in `PATH`)
* A **CDSE account** (username + password)
* A spatial grid CSV created by `make_grid.py` (e.g., Pan-Arctic 0.25° grid)

---

## Directory layout (created under `--work_dir`)

```
imagery_raw/     # downloaded .SAFE products (JP2 + XML)
indices/         # manifests, scene index, error logs
logs/            # per-MPI-rank logs
tmp/             # per-rank VRTs & intermediates (safe to delete)
composites/      # final per-gridcell GeoTIFFs
```

---

## Quick start

1. **Create a secure password file (recommended):**

```bash
mkdir -p ~/.cdse && chmod 700 ~/.cdse
printf '%s\n' 'YOUR_PASSWORD' > ~/.cdse/cdse_pw
chmod 600 ~/.cdse/cdse_pw
```

2. **Run the pipeline (example):**

```bash
export CDSE_USERNAME="you@example.org"
export CDSE_PASSWORD_FILE="$HOME/.cdse/cdse_pw"
export GDAL_VRT_ENABLE_PYTHON=YES

mpirun -x CDSE_USERNAME -x CDSE_PASSWORD_FILE -n 10 python s2sr_pipeline.py \
  --bbox -149.725 68.505 -149.475 68.755 \
  --grid_csv /path/to/panarctic_0p25_gridcells.csv \
  --start 2019-06-01T00:00:00Z --end 2019-08-31T00:00:00Z \
  --bands B02 B03 B04 B05 B06 B07 B8A B11 B12 \
  --tr 0.000179663056824 0.000179663056824 --tap \
  --work_dir /path/to/s2_sr \
  --keep-tmp
```

> Tip: `--tr` sets output pixel size in degrees (WGS84). Use `--tap` to align pixels to that grid. For consistent mosaics, keep `--tr/--tap` the same across runs.

---

## What the pipeline does

1. **Select gridcells**
   Uses your `--bbox` to pick gridcells from `--grid_csv`. Saves selected cells to `indices/selected_gridcells.csv`.

2. **Query CDSE**
   Queries the union envelope of selected gridcells within `--start/--end` for **S2MSI2A** products.

3. **Download (idempotent)**
   Downloads only missing 20 m bands you list in `--bands` **plus `SCL`** into `imagery_raw/`.

4. **Phase 1 (MPI, per-scene)**
   For each band image:

   * Parse **PB** from `.SAFE` name; if **PB ≥ 4.00**, subtract **1000 DN** (harmonization).
   * **Mask** with `SCL` (classes: `0,1,2,3,7,8,9,10,11`).
   * **Warp** to WGS84 (resampling: bilinear) into **per-rank VRTs** under `tmp/rank_*/wgs84_vrt/`.

5. **Phase 2 (MPI, per-gridcell-&-band)**
   For each gridcell & band:

   * Find warped scenes intersecting the cell.
   * Build a **stack VRT** (cropped to cell bbox).
   * Compute a **masked median VRT** (Float32; `nodata=-9999`).
   * **Clip to exact cell polygon** and write **one final GeoTIFF** to `composites/<cell_id>/`.

---

## Masking & harmonization details

* **SCL mask**: pixels with codes `0,1,2,3,7,8,9,10,11` are set to nodata (65535) before compositing.
* **PB offset**: if processing baseline **PB ≥ 4.00**, **subtract 1000 DN** on valid pixels (no wrap) to harmonize.
* **Median**: computed per-pixel over the valid stack (NaNs ignored). Output `Float32`, nodata `-9999`.

---

## Re-runs & consistency

* The pipeline **overwrites** intermediate VRTs and final composites to prevent seams when you change `--tr/--tap` or AOI.
* If you want to keep multiple grid specs side-by-side, use distinct `--work_dir` paths (or encode a tag in them).

---

## Common options

* `--bbox xmin ymin xmax ymax` AOI in lon/lat (WGS84)
* `--grid_csv PATH` Grid CSV from `make_grid.py`
* `--start / --end` ISO datetimes for CDSE query window
* `--bands ...` Any of `{B02,B03,B04,B05,B06,B07,B8A,B11,B12}` (20 m). `SCL` is auto-included.
* `--tr dLon dLat` Target pixel size (deg) in WGS84
* `--tap` Align pixels to the target grid
* `--keep-tmp` Keep `tmp/` for debugging/reuse (otherwise safe to delete)

---

## Logs & artifacts

* **Logs**: `logs/pipeline_rank*.log`
* **Query manifest**: `indices/query_manifest.csv`
* **Scene index**: `indices/scene_index.csv`
* **Phase errors**: `indices/phase1_errors.csv`, `logs/phase2_errors.csv` (if any)
* **Final outputs**: `composites/<cell_id>/cell_<id>_<band>_median_clip.tif`

---

## Notes on performance

* Use `mpirun -n <ranks>` to scale; Phase 1 parallelizes by **scene**, Phase 2 by **gridcell × band**.
* Internally sets `GDAL_NUM_THREADS=1` for PixelFunction stability under MPI (avoid thread oversubscription).
* Keep `--tr/--tap` uniform across tiles to ensure perfect edge alignment for downstream mosaics.

---