#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for S2 L2A Arctic processing:
1) select gridcells by AOI
2) compute max bbox & query CDSE
3) download missing files (rank 0)
4) PHASE 1 (MPI): per-scene -> SCL mask -> PB harmonize -> WGS84 VRT
5) PHASE 2 (MPI): per-gridcell -> pick overlapping scene VRTs -> median composite (Float32 reflectance)
6) cleanup tmp (optional)

Use this module:
Kick off this script using arguments specified in run_s2sr_pipeline.sh
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Sequence, Optional, List, Dict, Any
import pandas as pd
import requests
from shapely.geometry import box
from mpi4py import MPI
from collections import Counter
import numpy as np
from osgeo import gdal, ogr, osr

# project helpers
import s2sr_helpers as hf

# ---------------------------------------------------------------------
# GDAL / threading knobs (set once, per rank)
# ---------------------------------------------------------------------
os.environ.setdefault("GDAL_VRT_ENABLE_PYTHON", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".jp2,.xml")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_CACHEMAX", "512")  # MB per rank (tune)
os.environ.setdefault("GDAL_NUM_THREADS", "1")  # PixelFunction + MPI => 1
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Avoid oversubscription
os.environ.setdefault("GDAL_PAM_ENABLED", "NO") # prevent .aux.xml creation during reads

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="S2 L2A download + process pipeline (MPI).")
    aoi = p.add_mutually_exclusive_group(required=True)
    aoi.add_argument(
        "--bbox",
        nargs=4, 
        type=float, 
        metavar=("xmin","ymin","xmax","ymax"),
        help="Bounding box for your area of interest (e.g., xmin, ymin, xmax, ymax) in lat/lon"
    )
    p.add_argument(
        "--work_dir",
        required=True,
        type=Path,
        help="Path to working environment directory"
    )
    p.add_argument(
        "--grid_csv",
        required=True,
        type=Path,
        help="Path to study area grid created by 'make_grid.py' (e.g., 0.25-degree grid overlaying the Pan-Arctic)"
    )
    p.add_argument(
        "--start", 
        required=True,
        type=str,
        help="ISO start (e.g., 2019-06-01T00:00:00Z) for CDSE Sentinel-2 S2MSI2A product query"
    )
    p.add_argument(
        "--end", 
        required=True,
        type=str,
        help="ISO end (e.g., 2019-08-31T00:00:00Z) for CDSE Sentinel-2 S2MSI2A product query"
    )
    p.add_argument(
        "--bands",
        required=True,
        nargs="+",
        type=str,
        default=["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"],
        help="Band list (SCL is auto-included); e.g., --bands B02 B03 B04 B05 B06 B07 B8A B11 B12"
    )
    p.add_argument(
        "--bands_res",
        type=int,
        default=20,
        choices=[10,20,60],
        help="Band resolution in meters (10, 20, or 60)"
    )   
    p.add_argument(
        "--mask-classes",
        nargs="+",
        type=int,
        default=[0,1,2,3,7,8,9,10,11],
        choices=list(range(12)),
        help="SCL classes for masking. SCL classes are 0(no-data), 1(saturated), 2(topographic shadows), 3(cloud shadows), 4(vegetation), 5(non-vegetation), 6(water), 7(unclassified), 8(med cloud probability), 9(high cloud probability), 10(thin cirrus), 11(snow/ice)",
    )
    p.add_argument(
        "--tr", 
        nargs=2,
        type=float, 
        default=None, 
        metavar=("dLon","dLat"),
        help="Target WGS84 pixel size (deg). If omitted, let GDAL choose"
    )
    p.add_argument(
        "--tap", 
        action="store_true",
        help="Align pixels to target grid (use with --tr)"
    )
    p.add_argument(
        "--keep-tmp", 
        action="store_true",
        help="Keep intermediate tmp files"
    )
    return p.parse_args(argv)

# ---------------------------------------------------------------------
# MPI utility
# ---------------------------------------------------------------------
def mpi_scatter_roundrobin(items: List[Any], rank: int, size: int):
    """Return the slice of items for this rank (round-robin)."""
    return [items[i] for i in range(rank, len(items), size)]

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # #####################################################################
    # #####################################################################
    # I. Directories
    # #####################################################################
    # #####################################################################
    BASE_PATH: Path = args.work_dir
    OUT_DIR = BASE_PATH / "imagery_raw"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR = BASE_PATH / "indices"; INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR = BASE_PATH / "tmp"; TMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR = BASE_PATH / "logs"

    logger = hf.setup_rank_logger(LOG_DIR, rank, size, overwrite=True)

    # Per-rank scratch (keeps GDAL temp local & contention-free)
    rank_tmp = TMP_DIR / f"rank_{rank}"; rank_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CPL_TMPDIR", str(rank_tmp))

    # ---------------------------------------------------------------------
    # I.a. Reference files
    # ---------------------------------------------------------------------
    GRID_CSV: Path = args.grid_csv
    SCENE_INDEX_CSV = INDEX_DIR / "scene_index.csv"  # scene metadata index
    MANIFEST_CSV = INDEX_DIR / "query_manifest.csv"  # scenes returned by query
    SELECTED_CELLS_CSV = INDEX_DIR / "selected_gridcells.csv"

    # ---------------------------------------------------------------------
    # I.b. CDSE Parameters
    # ---------------------------------------------------------------------
    PRODUCT = "S2MSI2A"
    COLLECTION = "SENTINEL-2"
    CATALOGUE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    # ---------------------------------------------------------------------
    # I.c. Ensure creds are present (env or password file)
    # ---------------------------------------------------------------------
    have_creds = bool(os.getenv("CDSE_USERNAME") and (os.getenv("CDSE_PASSWORD") or os.getenv("CDSE_PASSWORD_FILE")))
    ok = comm.bcast(have_creds if rank == 0 else None, root=0)
    if not ok:
        if rank == 0:
            logger.critical("Missing creds. Set CDSE_USERNAME and CDSE_PASSWORD_FILE (or CDSE_PASSWORD).")
        sys.exit(5)

    # ---------------------------------------------------------------------
    # I.d. Build AOI bbox
    # ---------------------------------------------------------------------
    if args.bbox:
        xmin, ymin, xmax, ymax = tuple(args.bbox)
        aoi = box(xmin, ymin, xmax, ymax)
    else:
        raise SystemExit("[fatal] only --bbox AOI is supported in this script")

    # ---------------------------------------------------------------------
    # I.e. Gridcell selection from grid
    # ---------------------------------------------------------------------
    grid = hf.load_gridcells(GRID_CSV)
    selected = hf.select_intersecting_cells(aoi, grid)
    if selected.empty:
        if rank == 0:
            logger.critical("AOI intersects no gridcells.")
        sys.exit(2)

    if rank == 0:
        selected.to_csv(SELECTED_CELLS_CSV, index=False)
        logger.info(f"selected {len(selected)} gridcells -> {SELECTED_CELLS_CSV}")

    # ---------------------------------------------------------------------
    # I.f. Create bbox for query (union of selected gridcells)
    # ---------------------------------------------------------------------
    xmin, ymin, xmax, ymax = hf.union_bounds(selected)
    download_aoi = box(xmin, ymin, xmax, ymax)

    # ---------------------------------------------------------------------
    # I.g. [Rank 0] Query CDSE
    # ---------------------------------------------------------------------
    if rank == 0:
        try:
            search_q = hf.build_search_query(
                aoi=download_aoi,
                catalogue_odata=CATALOGUE_ODATA,
                collection_name=COLLECTION,
                product_type=PRODUCT,
                start_iso=args.start,
                end_iso=args.end,
            )
            result_df = hf.fetch_all_products(search_q)

            man = result_df[["Id","Name","ContentDate"]].copy()
            man["queried_utc"] = hf.utc_now_iso()
            man.to_csv(MANIFEST_CSV, index=False)

            product_names = (
                result_df["Name"].dropna().astype(str).str.rstrip("/")
                .drop_duplicates().sort_values().tolist()
            )
            logger.info(f"query returned {len(product_names)} products -> {MANIFEST_CSV}")
        except Exception as e:
            logger.critical(f"query failed: {e}")
            sys.exit(3)
    else:
        product_names = None
        result_df = None

    # ---------------------------------------------------------------------
    # I.h. Broadcast product names to all ranks
    # ---------------------------------------------------------------------
    product_names = comm.bcast(product_names if rank == 0 else None, root=0)

    # ---------------------------------------------------------------------
    # I.i. [Rank 0] Download
    # ---------------------------------------------------------------------
    target_res = str(args.bands_res)
    bands_res = ["20", "10"] if target_res == "20" else ["20", "60"] if target_res == "60" else [target_res]
    if rank == 0:
        backfilled = hf.backfill_index_from_existing_xmls(
            output_root=OUT_DIR, bands_res=bands_res, csv_path=SCENE_INDEX_CSV)
        if backfilled:
            logger.info(f"backfilled {backfilled} scenes into index.")

        dl_df = result_df if (result_df is not None and "Id" in result_df.columns) else None
        if dl_df is None:
            logger.warning("missing result_df Ids; cannot proceed with downloads.")
            sys.exit(4)

        logger.info(f"downloading {len(dl_df)} .SAFE products (missing only) to {OUT_DIR} with max 2 concurrent workers")
        failures = hf.download_rows_concurrent(
            dl_df, 
            OUT_DIR, 
            bands=set(args.bands) | {"SCL"},
            bands_res=bands_res,
            scene_csv=SCENE_INDEX_CSV,
            max_workers=2,
            logger=logger,
        )
        if failures:
            fail_csv = INDEX_DIR / "failed_downloads.csv"
            pd.DataFrame(failures).to_csv(fail_csv, index=False)
            logger.warning(f"{len(failures)} downloads failed → {fail_csv}")
        else:
            logger.info("download/check complete (all ok).")


    # ---------------------------------------------------------------------
    # I.j. Sync all ranks before continuing
    # ---------------------------------------------------------------------
    comm.Barrier()

    # #####################################################################
    # #####################################################################
    # II.a. Per-Scene Processing [mask -> offset -> warp to WGS84]
    # #####################################################################
    # #####################################################################

    # target published resolution to use from the SAFE folders
    req_bands = [b for b in args.bands if b.upper() != "SCL"]

    # ---------------------------------------------------------------------
    # II.b. Per-rank output directories
    # ---------------------------------------------------------------------
    masked_dir = rank_tmp / "masked_vrt"; masked_dir.mkdir(parents=True, exist_ok=True)
    wgs_dir = rank_tmp / "wgs84_vrt"; wgs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # II.c. Discover actual JP2s present (robust to partial/previous downloads)
    # ---------------------------------------------------------------------
    # discover JP2s present for the requested resolution
    jp2s = []
    missing = []
    for b in req_bands:
        res_pref = ["20","10"] if target_res == "20" else [target_res]
        hits = []
        for r in res_pref:
            hits = hf.find_band_jp2s_by_res(OUT_DIR, product_names, b, r)
            if hits:
                break
        if hits:
            jp2s.extend(hits)
        else:
            missing.append(b)

    if rank == 0:
        if missing:
            logger.warning(
                f"No {target_res} m files found for bands: {sorted(missing)} "
                f"(they may not be provided at that res in these products)"
            )
        by_band = Counter(p.name.split("_")[-2] for p in jp2s)  # e.g., 'B02', 'B8A'
        logger.info(
            f"Found {len(jp2s)} band-images across {len(product_names)} products; "
            f"unique bands at {target_res} m: {sorted(by_band.keys())}"
        )

    if not jp2s:
        if rank == 0:
            logger.warning("[phase1] no band JP2s found at requested resolution; nothing to process.")
        sys.exit(0)

    # ---------------------------------------------------------------------
    # II.d. Split work round-robin across ranks
    # ---------------------------------------------------------------------
    my_jobs = mpi_scatter_roundrobin(jp2s, rank, size)
    logger.info(f"[phase1] rank {rank} handling {len(my_jobs)} band-images")

    local_errors: List[Dict[str, str]] = []

    for band_jp2 in my_jobs:
        try:
            band_res_for_this = hf.res_from_band_filename(band_jp2.name) or target_res
            scl_jp2 = hf.corresponding_scl_for_band(band_jp2, band_res_for_this)
            if not scl_jp2.exists():
                logger.warning(f"[phase1][rank {rank}] missing SCL for {band_jp2}")
                continue

            # ---------------------------------------------------------------------
            # II.d.1. Get per-scene PB: offset -1000 DN when PB >= 4.00
            # ---------------------------------------------------------------------
            pb = hf.parse_pb_from_path(band_jp2)
            dn_off = 1000 if (pb is not None and pb >= 4.00) else 0

            # ---------------------------------------------------------------------
            # II.d.2. Output file names
            # ---------------------------------------------------------------------
            masked_vrt = masked_dir / band_jp2.name.replace(".jp2", "_masked_harmonized.vrt")
            wgs_vrt = wgs_dir / band_jp2.name.replace(".jp2", "_masked_harmonized_wgs84.vrt")

            # ---------------------------------------------------------------------
            # II.d.3. Atomic-create masked and offset VRT
            # ---------------------------------------------------------------------
            tmp_mask = masked_vrt.with_suffix(masked_vrt.suffix + ".tmp")
            hf.write_mask_and_offset_vrt(
                band_jp2,
                scl_jp2,
                tmp_mask,
                scl_classes=args.mask_classes,
                dn_offset=dn_off
            )
            tmp_mask.replace(masked_vrt)

            # ---------------------------------------------------------------------
            # II.d.4. Atomic-create warped VRT
            # ---------------------------------------------------------------------
            tmp_wgs = wgs_vrt.with_suffix(wgs_vrt.suffix + ".tmp")
            hf.warp_to_wgs84_vrt(
                masked_vrt, tmp_wgs,
                tr=tuple(args.tr) if args.tr else None,
                te=(xmin, ymin, xmax, ymax),
                tap=args.tap,
            )
            tmp_wgs.replace(wgs_vrt)

        except Exception as e:
            logger.warning(f"[phase1][rank {rank}] error for {band_jp2}: {e}")
            local_errors.append({"scene": str(band_jp2), "error": str(e)})

    # ---------------------------------------------------------------------
    # II.e. Gather & log errors
    # ---------------------------------------------------------------------
    all_errs = comm.gather(local_errors, root=0)
    if rank == 0:
        flat = [e for lst in all_errs for e in lst]
        if flat:
            err_csv = INDEX_DIR / "phase1_errors.csv"
            pd.DataFrame(flat).to_csv(err_csv, index=False)
            logger.warning(f"[phase1] {len(flat)} errors -> {err_csv}")
        else:
            logger.info("[phase1] all ranks completed with no errors.")
    comm.Barrier()
    print(f"[phase1] rank {rank} done.")

    # #####################################################################
    # #####################################################################
    # III.a Per-gridcell Processing [median composites]
    # #####################################################################
    # #####################################################################
    phase2_vrt_root = TMP_DIR / f"rank_{rank}" / "phase2_vrt"
    phase2_vrt_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        logger.info("[phase2] building per-gridcell composites...")

    # ---------------------------------------------------------------------
    # III.b. Collect all warped scene VRTs from every rank's tmp
    # ---------------------------------------------------------------------
    my_vrts = [str(p) for p in hf.list_wgs84_vrts(TMP_DIR / f"rank_{rank}")]
    all_lists = comm.allgather(my_vrts)
    all_vrts = [Path(p) for sub in all_lists for p in sub]
    if not all_vrts:
        if rank == 0:
            logger.warning("[phase2] no warped VRTs found; skipping.")
        sys.exit(0)

    # ---------------------------------------------------------------------
    # III.c. Create index of VRT Path to VRT bounds
    # ---------------------------------------------------------------------
    local_index = []
    # re-disitribute tasks from per-scene orientation to per-gridcell
    for p in all_vrts[rank::size]:
        try:
            local_index.append((str(p), hf.ds_bounds(p)))
        except Exception as e:
            pass
    parts = comm.allgather(local_index)
    scene_bounds = {Path(p): bbox for part in parts for (p, bbox) in part}

    # ---------------------------------------------------------------------
    # III.d. Choose gridcells by their ID
    # ---------------------------------------------------------------------
    cells_df = selected.copy()
    cells_df["geometry"] = [
        box(xmin, ymin, xmax, ymax)
        for xmin, ymin, xmax, ymax in cells_df[["xmin","ymin","xmax","ymax"]].to_numpy()
    ]
    if "tile_id_rc" not in cells_df.columns:
        cells_df["tile_id_rc"] = np.arange(len(cells_df), dtype=int)

    # ---------------------------------------------------------------------
    # III.e. Assign ranks their tasks
    # ---------------------------------------------------------------------
    # bands we will composite (per band)
    missing_set = set(missing)
    bands_for_phase2 = [b for b in req_bands if b not in missing_set]

    # make (cell_id, geom, band_code) tasks
    tasks = [(cid, g, b) for cid, g in cells_df[["tile_id_rc","geometry"]].itertuples(index=False, name=None)
                        for b in bands_for_phase2]

    # cap active ranks to avoid idle when tasks < size
    active = min(size, len(tasks)) if tasks else 0
    if active == 0:
        if rank == 0: logger.warning("[phase2] no tasks; skipping.")
        sys.exit(0)
    work_comm = comm.Split(color=0 if rank < active else MPI.UNDEFINED, key=rank)
    if work_comm == MPI.COMM_NULL:
        sys.exit(0)
    comm = work_comm; rank = comm.Get_rank(); size = comm.Get_size()

    # round-robin distribution of tasks to ranks
    my_tasks = tasks[rank::size]

    # ---------------------------------------------------------------------
    # III.f. Process per gridcell and per requested band
    # ---------------------------------------------------------------------
    comp_dir = BASE_PATH / "composites"; comp_dir.mkdir(parents=True, exist_ok=True)
    phase2_errors = []
    for cell_id, geom, band_code in my_tasks:

        # set up output directories and file-naming schema
        tmp_cell_dir = phase2_vrt_root / str(cell_id)
        tmp_cell_dir.mkdir(parents=True, exist_ok=True)
        stem = f"cell_{cell_id}_{band_code}"

        try:
            # ---------------------------------------------------------------------
            # III.f.1) pick scenes (candidates) that intersect the gridcell bbox
            # ---------------------------------------------------------------------
            cell_bbox = geom.bounds
            candidates = [p for p in all_vrts if hf.bbox_intersects(cell_bbox, scene_bounds[p])]
            scenes = [p for p in candidates if f"_{band_code}_" in p.name]
            if not scenes:
                continue
                
            # ---------------------------------------------------------------------
            # III.f.2) stack all scenes per band that intersect the gridcell bbox
            # ---------------------------------------------------------------------
            stack_vrt  = tmp_cell_dir / f"{stem}_stack.vrt"
            hf.build_stack_vrt(scenes, cell_bbox, stack_vrt)

            # ---------------------------------------------------------------------
            # III.f.3) get median composite of scenes
            # ---------------------------------------------------------------------
            median_vrt = tmp_cell_dir / f"{stem}_median.vrt"
            hf.build_median_vrt_from_stack(stack_vrt, median_vrt)

            # ---------------------------------------------------------------------
            # III.f.4) clip median composite gridcell to gridcell bounds
            # ---------------------------------------------------------------------
            out_dir = comp_dir / str(cell_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_tif = out_dir / f"{stem}_median_clip.tif"
            logging.info(f"[phase2][rank {rank}] cell {cell_id} band {band_code} -> {out_tif.name}")
            hf.warp_cutline_wkt_py(median_vrt, out_tif, cutline_wkt=geom.wkt)

        except Exception as e:
            phase2_errors.append({"cell": cell_id, "band": band_code, "error": str(e)})

    # ---------------------------------------------------------------------
    # III.f. Gather & log errors
    # ---------------------------------------------------------------------
    all_errs2 = comm.gather(phase2_errors, root=0)
    if rank == 0:
        flat2 = [e for lst in all_errs2 for e in lst]
        if flat2:
            pd.DataFrame(flat2).to_csv(LOG_DIR / "phase2_errors.csv", index=False)
            logger.warning(f"[phase2] {len(flat2)} errors -> {INDEX_DIR/'phase2_errors.csv'}")
        else:
            logger.info("[phase2] all ranks completed with no errors.")
    comm.Barrier()
    print(f"[phase2] rank {rank} done.")

if __name__ == "__main__":
    main()