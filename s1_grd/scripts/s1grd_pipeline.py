#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely import wkt as shapely_wkt
from mpi4py import MPI
from osgeo import gdal
import time

from s1grd_helpers import (
    ensure_dir, asf_search_aoi, predominant_flight_direction, download_asf_urls,
    warp_one_source_to_cell_vrt, write_median_uint16_from_vrts,
    transform_geom, gdal_info_json, load_grid_tasks_once, build_scene_index,
    setup_rank_logger, ensure_poeorb_via_s1_orbits, summarize_graph_nodes, run_snap_gpt,
    _link_shared_auxdata, snap_userdir, verify_aux_visibility, ensure_egm96_present
)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Sentinel-1 GRD pipeline: discover → download → SNAP → VRT mosaics → median → UInt16 dB.")
    # AOI / grid
    p.add_argument("--bbox", nargs=4, type=float, metavar=("xmin","ymin","xmax","ymax"), required=True, help="AOI bbox in EPSG:4326")
    p.add_argument("--grid_csv", type=Path, required=True, help="Pan-Arctic grid CSV (xmin,ymin,xmax,ymax[,tile_id_rc])")
    # Working dir
    p.add_argument("--work_dir", required=True, type=Path, help="Working directory")
    # ASF query
    p.add_argument("--start", required=True, type=str)
    p.add_argument("--end",   required=True, type=str)
    p.add_argument("--beam", type=str, default="IW")
    p.add_argument("--levels", nargs="+", default=["GRD_HD","GRD_FD","GRD_HS","GRD_MD","GRD_MS"])
    p.add_argument("--flightdir", type=str, default="predominant", choices=["predominant","ASCENDING","DESCENDING"])
    p.add_argument("--max-download-workers", type=int, default=6)
    p.add_argument("--earthdata-user", type=str, default=os.environ.get("EARTHDATA_USER"))
    p.add_argument("--earthdata-pass", type=str, default=os.environ.get("EARTHDATA_PASS"))
    # SNAP processing
    p.add_argument("--snap-xml", type=str, required=True)
    p.add_argument("--snap-props", type=str, required=True)
    p.add_argument("--gpt-bin", type=str, default="gpt")
    p.add_argument("--snap-outdir", type=Path, required=True,
                help="Target directory for SNAP outputs (like bash targetDirectory)")
    p.add_argument("--snap-prefix", type=str, default="Orb_NR_Cal_TC",
                help="Prefix for output names (like bash targetFilePrefix)")
    p.add_argument("--snap-format", type=str, default="GeoTIFF-BigTIFF",
                choices=["GeoTIFF","GeoTIFF-BigTIFF"],
                help="Output format for -f")
    # Mosaic grid / projection (GDAL)
    p.add_argument("--dst_srs", type=str, default="EPSG:4326")
    p.add_argument("--tr", nargs=2, type=float, metavar=("xres","yres"), required=True)
    p.add_argument("--tap", action="store_true")
    p.add_argument("--pols", nargs="+", default=["VV","VH"], choices=["VV","VH","HH","HV"])
    # dB packing
    p.add_argument("--db-min", type=float, default=-50)
    p.add_argument("--db-max", type=float, default=1)
    return p.parse_args()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # dirs
    BASE = Path(args.work_dir)
    RAW = BASE / "imagery_raw"
    TMP = BASE / "tmp"
    IDX = BASE / "indices"
    OUT = BASE / "composites"
    LOGS = BASE / "logs"
    AUXDATA_DIR = BASE / "auxdata"
    USERDIR = snap_userdir(BASE)
    OUT = args.snap_outdir
    for p in (RAW, TMP, IDX, OUT, LOGS, AUXDATA_DIR): 
        ensure_dir(p)

    logger = setup_rank_logger(LOGS, rank, size, overwrite=False, per_run_suffix=True)

    # -----------------------------------------------------------------
    # I. ASF query + download (rank 0 only)
    # -----------------------------------------------------------------
    if rank == 0:

        # search ASF given AOI, time, levels, beam, & flightdir from args
        aoi_poly = box(*args.bbox)
        logger.info("discovering scenes via ASF...")
        manifest = asf_search_aoi(aoi_poly.wkt, args.start, args.end, args.levels, args.beam,
                                  None if args.flightdir in ("both","predominant") else args.flightdir)
        if manifest.empty:
            logger.critical("no scenes returned by ASF for the given AOI/time.")
            sys.exit(2)

        # filter manifest by predominant flight direction if requested in args
        if args.flightdir == "predominant":
            fdir = predominant_flight_direction(manifest)
        else:
            fdir = args.flightdir
        manifest = manifest[manifest["flightDirection"] == fdir].copy()
        logger.info(f"using {fdir} flight direction: ({len(manifest)} scenes).")

        # write manifest to CSV in IDX directory
        manifest.to_csv(IDX / "s1_manifest.csv", index=False)
        logger.info(f"manifest written: {IDX/'s1_manifest.csv'}  ({len(manifest)} urls).")

        # download URLs from manifest to RAW directory
        logger.info("downloading ZIPs...")
        user = args.earthdata_user or os.environ.get("EARTHDATA_USER")
        pwd = args.earthdata_pass or os.environ.get("EARTHDATA_PASS")
        if not user or not pwd:
            logger.critical("EARTHDATA credentials missing.")
            sys.exit(2)
        msgs = download_asf_urls(
            manifest["url"].tolist(),
            out_dir=str(RAW),
            username=user,
            password=pwd,
            processes=args.max_download_workers,
        )
        logger.info("\n".join(msgs))
    comm.Barrier()

    # download the POEORB files we need via s1_orbits library (rank 0 only)
    if rank == 0:
        zips = sorted(str(p) for p in RAW.glob("S1*.zip"))
        if zips:
            logger.info(f"Ensuring POEORB orbit files exist in {AUXDATA_DIR} for {len(zips)} ZIPs...")
            orbit_msgs = ensure_poeorb_via_s1_orbits(
                zips,
                shared_root=AUXDATA_DIR,
                require_poeorb=True
            )
            logger.info("\n".join(orbit_msgs))
    if comm.Get_size() > 1:
        comm.Barrier()

    # -----------------------------------------------------------------
    # II. SNAP pre-processing per ZIP (parallel across ranks)
    # -----------------------------------------------------------------

    if rank != 0:
        if comm.Get_size() > 1:
            comm.Barrier()
        return

    # Link persistent auxdata into persistent userdir that SNAP will use
    ensure_egm96_present(AUXDATA_DIR, logger)
    _link_shared_auxdata(USERDIR, AUXDATA_DIR)
    verify_aux_visibility(USERDIR, logger)

    logger.info(f"SNAP processing via graph: {args.snap_xml}")
    logger.info(f"SNAP output format: {args.snap_format} (prefix: {args.snap_prefix})")
    logger.info(f"SNAP outputs to: {OUT}")
    logger.info(f"Temporary files in: {TMP}")
    logger.info(f"Using GPT binary: {args.gpt_bin}")
    stages = summarize_graph_nodes(args.snap_xml)
    logger.info("Graph stages: " + " -> ".join(stages))

    # List ZIPs to process as Path objects
    zpaths = sorted(RAW.glob("S1*.zip"))
    if not zpaths:
        logger.critical("No ZIPs found to process.")
        sys.exit(3)

    def make_target(zip_path: Path) -> Path:
        return OUT / f"{args.snap_prefix}_{zip_path.stem}.tif"

    for i, zp in enumerate(zpaths, 1):   # <--- use zpaths here
        tgt = make_target(zp)
        if tgt.exists() and tgt.stat().st_size > 0:
            logger.info(f"[{i}/{len(zpaths)}] SKIP existing {tgt.name}")
            continue

        tmp_tif = tgt.parent / (tgt.stem + ".tmp" + tgt.suffix)
        if tmp_tif.exists():
            try: os.remove(tmp_tif)
            except FileNotFoundError: pass

        logger.info(f"[{i}/{len(zpaths)}] processing -> {tmp_tif}")
        t0 = time.time()
        try:
            run_snap_gpt(
                zip_path=str(zp),
                gpt_bin=args.gpt_bin,
                graph_xml=args.snap_xml,
                prop_file=args.snap_props,
                out_path=str(tmp_tif),
                out_format=args.snap_format,
                user_dir=USERDIR,
                tmp_dir=TMP,
                shared_aux_root=AUXDATA_DIR,
                q_threads=16,
                cache_gb="16G",
            )
            os.replace(tmp_tif, tgt)
        except Exception as e:
            try:
                if tmp_tif.exists(): os.remove(tmp_tif)
            except Exception:
                pass
            logger.error(f"ERROR SNAP on {zp.name}: {e}")
            continue

        dt_min = (time.time() - t0) / 60.0
        logger.info(f"TIME PASSED: {dt_min:.1f} minutes — EXPORTED TO {tgt}")

    if comm.Get_size() > 1:
        comm.Barrier()

    # -----------------------------------------------------------------
    # III. Median composite per cell/pol → UInt16 dB (parallel across ranks)
    # -----------------------------------------------------------------
    # all_proc = sorted(str(p) for p in TMP.glob("*.tif"))
    # if rank == 0 and not all_proc:
    #     logger.critical("No processed per-scene products found; aborting.")
    #     sys.exit(3)
    # comm.Barrier()
    # if not all_proc: 
    #     sys.exit(3)

    # # rank 0: inspect first product for band names
    # if rank == 0:
    #     tif_info = gdal_info_json(all_proc[0])
    #     bands = [band.get("bandName", f"band{i+1}") for i, band in enumerate(tif_info.get("bands", []))]
    #     band_names = [band.upper() for band in bands]
    #     idx_vv = (band_names.index("SIGMA0_VV")+1) if "SIGMA0_VV" in band_names else (band_names.index("VV")+1 if "VV" in band_names else 1)
    #     idx_vh = (band_names.index("SIGMA0_VH")+1) if "SIGMA0_VH" in band_names else (band_names.index("VH")+1 if "VH" in band_names else 2)
    #     band_index_for = {"VV": idx_vv, "VH": idx_vh, "HH": 1, "HV": 2}
    # else:
    #     band_index_for = None
    # band_index_for = comm.bcast(band_index_for, root=0)

    # #  build scene index & load grid tasks (rank 0 only), then broadcast
    # if rank == 0:
    #     proc_wkt, idx_gdf = build_scene_index(all_proc)
    #     tasks = load_grid_tasks_once(args.grid_csv, tuple(args.bbox))
    # else:
    #     proc_wkt, idx_gdf, tasks = None, None, None
    # proc_wkt = comm.bcast(proc_wkt, root=0)

    # # broadcast idx_gdf as records (avoid geopandas pickling overhead)
    # if rank == 0:
    #     idx_rows = list(idx_gdf[["path","xmin","ymin","xmax","ymax"]].itertuples(index=False, name=None))
    # else:
    #     idx_rows = None
    # idx_rows = comm.bcast(idx_rows, root=0)

    # # rebuild simple GeoDataFrame locally
    # idx_gdf = gpd.GeoDataFrame(
    #     [{"path": p, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax} for (p,xmin,ymin,xmax,ymax) in idx_rows],
    #     geometry=[box(xmin, ymin, xmax, ymax) for (_, xmin, ymin, xmax, ymax) in idx_rows],
    #     crs=proc_wkt
    # )
    # tasks = comm.bcast(tasks, root=0)

    # # split cells across ranks
    # my_tasks = tasks[rank::size]
    # logger.info(f"Cells assigned: {len(my_tasks)}")

    # dst_srs = args.dst_srs
    # xres, yres = float(args.tr[0]), float(args.tr[1])
    # db_min, db_max = float(args.db_min), float(args.db_max)

    # for tile_id, wkt_ll in my_tasks:
    #     geom_ll  = shapely_wkt.loads(wkt_ll)                  # EPSG:4326
    #     geom_dst = transform_geom(geom_ll, "EPSG:4326", dst_srs)
    #     geom_prc = transform_geom(geom_ll, "EPSG:4326", proc_wkt)

    #     hits = idx_gdf[idx_gdf.geometry.intersects(geom_prc)]
    #     if hits.empty:
    #         logger.info(f"Cell {tile_id}: no intersecting scenes"); continue

    #     for pol in args.pols:
    #         out_tif = OUT / f"GRIDCELL_{tile_id}_{pol}_u16.tif"
    #         if out_tif.exists() and out_tif.stat().st_size > 0:
    #             logger.info("SKIP", out_tif.name); continue
    #         band_index = band_index_for[pol]

    #         # build per-scene warped VRTs (in memory) for just the hits
    #         vrts = []
    #         try:
    #             for src in hits["path"].tolist():
    #                 ds = gdal.Open(src)
    #                 if ds is None or band_index > ds.RasterCount:
    #                     ds = None; continue
    #                 ds = None
    #                 v = warp_one_source_to_cell_vrt(src, band_index, geom_dst, dst_srs, xres, yres, tap=bool(args.tap))
    #                 vrts.append(v)
    #             if not vrts:
    #                 logger.info(f"Cell {tile_id} {pol}: no sources after warp")
    #                 continue

    #             write_median_uint16_from_vrts(vrts, out_tif, db_min=db_min, db_max=db_max)
    #             logger.info("WROTE", out_tif.name)
    #         except Exception as e:
    #             logger.error(f"ERROR cell {tile_id} {pol}: {e}")
    #             for v in vrts:
    #                 try: gdal.Unlink(v)
    #                 except Exception: pass

    # comm.Barrier()
    # if rank == 0:
    #     logger.info("Median composites complete.")

if __name__ == "__main__":
    main()
