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
    setup_rank_logger, ensure_poeorb_via_s1_orbits, summarize_graph_nodes, run_snap_gpt,
    ensure_egm96_present, remove_beam_dimap
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
    p.add_argument("--snap-userdir", type=Path, default=os.environ.get("SNAP_USER_DIR"), help="Persistent SNAP userdir (where auxdata is stored)")
    p.add_argument("--snap-outdir", type=Path, required=True, help="Target directory for SNAP outputs (like bash targetDirectory)")
    p.add_argument("--snap-prefix", type=str, default="Orb_NR_Cal_TC", help="Prefix for output names (like bash targetFilePrefix)")
    p.add_argument("--snap-overwrite", action="store_true", help="Whether to overwrite existing SNAP outputs")
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

    # MPI setup (but we won't use it until Stage III)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    is_rank0 = (rank == 0)

    # define directories
    BASE = Path(args.work_dir)
    RAW = BASE / "imagery_raw"
    TMP = BASE / "tmp"
    IDX = BASE / "indices"
    LOGS = BASE / "logs"
    SNAP_OUT = args.snap_outdir
    USERDIR = args.snap_userdir

    # ensure the directories exist
    for path in (RAW, TMP, IDX, SNAP_OUT, LOGS, USERDIR): 
        ensure_dir(path)

    # set up logging
    logger = setup_rank_logger(LOGS, rank, size, overwrite=True)

    # -----------------------------------------------------------------
    # I. ASF query + download (rank 0 only)
    # -----------------------------------------------------------------
    if is_rank0:

        # search ASF given AOI, time, levels, beam, & flightdir from args
        aoi_poly = box(*args.bbox)
        logger.info("discovering scenes via ASF...")
        manifest = asf_search_aoi(
            aoi_wkt=aoi_poly.wkt, 
            date_start=args.start, date_end=args.end,
            product_levels=args.levels, beam_mode=args.beam,
            flight_direction=None if args.flightdir in ("predominant") else args.flightdir
        )
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

    # -----------------------------------------------------------------
    # II. SNAP pre-processing per ZIP (parallel across ranks)
    # -----------------------------------------------------------------

    if is_rank0:

        # ensure POEORB files exist in USERDIR auxdata (rank 0 only)
        zips = sorted(str(p) for p in RAW.glob("S1*.zip"))
        if zips:
            logger.info(f"Ensuring POEORB orbit files exist in {USERDIR} for {len(zips)} ZIPs...")
            orbit_msgs = ensure_poeorb_via_s1_orbits(zip_paths=zips, user_dir=USERDIR, temp_dir=TMP)
            logger.info("\n".join(orbit_msgs))

        # ensure egm96 is present in userdir auxdata
        ensure_egm96_present(user_dir=USERDIR, logger=logger)

        # log info
        logger.info(f"SNAP processing via graph: {args.snap_xml}")
        logger.info(f"SNAP outputs to: {SNAP_OUT}")
        logger.info(f"Temporary files in: {TMP}")
        logger.info(f"Using GPT binary: {args.gpt_bin}")
        stages = summarize_graph_nodes(args.snap_xml)
        logger.info("Graph stages: " + " -> ".join(stages))

        # list ZIPs to process as Path objects
        zip_paths = sorted(RAW.glob("S1*.zip"))
        if not zip_paths:
            logger.critical("No ZIPs found to process.")
            sys.exit(3)

        # loop over ZIPs, skipping existing outputs
        for i, zip_path in enumerate(zip_paths, 1):

            out_file = SNAP_OUT / f"{args.snap_prefix}_{zip_path.stem}.dim"

            # skip existing output unless overwrite requested
            if not args.snap_overwrite:
                if out_file.exists() and out_file.stat().st_size > 0:
                    logger.info(f"[{i}/{len(zip_paths)}] SKIPPING existing -> {out_file}")
                    continue

            # overwrite case: nuke existing BEAM-DIMAP first
            if out_file.exists():
                logger.warning(f"[{i}/{len(zip_paths)}] OVERWRITING existing -> {out_file}")
                try:
                    remove_beam_dimap(out_file, logger=logger)
                except Exception as e:
                    logger.error(f"Failed to remove existing product {out_file}: {e}")
                    continue

            logger.info(f"[{i}/{len(zip_paths)}] PROCESSING -> {out_file}")
            t0 = time.time()
            try:
                os.environ["SNAP_USER_DIR"] = str(USERDIR)
                os.environ["GDAL_VRT_ENABLE_PYTHON"] = "YES"
                run_snap_gpt(
                    zip_path=str(zip_path),
                    gpt_bin=args.gpt_bin,
                    graph_xml=args.snap_xml,
                    prop_file=args.snap_props,
                    out_path=str(out_file),
                    user_dir=str(USERDIR),
                )
            except Exception as e:
                logger.error(f"ERROR SNAP on {zip_path.name}: {e}")
                continue

            dt_min = (time.time() - t0) / 60.0
            logger.info(f"TIME PASSED: {dt_min:.1f} minutes — EXPORTED TO {out_file}")

        # MASTER gathers the list of processed products to hand to all ranks
        processed = sorted(str(p) for p in SNAP_OUT.glob("**/*.img"))
        if not processed:
            logger.critical("No processed products found; aborting composites.")
            # Broadcast an empty list so others can exit cleanly
        else:
            logger.info(f"Processed products ready for composites: {len(processed)}")

    # all other ranks set processed to None
    else:
        processed = None

    # broadcast the processed list to all ranks
    processed = comm.bcast(processed, root=0)
    if not processed:
        if is_rank0:
            logger.critical("Nothing to composite.")
        return

    comm.Barrier()  # all ranks start composites together

    # -----------------------------------------------------------------
    # III. Extract scenes intersecting each grid cell
    # -----------------------------------------------------------------

    # all_proc = processed # all processed products
    # if is_rank0:

    #     # subselect gridcells from the panarctic grid that intersect the AOI
    #     logger.info("loading grid CSV...")
    #     grid = pd.read_csv(args.grid_csv, dtype={"tile_id_rc": str} if "tile_id_rc" in pd.read_csv(args.grid_csv, nrows=0).columns else None)
    #     if not {"xmin","ymin","xmax","ymax"}.issubset(grid.columns):
    #         logger.critical("grid CSV must contain xmin,ymin,xmax,ymax columns.")
    #         sys.exit(4)
    #     grid["geometry"] = grid.apply(lambda r: box(r.xmin, r.ymin, r.xmax, r.ymax), axis=1)
    #     grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326")
    #     aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly], crs="EPSG:4326")
    #     grid = gpd.overlay(grid, aoi_gdf, how="intersection")
    #     if grid.empty:
    #         logger.critical("no grid cells intersect the AOI.")
    #         sys.exit(5)
    #     logger.info(f"{len(grid)} grid cells intersect the AOI.")

    #     # build an index of processed products with their footprints
    #     logger.info("building scene index...")
    #     scenes = []
    #     for p in all_proc:

    #         # read gdal.Info JSON
    #         info = gdal.Info(p, format="json")
    #         if "cornerCoordinates" not in info:
    #             logger.warning(f"no cornerCoordinates in {p}; skipping")
    #             continue
            
    #         # build footprint polygon from cornerCoordinates
    #         cc = info["cornerCoordinates"]
    #         poly_wkt = shapely_wkt.dumps(box(cc["lowerLeft"][0], cc["lowerLeft"][1], cc["upperRight"][0], cc["upperRight"][1]))
    #         scenes.append({
    #             "path": p,
    #             "xmin": cc["lowerLeft"][0],
    #             "ymin": cc["lowerLeft"][1],
    #             "xmax": cc["upperRight"][0],
    #             "ymax": cc["upperRight"][1],
    #             "geometry": poly_wkt
    #         })

    #     # create a GeoDataFrame of scenes
    #     scenes = pd.DataFrame(scenes)
    #     if scenes.empty:
    #         logger.critical("no valid scenes found after building index.")
    #         sys.exit(6)
    #     scenes["geometry"] = scenes["geometry"].apply(shapely_wkt.loads)
    #     scenes_gdf = gpd.GeoDataFrame(scenes, geometry="geometry", crs="EPSG:4326")
    #     logger.info(f"{len(scenes_gdf)} scenes in the index.")

    #     # Prepare lightweight payloads to broadcast to all ranks
    #     id_name = "tile_id_rc"
    #     grid_rows = [(float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax), r[id_name])
    #                 for _, r in grid.iterrows()]
    #     scene_rows = [(row.path, float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
    #                 for row in scenes_gdf[["path","xmin","ymin","xmax","ymax"]].itertuples(index=False)]
    #     payload = {"grid_rows": grid_rows, "id_name": id_name, "scene_rows": scene_rows}

    # else:
    #     payload = None

    # comm.Barrier()  # ensure rank 0 is done before others proceed

    # # -----------------------------------------------------------------
    # # IV. Build VRT stacks per cell + polarization
    # # -----------------------------------------------------------------

    # payload = comm.bcast(payload, root=0)
    # grid_rows  = payload["grid_rows"]
    # id_name = payload["id_name"]
    # scene_rows = payload["scene_rows"]

    # # Rebuild GeoDataFrames locally on each rank
    # grid = gpd.GeoDataFrame(
    #     [{id_name: cid, "geometry": box(xmin, ymin, xmax, ymax)}
    #     for (xmin, ymin, xmax, ymax, cid) in grid_rows],
    #     crs="EPSG:4326",
    # )

    # scenes_gdf = gpd.GeoDataFrame(
    #     [{"path": p, "geometry": box(xmin, ymin, xmax, ymax)}
    #     for (p, xmin, ymin, xmax, ymax) in scene_rows],
    #     crs="EPSG:4326",
    # )

    # # assign each rank some grid cells (round-robin)
    # my_cells = [row for i, row in grid.iterrows() if (i % size) == rank]
    # logger.info(f"{len(my_cells)} grid cells assigned to rank {rank}.")

    # # for each assigned cell, make a list of scenes that intersect the cell
    # for i, cell in enumerate(my_cells, 1):
    #     cell_id = cell["tile_id_rc"]
    #     logger.info(f"[{i}/{len(my_cells)}] rank {rank} processing cell {cell_id}...")

    #     # find scenes that intersect this cell
    #     cell_geom = gpd.GeoDataFrame(geometry=[cell.geometry], crs="EPSG:4326")
    #     intersecting = gpd.overlay(scenes_gdf, cell_geom, how="intersection")
    #     if intersecting.empty:
    #         logger.warning(f"no scenes intersect cell {cell_id}; skipping.")
    #         continue

    #     # for each polarization, build a simple VRT (stack of scenes intersecting thiscell)
    #     for pol in args.pols:
    #         pol = pol.upper()
    #         pol_scenes = [p for p in intersecting["path"].tolist() if f"_{pol}." in os.path.basename(p)]
    #         logger.info(f"{len(pol_scenes)} scenes intersect cell {cell_id}.")

    #         # build the VRT
    #         vrt_dir = TMP / "vrts"
    #         ensure_dir(vrt_dir)
    #         vrt_path = vrt_dir / f"{cell_id}_{pol}.vrt"
    #         try:
    #             gdal.BuildVRT(str(vrt_path), pol_scenes)
    #         except Exception as e:
    #             logger.error(f"ERROR building VRT for cell {cell_id} pol {pol}: {e}")
    #             continue

    #         # add PixelFunction to VRT XML for median calculation
    #         try:
    #             add_pixelfunc_to_vrt(vrt_path, "median")
    #         except Exception as e:
    #             logger.error(f"ERROR adding PixelFunction to VRT for cell {cell_id} pol {pol}: {e}")
    #             continue

    # comm.Barrier()

if __name__ == "__main__":
    main()
