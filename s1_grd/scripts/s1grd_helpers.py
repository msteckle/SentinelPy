#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helpers for Sentinel-1 GRD discovery, download, SNAP processing, and per-cell mosaics.
Dependencies:
  - asf_search, requests, numpy, pandas, geopandas, shapely, pyproj, GDAL
"""

from __future__ import annotations
import os, json, time, subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Sequence, Union

import shutil
import shlex
import s1_orbits
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping, Polygon
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import logging
from datetime import datetime
from lxml import etree

import asf_search as asf

from osgeo import gdal
gdal.UseExceptions()

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------

def ensure_dir(p: os.PathLike) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def gdal_info_json(path: str) -> dict:
    return json.loads(gdal.Info(path, format="json"))

def transform_geom(geom: Polygon, src_srs: str, dst_srs: str) -> Polygon:
    tr = Transformer.from_crs(src_srs, dst_srs, always_xy=True).transform
    return shp_transform(tr, geom)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_rank_logger(
    log_dir: Path,
    rank: int,
    size: int,
    name: str = "s1_pipeline",
    level: int | None = None,
    overwrite: bool = False,  # True: 'w' (fresh file); False: 'a' (append)
    per_run_suffix: bool = True,  # True: add datetime to filename for this run
) -> logging.Logger:
    """
    Per-rank logger that writes to logs/<name>_rank{rank}[_{YYYYmmdd-HHMMSS}].log
    and also mirrors to console. No rotation — delete whenever you want.
    """
    rank = rank + 1
    log_dir.mkdir(parents=True, exist_ok=True)

    if level is None:
        level_name = os.environ.get("S1_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    # filename (optionally include run timestamp)
    suffix = f"_{datetime.now().strftime('%Y%m%d')}" if per_run_suffix else ""
    logfile = log_dir / f"{name}_rank{rank}{suffix}.log"

    logger_name = f"{name}.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicates if re-initialized

    fmt = logging.Formatter(
        fmt="%(asctime)s [rank %(rank)s/%(size)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    class _RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            record.size = size
            return True

    # File handler
    fh = logging.FileHandler(logfile, mode=("w" if overwrite else "a"), encoding="utf-8")
    fh.setFormatter(fmt); fh.addFilter(_RankFilter()); fh.setLevel(level)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.addFilter(_RankFilter()); ch.setLevel(level)
    logger.addHandler(ch)

    # Tame noisy libs if desired
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # make the path easy to see at start
    logger.info("Logging to %s", logfile)
    return logger


# ---------------------------------------------------------------------
# ASF search & download
# ---------------------------------------------------------------------

def asf_search_aoi(
    aoi_wkt: str,
    date_start: str,
    date_end: str,
    product_levels: Iterable[str],
    beam_mode: str = "IW",
    flight_direction: Optional[str] = None,
) -> pd.DataFrame:
    """Return manifest pandas dataframe with unique URLs."""

    asf.constants.INTERNAL.CMR_TIMEOUT = 60
    lvls = [getattr(asf.PRODUCT_TYPE, p) for p in product_levels]
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=lvls,
        start=pd.to_datetime(date_start).date(),
        end=pd.to_datetime(date_end).date(),
        beamMode=beam_mode,
        intersectsWith=aoi_wkt,
        flightDirection=flight_direction if flight_direction else None,
    )
    geoj = results.geojson()
    rows = []
    for feat in geoj["features"]:
        prop = feat["properties"]
        rows.append({
            "granule": prop.get("fileName"),
            "url": prop.get("url"),
            "beamMode": prop.get("beamMode"),
            "flightDirection": prop.get("flightDirection"),
            "startTime": prop.get("startTime"),
            "stopTime": prop.get("stopTime"),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).sort_values("granule")
    return df


def predominant_flight_direction(manifest: pd.DataFrame) -> Optional[str]:
    vals = manifest["flightDirection"].dropna().values
    if len(vals) == 0:
        return None
    u, c = np.unique(vals, return_counts=True)
    return str(u[int(np.argmax(c))])


def download_asf_urls(urls, out_dir, username=None, password=None, token=None, processes=4):
    """
    Use ASF's official helpers. Auth via .netrc (preferred), creds, or token.
    Logs start/end timestamps and duration. Returns list of summary strings.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    session = None
    if token:
        session = asf.ASFSession().auth_with_token(token)
    elif username and password:
        session = asf.ASFSession().auth_with_creds(username, password)

    start = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))

    asf.download_urls(urls=urls, path=str(out_dir), session=session, processes=processes)

    end = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    elapsed = end - start

    msgs = [
        f"Download started: {start_str}",
        f"Download ended:   {end_str}",
        f"Elapsed time:     {elapsed:.1f} seconds for {len(urls)} files",
    ]
    return msgs


# ---------------------------------------------------------------------
# SNAP (gpt): get orbit files via s1_orbits
# ---------------------------------------------------------------------


_ORB_RE = re.compile(
    r"(?P<sat>S1[AB])_OPER_AUX_(?P<otype>POEORB|RESORB)_.*?_V(?P<vstart>\d{8}T\d{6})_(?P<vstop>\d{8}T\d{6})\.EOF$"
)

def dest_path_from_orbit(user_dir: Path, orbit_name: str) -> Path:
    """
    Compute SNAP auxdata destination for a *POEORB* orbit filename.
    user_dir should be the SNAP user dir (i.e., SNAP_USER_DIR), not auxdata root.
    """
    matches = _ORB_RE.match(orbit_name)
    if not matches:
        return None

    sat = matches["sat"]  # S1A or S1B
    vstart = matches["vstart"]  # e.g., 20190601T235944
    year = int(vstart[:4])
    month = int(vstart[4:6])

    # SNAP stores under: $SNAP_USER_DIR/auxdata/Orbits/Sentinel-1/POEORB/S1A/YYYY/MM/...
    return user_dir / "auxdata" / "Orbits" / "Sentinel-1" / "POEORB" / sat / f"{year:04d}" / f"{month:02d}" / orbit_name


def ensure_poeorb_via_s1_orbits(
    zip_paths: Sequence[Union[str, Path]],
    user_dir: Path,
    temp_dir: Path,
) -> List[str]:
    """
    For each SCENE ZIP, download the corresponding orbit via s1_orbits.fetch_for_scene(scene_id, dir=...),
    and place it under $SNAP_USER_DIR/auxdata/Orbits/Sentinel-1/POEORB/<SAT>/<YYYY>/<MM>/<file>.EOF
    Skips non-POEORB (i.e., RESORB) files.
    """

    # ensure s1_orbits directory is configured
    msgs: List[str] = []
    temp_orbit_dir = temp_dir / "orbits"
    ensure_dir(temp_orbit_dir)

    # loop over zips
    for zip_path in zip_paths:
        zip_path = Path(zip_path)
        scene_id = zip_path.stem

        # fetch orbit file to temp dir
        try:
            download_path = Path(s1_orbits.fetch_for_scene(scene_id, dir=temp_orbit_dir))
        except Exception as e:
            msgs.append(f"ERR  {zip_path.name}: fetch failed ({e})")
            continue

        orbit_name = download_path.name

        # only keep POEORB files
        if "POEORB" not in orbit_name:
            msgs.append(f"SKIP {zip_path.name}: not a POEORB file ({orbit_name})")
            try:
                download_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # compute dest path
        try:
            dest_path = dest_path_from_orbit(user_dir, orbit_name)
        except Exception as e:
            msgs.append(f"ERR  {zip_path.name}: {e}")
            try:
                download_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # if dest path could not be computed
        if dest_path is None:
            msgs.append(f"ERR  {zip_path.name}: could not parse orbit filename ({orbit_name})")
            try:
                download_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # skip if already present
        if dest_path.exists() and dest_path.stat().st_size > 0:
            msgs.append(f"SKIP {zip_path.name}: already present at {dest_path}")
            try:
                download_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # move orbit file into final location and remove temp file
        ensure_dir(dest_path.parent)
        try:
            os.replace(download_path, dest_path)
            try:
                os.chmod(dest_path, 0o644)
            except Exception:
                pass
        except Exception as e:
            msgs.append(f"ERR  {zip_path.name}: move to {dest_path} failed ({e})")
            try:
                download_path.unlink(missing_ok=True)
            except Exception:
                pass

    return msgs


# ---------------------------------------------------------------------
# SNAP (gpt): get EGM96 via atomic download
# ---------------------------------------------------------------------
import urllib.request, tempfile

EGM96_URL = "http://step.esa.int/auxdata/dem/egm96/ww15mgh_b.zip"

def ensure_egm96_present(user_dir: Path, logger: logging.Logger) -> Path:
    """
    Ensure <user_dir>/auxdata/dem/egm96/ww15mgh_b.zip exists and is non-empty.
    If missing/empty, download atomically. Returns the path.
    """
    # ensure target dir exists
    egm_dir = user_dir / "auxdata" / "dem" / "egm96"
    ensure_dir(egm_dir)
    egm_zip = egm_dir / "ww15mgh_b.zip"

    # already present?
    if egm_zip.exists() and egm_zip.stat().st_size > 0:
        logger.info(f"EGM96 present: {egm_zip} ({egm_zip.stat().st_size} bytes)")
        return egm_zip

    # download to a temp file in the same directory, then move into place
    logger.info(f"EGM96 not found; downloading to {egm_zip} ...")
    with tempfile.NamedTemporaryFile(dir=str(egm_dir), delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        with urllib.request.urlopen(EGM96_URL, timeout=120) as r, open(tmp_path, "wb") as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        size = tmp_path.stat().st_size
        if size == 0:
            raise RuntimeError("Downloaded EGM96 file is size 0")
        
        # move into place
        os.replace(tmp_path, egm_zip)
        try: 
            os.chmod(egm_zip, 0o644)
        except Exception: 
            pass
        logger.info(f"EGM96 downloaded: {egm_zip} ({size} bytes)")
        return egm_zip
    except Exception as e:
        try: 
            tmp_path.unlink(missing_ok=True)
        except Exception: 
            pass
        raise


# ---------------------------------------------------------------------
# SNAP (gpt): core processing
# ---------------------------------------------------------------------

def remove_beam_dimap(out_dim: Path, logger=None):
    """
    Delete a BEAM-DIMAP product: the .data dir and the .dim file.
    """
    out_dim = Path(out_dim)
    if out_dim.suffix.lower() != ".dim":
        raise ValueError(f"Expected a .dim path, got: {out_dim}")

    data_dir = out_dim.with_suffix(".data")  # foo.dim -> foo.data

    # remove .data directory first
    if data_dir.exists():
        if data_dir.is_dir():
            if logger: logger.debug(f"Removing directory: {data_dir}")
            shutil.rmtree(data_dir)
        else:
            # very rare, but just in case
            if logger: logger.debug(f"Removing file: {data_dir}")
            data_dir.unlink()

    # then remove the .dim
    if out_dim.exists():
        if logger: logger.debug(f"Removing file: {out_dim}")
        out_dim.unlink()

def run_snap_gpt(
    zip_path: str,
    gpt_bin: str,
    graph_xml: str,
    prop_file: str,
    out_path: str,
    user_dir: str,
) -> None:

    # build command
    cmd = [
        gpt_bin, graph_xml,
        "-e",
        # system properties
        f"-Dsnap.userdir={user_dir}",
        # graph properties
        "-p", prop_file,
        "-t", str(out_path),
        zip_path,
    ]

    # execute command
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"gpt failed (rc={rc}) for {zip_path}\nCMD: {' '.join(cmd)}")


def summarize_graph_nodes(graph_xml: str) -> list[str]:
    import xml.etree.ElementTree as ET
    tree = ET.parse(graph_xml)
    root = tree.getroot()
    pres = root.find(".//applicationData[@id='Presentation']")
    order = []
    if pres is not None:
        pts = []
        for n in pres.findall("./node"):
            nid = n.get("id")
            pos = n.find("./displayPosition")
            if nid and pos is not None:
                try:
                    x = float(pos.get("x", "0"))
                except Exception:
                    x = 0.0
                pts.append((x, nid))
        order = [nid for _, nid in sorted(pts)]
    if not order:
        order = [n.get("id") for n in root.findall("./node") if n.get("id")]
    graph_nodes = {n.get("id") for n in root.findall("./node") if n.get("id")}
    return [nid for nid in order if nid in graph_nodes]


# ---------------------------------------------------------------------
# Scene index (footprints)
# ---------------------------------------------------------------------

def build_scene_index(all_proc: List[str]) -> Tuple[str, gpd.GeoDataFrame]:
    """Return (proj_wkt, GeoDataFrame with 'path' geometry=bbox) for processed scenes."""
    if not all_proc:
        raise RuntimeError("No processed products to index.")
    info0 = gdal_info_json(all_proc[0])
    proj_wkt = info0.get("coordinateSystem", {}).get("wkt", "")
    if not proj_wkt:
        ds = gdal.Open(all_proc[0]); proj_wkt = ds.GetProjection(); ds = None
    rows = []
    for pth in all_proc:
        j = gdal_info_json(pth)
        cc = j.get("cornerCoordinates", {})
        xs = [cc["lowerLeft"][0], cc["lowerRight"][0], cc["upperLeft"][0], cc["upperRight"][0]]
        ys = [cc["lowerLeft"][1], cc["lowerRight"][1], cc["upperLeft"][1], cc["upperRight"][1]]
        rows.append({"path": pth, "xmin": min(xs), "ymin": min(ys), "xmax": max(xs), "ymax": max(ys)})
    gdf = gpd.GeoDataFrame(rows,
        geometry=[box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in rows],
        crs=proj_wkt
    )
    return proj_wkt, gdf

# ---------------------------------------------------------------------
# Per-cell VRT warp + median → UInt16 dB
# ---------------------------------------------------------------------

# def add_pixelfunc_to_vrt(vrt_path: Path, func_name: str) -> None:
#     """
#     Add a PixelFunctionType to a VRT file. The function should be defined in GDAL.
#     Modifies the VRT file in place.
#     """
#     # read the VRT XML
#     tree = etree.parse(str(vrt_path))
#     root = tree.getroot()
#     band1 = root.findall(".//VRTRasterBand[@band='1']")[0]

#     # add PixelFunctionType element
#     band1.set("subClass","VRTDerivedRasterBand")
#     pixelFunctionType = etree.SubElement(band1, 'PixelFunctionType')
#     pixelFunctionType.text = func_name
#     pixelFunctionLanguage = etree.SubElement(band1, 'PixelFunctionLanguage')
#     pixelFunctionLanguage.text = "Python"
#     pixelFunctionCode = etree.SubElement(band1, 'PixelFunctionCode')
#     # function that finds median of dB data
#     pixelFunctionCode.text = etree.CDATA("""
# import numpy as np
# def {func_name}(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
#     data = in_ar[0]
#     data = np.where(data > 0, data, np.nan)
#     # compute median along the stack axis (0)
#     median = np.nanmedian(data, axis=0)
#     # convert back to UInt16 dB
#     median_db = np.where(np.isnan(median), 0, np.clip(np.round(10 * np.log10(median)), 0, 65535)).astype(np.uint16)
#     out_ar[:] = median_db
# """.format(func_name=func_name))

#     # write back the modified VRT
#     tree.write(str(vrt_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")

