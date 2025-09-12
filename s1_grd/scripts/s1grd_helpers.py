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
from typing import Iterable, List, Optional, Tuple, Dict

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
    log_dir.mkdir(parents=True, exist_ok=True)

    if level is None:
        level_name = os.environ.get("S1_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    # filename (optionally include run timestamp)
    suffix = f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}" if per_run_suffix else ""
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
# Grid
# ---------------------------------------------------------------------

def load_grid_tasks_once(grid_csv: Path, aoi_bbox: Tuple[float, float, float, float]):
    """Read grid once, intersect with AOI (EPSG:4326), return [(tile_id, WKT), ...]."""
    df = pd.read_csv(grid_csv)
    need = {"xmin","ymin","xmax","ymax"}
    if not need.issubset(df.columns):
        raise ValueError(f"{grid_csv} must contain columns {need}")
    df["tile_id"] = df["tile_id_rc"] if "tile_id_rc" in df.columns else df.index.astype(str)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in df[["xmin","ymin","xmax","ymax"]].to_numpy()],
        crs="EPSG:4326",
    )
    aoi_poly = box(*aoi_bbox)
    cand = list(gdf.sindex.intersection(aoi_poly.bounds))
    sel = gdf.iloc[cand]
    sel = sel[sel.intersects(aoi_poly)].copy()
    if sel.empty:
        raise RuntimeError("No grid cells intersect AOI.")
    sel["wkt"] = sel.geometry.to_wkt()
    return list(sel[["tile_id","wkt"]].itertuples(index=False, name=None))

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

    asf.CMR_TIMEOUT = 60
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
# SNAP (gpt): basic funcs
# ---------------------------------------------------------------------

_ORB_RE = re.compile(
    r"(?P<sat>S1[AB])_OPER_AUX_(?P<otype>POEORB|RESORB)_.*?_V(?P<vstart>\d{8}T\d{6})_(?P<vstop>\d{8}T\d{6})\.EOF$"
)

def _scene_id_from_zip(zip_path: str) -> str:
    # e.g., "/path/.../S1A_IW_GRDH_..._2A1F.zip" -> "S1A_IW_GRDH_..._2A1F"
    return Path(zip_path).stem


# ---------------------------------------------------------------------
# SNAP (gpt): get orbit files via s1_orbits
# ---------------------------------------------------------------------


def _dest_path_for_orbit(shared_root: Path, orbit_name: str) -> Path:
    m = _ORB_RE.match(orbit_name)
    if not m:
        # fail and exit rather than guessing
        raise ValueError(f"Unrecognized orbit filename: {orbit_name}")
    sat = m["sat"]
    vstart = m["vstart"]
    y = int(vstart[:4])
    mth = int(vstart[4:6])
    return shared_root / "Orbits" / "Sentinel-1" / "POEORB" / sat / f"{y:04d}" / f"{mth:02d}" / orbit_name


def ensure_poeorb_via_s1_orbits(zip_paths: List[str], shared_root: Path, require_poeorb: bool = True) -> List[str]:
    """
    For each ZIP, download the orbit file into `shared_root/` via s1_orbits,
    then place it under:
      shared_root/Orbits/Sentinel-1/POEORB/<SAT>/<YYYY>/<MM>/<file>.EOF
    """
    msgs: List[str] = []
    shared_root.mkdir(parents=True, exist_ok=True)

    for z in zip_paths:
        scene_id = _scene_id_from_zip(z)
        try:
            # download directly into shared_root (no more files landing in scripts/)
            dl_path = Path(s1_orbits.fetch_for_scene(scene_id, dir=shared_root))
        except Exception as e:
            msgs.append(f"ERR  {Path(z).name}: fetch failed ({e})")
            continue

        # normalize and validate
        try:
            dl_path = dl_path.resolve()
        except FileNotFoundError:
            msgs.append(f"ERR  {scene_id}: downloaded file missing ({dl_path})")
            continue

        name = dl_path.name
        m = _ORB_RE.match(name)
        if not m:
            # clean up unexpected downloads
            try: dl_path.unlink()
            except Exception: pass
            msgs.append(f"ERR  {scene_id}: unrecognized orbit filename {name}")
            continue

        # Respect POEORB requirement
        if require_poeorb and m["otype"] != "POEORB":
            # remove the RESORB we just fetched (keep cache clean)
            try: dl_path.unlink()
            except Exception: pass
            msgs.append(f"SKIP {scene_id}: got {name} (RESORB); require_poeorb=True")
            continue

        # Final destination in SNAP-style tree
        dest = _dest_path_for_orbit(shared_root, name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 0:
            # We already have it—discard duplicate download
            if dl_path != dest:
                try: dl_path.unlink()
                except Exception: pass
            msgs.append(f"SKIP {name} (exists)")
            continue

        # Move into place (atomic if same FS), else copy+unlink
        try:
            os.replace(dl_path, dest)
        except Exception:
            shutil.copy2(dl_path, dest)
            try: dl_path.unlink()
            except Exception: pass

        msgs.append(f"OK   {dest.relative_to(shared_root)}")

    return msgs


# ---------------------------------------------------------------------
# SNAP (gpt): get EGM96 via atomic download
# ---------------------------------------------------------------------
import urllib.request, tempfile

EGM96_URL = "http://step.esa.int/auxdata/dem/egm96/ww15mgh_b.zip"

def ensure_egm96_present(auxdata_dir: Path, logger: logging.Logger) -> Path:
    """
    Ensure <auxdata_dir>/dem/egm96/ww15mgh_b.zip exists and is non-empty.
    If missing/empty, download atomically. Returns the path.
    """
    egm_dir = auxdata_dir / "dem" / "egm96"
    egm_dir.mkdir(parents=True, exist_ok=True)
    egm_zip = egm_dir / "ww15mgh_b.zip"

    if egm_zip.exists() and egm_zip.stat().st_size > 0:
        logger.info(f"EGM96 present: {egm_zip} ({egm_zip.stat().st_size} bytes)")
        return egm_zip

    logger.info(f"EGM96 not found; downloading to {egm_zip} …")
    # Download to a temp file in the same directory, then move into place
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
        os.replace(tmp_path, egm_zip)
        try: os.chmod(egm_zip, 0o644)
        except Exception: pass
        logger.info(f"EGM96 downloaded: {egm_zip} ({size} bytes)")
        return egm_zip
    except Exception as e:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass
        raise


# ---------------------------------------------------------------------
# SNAP (gpt): core processing
# ---------------------------------------------------------------------

def _link_shared_auxdata(userdir: Path, auxdata_dir: Path) -> None:
    auxdata_dir = auxdata_dir.resolve()
    local = userdir / "auxdata"
    local.parent.mkdir(parents=True, exist_ok=True)

    # replace whatever is there
    try:
        if local.is_symlink():
            local.unlink(missing_ok=True)
        elif local.exists():
            shutil.rmtree(local, ignore_errors=True)
    except Exception:
        pass

    local.symlink_to(auxdata_dir, target_is_directory=True)


def snap_userdir(base_work_dir: Path) -> Path:
    # persistent, not under tmp
    return base_work_dir / "snap_user"


def verify_aux_visibility(userdir: Path, logger: logging.Logger) -> None:
    """Log what SNAP will see and raise if the EGM96 file is missing."""
    aux = userdir / "auxdata"
    egm = aux / "dem" / "egm96" / "ww15mgh_b.zip"
    logger.info(f"SNAP userdir: {userdir}")
    logger.info(f"SNAP auxdata symlink: {aux} -> {aux.resolve() if aux.exists() else 'MISSING'}")
    if not egm.exists() or egm.stat().st_size == 0:
        raise RuntimeError(
            f"EGM96 not found at {egm} — place ww15mgh_b.zip there or disable <externalDEMApplyEGM>."
        )
    logger.info(f"EGM96 present: {egm} ({egm.stat().st_size} bytes)")


def run_snap_gpt(
    zip_path: str,
    gpt_bin: str,
    graph_xml: str,
    prop_file: str | None,
    out_path: str,
    out_format: str,
    user_dir: Path,
    tmp_dir: Path,
    shared_aux_root: Path,
    q_threads: int = 12,
    cache_gb: str = "8G",
) -> None:
    
    # prepare dirs
    jtmp = tmp_dir / "java_tmp"
    ensure_dir(jtmp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # link shared auxdata if given
    if shared_aux_root is not None:
        _link_shared_auxdata(user_dir, shared_aux_root)

    # build command
    cmd = [
        gpt_bin, graph_xml,
        "-e",
        "-q", str(q_threads),
        "-c", cache_gb,
        "-f", out_format,
        "-t", out_path,
    ]
    if prop_file:
        cmd += ["-p", prop_file]
    cmd += [
        f"-Dsnap.userdir={user_dir}",
        f"-Duser.home={user_dir}",
        f"-Djava.io.tmpdir={jtmp}",
        "-Dsnap.engine.skipUpdateCheck=true",
        "-Dsnap.ui.disabled=true",
        "-Ds1tbx.downloadAuxData=false",
        "-Dsnap.product.library.disable=true",
        "-Dceres.logging.level=FINE",
        zip_path,
    ]

    env = os.environ.copy()
    env["HOME"] = str(user_dir)
    env["XDG_CACHE_HOME"] = str(user_dir / ".cache")
    env["XDG_CONFIG_HOME"] = str(user_dir / ".config")

    # execute command
    rc = subprocess.call(cmd, env=env)
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

def warp_one_source_to_cell_vrt(
    src_path: str, band_index: int, geom_dst, dst_srs: str, xres: float, yres: float, tap: bool=True
) -> str:
    """Warp a single source band to the cell grid as a /vsimem VRT and return its path."""
    xmin, ymin, xmax, ymax = geom_dst.bounds
    vrt_path = f"/vsimem/{Path(src_path).stem}_{abs(hash((xmin,ymin,xmax,ymax,band_index))) & 0xffffffff:x}.vrt"
    opts = gdal.WarpOptions(
        format="VRT",
        dstSRS=dst_srs,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres, yRes=yres,
        targetAlignedPixels=tap,
        resampleAlg="bilinear",
        cutlineWKT=geom_dst.wkt, cutlineSRS=dst_srs, cropToCutline=True,
        dstNodata=-9999.0,
        outputType=gdal.GDT_Float32,
        warpOptions={"NUM_THREADS":"ALL_CPUS","INIT_DEST":"NO_DATA","UNIFIED_SRC_NODATA":"YES"},
        srcBands=[band_index],
    )
    gdal.Warp(vrt_path, src_path, options=opts)
    return vrt_path


def write_median_uint16_from_vrts(
    vrt_paths: List[str], out_path: Path, db_min: float, db_max: float, block: int = 512
):
    """NaN-median stack → UInt16 dB (0=NoData), writes Scale/Offset, ZSTD+PREDICTOR=2."""
    if not vrt_paths:
        return
    ds0 = gdal.Open(vrt_paths[0])
    nx, ny = ds0.RasterXSize, ds0.RasterYSize
    gt, prj = ds0.GetGeoTransform(), ds0.GetProjection()
    ds0 = None

    co = ["TILED=YES","COMPRESS=ZSTD","ZSTD_LEVEL=15","PREDICTOR=2","BIGTIFF=YES","BLOCKXSIZE=512","BLOCKYSIZE=512"]
    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(str(out_path), nx, ny, 1, gdal.GDT_UInt16, options=co)
    dst.SetGeoTransform(gt); dst.SetProjection(prj)
    b = dst.GetRasterBand(1); b.SetNoDataValue(0)

    scale = (db_max - db_min) / (65534 - 1)
    offset = db_min - scale * 1
    b.SetScale(scale); b.SetOffset(offset)

    for y0 in range(0, ny, block):
        ysize = min(block, ny - y0)
        for x0 in range(0, nx, block):
            xsize = min(block, nx - x0)
            stack = []
            for p in vrt_paths:
                d = gdal.Open(p)
                a = d.GetRasterBand(1).ReadAsArray(x0, y0, xsize, ysize).astype(np.float32)
                a[a <= -9998.5] = np.nan
                stack.append(a)
                d = None
            cube = np.stack(stack, axis=0)
            med = np.nanmedian(cube, axis=0).astype(np.float32)

            dn = np.round((med - db_min) * (65533.0 / (db_max - db_min)) + 1.0)
            dn = np.clip(dn, 1.0, 65534.0).astype(np.uint16)
            dn[np.isnan(med)] = 0
            b.WriteArray(dn, xoff=x0, yoff=y0)

    b.FlushCache(); dst.FlushCache(); dst = None
    for p in vrt_paths:
        gdal.Unlink(p)
