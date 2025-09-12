#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper utilities for Sentinel-2 L2A download + processing on CDSE.
Requires: requests, pandas, shapely, GDAL (gdal-bin), Python 3.10+
"""

from __future__ import annotations
import os
import re
import csv
import json
import time
import sys
import shutil
import stat
import subprocess
import threading
import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List
from glob import glob
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from osgeo import gdal
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from shapely import wkt as shp_wkt
import xml.etree.ElementTree as ET
from getpass import getpass  # fix: use function, not module


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
CDSE_BASE = "https://download.dataspace.copernicus.eu"
AUTH_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


# ---------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------

def setup_rank_logging(log_dir: Path, rank: int, level=logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.addLevelName(logging.CRITICAL, "FATAL")
    logging.addLevelName(logging.WARNING,  "WARN")

    logger = logging.getLogger("s2_pipeline")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    class _LowercaseFilter(logging.Filter):
        def filter(self, record):
            record.levelname_lower = record.levelname.lower()
            record.rank = rank
            return True

    filt = _LowercaseFilter()
    fmt = logging.Formatter(
        "%(asctime)s [rank %(rank)d] [%(levelname_lower)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_dir / f"pipeline_rank{rank}.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt); fh.addFilter(filt)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt); ch.addFilter(filt)
        logger.addHandler(ch)

    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run(cmd: Sequence[str], check=True, env: dict | None = None) -> None:
    subprocess.run(cmd, check=check, env=env)


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def gdalinfo_json(p: Path) -> dict:
    out = subprocess.run(["gdalinfo", "-json", str(p)],
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


# ---------------------------------------------------------------------
# AOI & GRID
# ---------------------------------------------------------------------

def load_gridcells(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"tile_id_rc","tile_id_geo","xmin","ymin","xmax","ymax","row","col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Grid CSV missing columns: {missing}")
    return df


def select_intersecting_cells(aoi: BaseGeometry, grid: pd.DataFrame) -> pd.DataFrame:
    axmin, aymin, axmax, aymax = aoi.bounds
    rough = grid.loc[
        (grid["xmax"] > axmin) & (grid["xmin"] < axmax) &
        (grid["ymax"] > aymin) & (grid["ymin"] < aymax)
    ].copy()

    keep_rows = []
    for _, r in rough.iterrows():
        cell_poly = box(r["xmin"], r["ymin"], r["xmax"], r["ymax"])
        if aoi.intersects(cell_poly):
            keep_rows.append(r)
    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    if not out.empty:
        out = out.sort_values(["row","col"]).reset_index(drop=True)
    return out


def union_bounds(df_cells: pd.DataFrame) -> tuple[float,float,float,float]:
    return (df_cells["xmin"].min(), df_cells["ymin"].min(),
            df_cells["xmax"].max(), df_cells["ymax"].max())

# ---------------------------------------------------------------------
# ODATA QUERY
# ---------------------------------------------------------------------

def fetch_all_products(base_query: str, top: int = 200, timeout: int = 60) -> pd.DataFrame:
    url = f"{base_query}&$count=true&$top={top}"
    items = []
    while url:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        items.extend(j.get("value", []))
        url = j.get("@odata.nextLink")
    return pd.DataFrame(items)


def build_search_query(
    aoi: BaseGeometry,
    catalogue_odata: str,
    collection_name: str,
    product_type: str,
    start_iso: str,
    end_iso: str,
) -> str:
    aoi_wkt = aoi.wkt
    return (
        f"{catalogue_odata}/Products?"
        f"$filter=Collection/Name eq '{collection_name}' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start gt {start_iso} and ContentDate/Start lt {end_iso}"
    )


# ---------------------------------------------------------------------
# AUTH & SESSION
# ---------------------------------------------------------------------

def _now() -> float: 
    return time.time()


def _password_grant(credentials: dict) -> dict:
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": credentials["username"],
        "password": credentials["password"],
    }
    r = requests.post(AUTH_URL, data=data, timeout=30)
    r.raise_for_status()
    return r.json()


def _refresh_grant(refresh_token: str) -> dict:
    data = {
        "client_id": "cdse-public",
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    r = requests.post(AUTH_URL, data=data, timeout=30)
    r.raise_for_status()
    return r.json()


def _read_secret_file(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path).expanduser()
    st = p.stat()
    # refuse group/other perms for safety
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError(f"Insecure permissions on {p}; run: chmod 600 {p}")
    return p.read_text(encoding="utf-8").strip()


def get_access_token(credentials: dict, token_cache: dict, *, force_refresh: bool=False) -> str:
    """
    credentials: {"username": str|None, "password": str|None}
    token_cache: {"access_token": str|None, "expires_at": float,
                  "refresh_token": str|None, "refresh_expires_at": float}
    Looks for creds in this order:
      1) provided credentials dict
      2) env CDSE_USERNAME / CDSE_PASSWORD
      3) env CDSE_PASSWORD_FILE (file contents)
    No interactive prompt is used unless both password sources are missing and a TTY exists.
    """
    now = _now()

    tok = token_cache.get("access_token")
    exp = float(token_cache.get("expires_at") or 0)
    if tok and (exp > now + 60) and not force_refresh:
        return tok

    # try refresh token first
    rtok = token_cache.get("refresh_token")
    rtexp = float(token_cache.get("refresh_expires_at") or 0)
    if rtok and (rtexp > now + 60):
        try:
            j = _refresh_grant(rtok)
            token_cache["access_token"] = j["access_token"]
            token_cache["expires_at"] = now + int(j.get("expires_in", 3600))
            if "refresh_token" in j:
                token_cache["refresh_token"] = j["refresh_token"]
                token_cache["refresh_expires_at"] = now + int(j.get("refresh_expires_in", 0) or 0)
            return token_cache["access_token"]
        except Exception:
            pass  # fall back to password grant

    # fill username/password non-interactively
    if not credentials.get("username"):
        credentials["username"] = os.getenv("CDSE_USERNAME")

    if not credentials.get("password"):
        pw = os.getenv("CDSE_PASSWORD")
        if not pw:
            pw = _read_secret_file(os.getenv("CDSE_PASSWORD_FILE"))
        credentials["password"] = pw

    # if still missing, only prompt if a TTY is available
    if not credentials.get("username") or not credentials.get("password"):
        if sys.stdin.isatty():
            # prompt once in interactive runs
            if not credentials.get("username"):
                credentials["username"] = input("CDSE username: ").strip()
            if not credentials.get("password"):
                credentials["password"] = getpass("CDSE password: ")
        else:
            raise RuntimeError("Missing CDSE credentials and no TTY; set CDSE_USERNAME and CDSE_PASSWORD_FILE.")

    # password grant
    j = _password_grant(credentials)
    token_cache["access_token"] = j["access_token"]
    token_cache["expires_at"] = now + int(j.get("expires_in", 3600))
    token_cache["refresh_token"] = j.get("refresh_token")
    token_cache["refresh_expires_at"] = now + int(j.get("refresh_expires_in", 0) or 0)
    return token_cache["access_token"]


# ---------------------------------------------------------------------
# NODE BROWSER
# ---------------------------------------------------------------------

def _nodes_url(product_id, *segments, list_children: bool = False) -> str:
    """
    Build Nodes URL using global CDSE_BASE.
    """
    from urllib.parse import quote
    path = f"{CDSE_BASE}/odata/v1/Products({product_id})"
    for seg in segments:
        path += f"/Nodes({quote(str(seg), safe='')})"
    path += "/Nodes" if list_children else "/$value"
    return path


def list_children(session: requests.Session, product_id, *segments) -> list[dict]:
    """
    List children of a node. For top-level, call with only (session, product_id).
    AutoRefreshSession (from pipeline) will handle any 401s internally.
    """
    url = _nodes_url(product_id, *segments, list_children=True)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json().get("result", [])


def find_safe_root_node(session: requests.Session, product_id) -> str:
    """
    Return the name of the <PRODUCT>.SAFE root node.
    """
    kids = list_children(session, product_id)
    if not kids:
        raise RuntimeError("No nodes found for product.")
    for k in kids:
        nm = k.get("Name", "")
        if nm.endswith(".SAFE"):
            return nm
    return kids[0]["Name"]  # fallback


def extract_tile_from_name(product_name: str) -> str | None:
    for part in str(product_name).split("_"):
        if part.startswith("T") and len(part) == 6:
            return part
    return None


def select_targets(session: requests.Session, product_id, product_name: str,
                   bands: Iterable[str], bands_res: str) -> tuple[list[tuple], str, str | None]:
    """
    Build the list of file-node paths (as tuples of node segments) to download.
    """
    root = find_safe_root_node(session, product_id)

    granule_dir = None
    tile = extract_tile_from_name(product_name)
    granules = list_children(session, product_id, root, "GRANULE")
    for g in granules:
        gname = g.get("Name", "")
        if (tile and tile in gname) or gname.startswith("L2A_"):
            granule_dir = gname
            break
    if not granule_dir and granules:
        granule_dir = granules[0]["Name"]

    targets: list[tuple] = []
    if granule_dir:
        try:
            res = list_children(session, product_id, root, "GRANULE", granule_dir, "IMG_DATA", f"R{bands_res}m")
            for node in res:
                fname = node.get("Name", "")
                for b in bands:
                    if fname.endswith(f"_{b}_{bands_res}m.jp2"):
                        targets.append((root, "GRANULE", granule_dir, "IMG_DATA", f"R{bands_res}m", fname))
                        break
        except requests.HTTPError:
            pass
        # tile-level XML
        targets.append((root, "GRANULE", granule_dir, "MTD_TL.xml"))

    # product-level XML
    targets.append((root, "MTD_MSIL2A.xml"))
    return targets, root, granule_dir


def relpath_for_segments(segments: Sequence[str], include_safe_root: bool = True,
                         pb_tag: str | None = None) -> str:
    if include_safe_root:
        return os.path.join(*segments)
    parts = list(segments)
    fname = parts[-1]
    base, ext = os.path.splitext(fname)
    if pb_tag:
        fname = f"{base}_{pb_tag}{ext}"
    return os.path.join(*parts[1:-1], fname)


def download_node(session: requests.Session, product_id, segments: Sequence[str],
                  output_root: str | Path, *,
                  include_safe_root: bool = True, pb_tag: str | None = None) -> str:
    """
    Stream a single node (file) to disk. Returns "ok" or an error string.
    """
    # build local path
    if include_safe_root:
        subpath = os.path.join(*segments)
        outpath = os.path.join(output_root, subpath)
    else:
        parts = list(segments)
        fname = parts[-1]
        base, ext = os.path.splitext(fname)
        if pb_tag:
            fname = f"{base}_{pb_tag}{ext}"
        subpath = os.path.join(*parts[1:-1], fname)
        outpath = os.path.join(output_root, subpath)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # GET once; AutoRefreshSession handles 401 retry
    url = _nodes_url(product_id, *segments)
    r = session.get(url, stream=True, timeout=300)
    if r.status_code != 200:
        return f"error: HTTP {r.status_code}"

    tmp = outpath + ".part"
    try:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, outpath)
        return "ok"
    except Exception as ex:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        finally:
            return f"write error: {ex}"
        

class AutoRefreshSession(requests.Session):
    _refresh_lock = threading.Lock()

    def __init__(self, credentials: dict, token_cache: dict, logger: logging.Logger | None = None):
        super().__init__()
        self._credentials = credentials
        self._token_cache = token_cache
        self._logger = logger

        tok = get_access_token(self._credentials, self._token_cache, force_refresh=False)
        self.headers.update({"Authorization": f"Bearer {tok}"})

        retries = Retry(
            total=5, backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=retries)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def request(self, method, url, **kwargs):
        r = super().request(method, url, **kwargs)
        if r.status_code == 401:
            if self._logger:
                self._logger.info("token expired; refreshing and retrying once...")
            with AutoRefreshSession._refresh_lock:
                new_tok = get_access_token(self._credentials, self._token_cache, force_refresh=True)
                self.headers["Authorization"] = f"Bearer {new_tok}"
            r = super().request(method, url, **kwargs)
        return r


def make_auto_session(credentials: dict, token_cache: dict, logger: logging.Logger | None = None) -> AutoRefreshSession:
    return AutoRefreshSession(credentials, token_cache, logger=logger)


def download_rows_concurrent(
    df: pd.DataFrame,
    output_dir: Path,
    bands: Iterable[str],
    bands_res: str,
    scene_csv: Path,
    *,
    max_workers: int = 2,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Download rows concurrently (at most max_workers at a time).
    Returns a list of failure dicts [{Id, Name, status}, ...]."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    CRED: dict = {}
    TOK: dict = {}

    def one(row: pd.Series) -> tuple[str, str, str]:
        name = row.get("Name", "unknown")
        pid = row.get("Id", "")
        try:
            with make_auto_session(CRED, TOK, logger=logger) as sess:
                status = download_selected_files_from_cdse_row(
                    row, sess, output_dir,
                    bands=bands, bands_res=bands_res, scene_csv=scene_csv
                )
            return (pid, name, status)
        except Exception as e:
            return (pid, name, f"exception: {e}")

    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, row) for _, row in df.iterrows()]
        for fut in as_completed(futs):
            pid, name, status = fut.result()
            if status != "ok":
                failures.append({"Id": pid, "Name": name, "status": status})
                if logger:
                    logger.warning(f"download failed for {name} ({pid}): {status}")
    return failures


# ---------------------------------------------------------------------
# INDEXING
# ---------------------------------------------------------------------

def _local(el):
    t = el.tag
    return t.split('}', 1)[1].lower() if '}' in t else t.lower()


def _num(s):
    if s is None:
        return None
    m = re.search(r'(-?\d+(?:\.\d+)?)', str(s))
    return float(m.group(1)) if m else None


def _get_child_text(el, name):
    lname = name.lower()
    for ch in el:
        if _local(ch) == lname:
            return ch.text
    return None


def _res_from(el):
    for k, v in el.attrib.items():
        if k.lower() == "resolution":
            rv = _num(v)
            if rv is not None:
                return rv
    rv = _num(_get_child_text(el, "Resolution"))
    if rv is not None:
        return rv
    m = re.search(r'(\d+)\s*m?$', _local(el))
    return float(m.group(1)) if m else None


def parse_tile_xml(xml_path: str, resolution: str) -> dict:
    target = int(float(resolution))
    root = ET.parse(xml_path).getroot()

    epsg = None
    for el in root.iter():
        if _local(el).endswith("horizontal_cs_code"):
            m = re.search(r"EPSG:(\d+)", (el.text or ""), re.IGNORECASE)
            if m:
                epsg = int(m.group(1)); break

    search_root = None
    for el in root.iter():
        if _local(el).endswith("tile_geocoding"):
            search_root = el; break
    if search_root is None:
        search_root = root

    geopos, sizes = [], []
    for el in search_root.iter():
        name = _local(el)
        if name.startswith("geoposition"):
            geopos.append(el)
        elif name.startswith("size"):
            sizes.append(el)

    gp_best, best_d = None, 1e9
    for gp in geopos:
        ulx  = _num(_get_child_text(gp, "ULX"))
        uly  = _num(_get_child_text(gp, "ULY"))
        xdim = _num(_get_child_text(gp, "XDIM"))
        ydim = _num(_get_child_text(gp, "YDIM"))
        r    = _res_from(gp)
        if r == target and None not in (ulx, uly, xdim, ydim):
            gp_best, best_d = (ulx, uly, xdim, ydim), 0.0; break
        dists = [abs(abs(v) - target) for v in (xdim, ydim) if v is not None]
        d = min(dists) if dists else 1e9
        if d < best_d and None not in (ulx, uly, xdim, ydim):
            gp_best, best_d = (ulx, uly, xdim, ydim), d

    sz_best, best_s = None, 1e9
    for sz in sizes:
        nrows = _num(_get_child_text(sz, "NROWS"))
        ncols = _num(_get_child_text(sz, "NCOLS"))
        r     = _res_from(sz)
        if r == target and None not in (nrows, ncols):
            sz_best, best_s = (int(ncols), int(nrows)), 0.0; break
        if None not in (nrows, ncols):
            d = abs(nrows - 5490) + abs(ncols - 5490)
            if d < best_s:
                sz_best, best_s = (int(ncols), int(nrows)), d

    if not gp_best or not sz_best:
        raise ValueError(f"Incomplete geocoding in {xml_path} for resolution {resolution} m")

    ulx, uly, xdim, ydim = gp_best
    ncols, nrows = sz_best
    x2 = ulx + ncols * xdim
    y2 = uly + nrows * ydim
    xmin, xmax = (ulx, x2) if ulx <= x2 else (x2, ulx)
    ymin, ymax = (y2, uly) if y2 <= uly else (uly, y2)

    return dict(
        epsg=int(epsg) if epsg is not None else None,
        xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax),
        xres=abs(float(xdim)), yres=abs(float(ydim)),
        ncols=int(ncols), nrows=int(nrows),
    )


def append_scene_row(csv_path: Path, row: dict):
    fields = ["scene_id","product_name","epsg","xmin","ymin","xmax","ymax",
              "xres","yres","ncols","nrows","tile_xml_path","added_utc"]
    write_header = not csv_path.exists()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("scene_id") == row["scene_id"]:
                    return
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def append_scene_rows_bulk(csv_path: Path, rows: list[dict]) -> int:
    fields = ["scene_id","product_name","epsg","xmin","ymin","xmax","ymax",
              "xres","yres","ncols","nrows","tile_xml_path","added_utc"]
    wrote = 0
    existing = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                sid = r.get("scene_id")
                if sid: existing.add(sid)
    new_rows = [r for r in rows if r.get("scene_id") not in existing]
    if not new_rows:
        return 0
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in new_rows:
            w.writerow({k: r.get(k) for k in fields})
            wrote += 1
    return wrote


def backfill_index_from_existing_xmls(output_root: str | Path, bands_res: str, csv_path: Path) -> int:
    rows = []
    for xml in Path(output_root).glob("**/GRANULE/*/MTD_T*.xml"):
        try:
            geo = parse_tile_xml(str(xml), resolution=bands_res)
        except Exception as e:
            print("BACKFILL parse error:", xml, e); continue
        safe = next((p for p in xml.parents if p.name.endswith(".SAFE")), None)
        product_name = safe.name.replace(".SAFE","") if safe else "UNKNOWN"
        scene_id = xml.parent.name
        rows.append(dict(
            scene_id=scene_id, product_name=product_name,
            epsg=geo["epsg"], xmin=geo["xmin"], ymin=geo["ymin"],
            xmax=geo["xmax"], ymax=geo["ymax"], xres=geo["xres"], yres=geo["yres"],
            ncols=geo["ncols"], nrows=geo["nrows"], tile_xml_path=str(xml),
            added_utc=utc_now_iso(),
        ))
    wrote = append_scene_rows_bulk(csv_path, rows)
    return wrote


# ---------------------------------------------------------------------
# DOWNLOAD ONE PRODUCT
# ---------------------------------------------------------------------

def _guess_local_granule_dir(safe_dir: Path, product_name: str) -> Path | None:
    """Pick the most likely GRANULE dir on disk without hitting the API."""
    tile = extract_tile_from_name(product_name)
    candidates = sorted((safe_dir / "GRANULE").glob("*"))
    if not candidates:
        return None
    # Prefer one that includes the tile id, else first match
    for c in candidates:
        if tile and tile in c.name:
            return c
    for c in candidates:
        if c.name.startswith("L2A_"):
            return c
    return candidates[0]


def fast_local_complete_safe(output_root: Path, product_name: str,
                             bands_res: str,
                             bands: Iterable[str] | None = None,
                             require_xml: bool = True) -> tuple[bool, dict]:
    """
    Returns (is_complete, detail_dict). Does **no network I/O**.
    Complete means: for the chosen GRANULE, all requested bands (plus SCL)
    exist under IMG_DATA/R{bands_res}m, and (optionally) MTD_TL.xml + MTD_MSIL2A.xml exist.
    """
    safe_dir = output_root / (product_name if product_name.endswith(".SAFE") else f"{product_name}.SAFE")
    if not safe_dir.exists():
        return False, {"reason": "safe_dir_missing", "safe_dir": str(safe_dir)}

    gran = _guess_local_granule_dir(safe_dir, product_name)
    if gran is None:
        return False, {"reason": "granule_missing", "safe_dir": str(safe_dir)}

    res_dir = gran / "IMG_DATA" / f"R{bands_res}m"
    if not res_dir.exists():
        return False, {"reason": f"r{bands_res}m_missing", f"r{bands_res}": str(res_dir)}

    missing = []
    present = 0
    for b in bands:
        pat = str(res_dir / f"*_{b}_{bands_res}m.jp2")
        matches = glob(pat)
        if matches:
            present += 1
        else:
            missing.append(b)

    if require_xml:
        tile_xml = gran / "MTD_TL.xml"
        prod_xml = safe_dir / "MTD_MSIL2A.xml"
        xml_ok = tile_xml.exists() and prod_xml.exists()
    else:
        xml_ok = True

    ok = (not missing) and xml_ok
    return ok, {
        "present_bands": int(present),
        "missing_bands": missing,
        "xml_ok": xml_ok,
        "granule_dir": str(gran),
        "safe_dir": str(safe_dir),
    }


def download_selected_files_from_cdse_row(
    row: pd.Series,
    session: requests.Session,
    output_dir: str | Path,
    bands: Iterable[str],
    bands_res: str,
    scene_csv: str | Path,
    id_col: str = "Id",
    name_col: str = "Name",
) -> str:
    product_id = row[id_col]
    product_name = row[name_col]

    # ---- FAST LOCAL CHECK (no network) ----
    ok, detail = fast_local_complete_safe(Path(output_dir), product_name, bands_res, bands)
    if ok:
        # we still try to index (scene_xml parse) later if needed, but skip any node listing
        print(f"SKIP: complete for {product_name} (all bands + XML present)")
        # attempt index append from existing tile XML
        gran_dir = Path(detail["granule_dir"])
        tile_xml_path = gran_dir / "MTD_TL.xml"
        if tile_xml_path.exists():
            try:
                geo = parse_tile_xml(str(tile_xml_path), resolution=bands_res)
                append_scene_row(scene_csv, dict(
                    scene_id=gran_dir.name, product_name=product_name,
                    epsg=geo["epsg"], xmin=geo["xmin"], ymin=geo["ymin"], xmax=geo["xmax"], ymax=geo["ymax"],
                    xres=geo["xres"], yres=geo["yres"], ncols=geo["ncols"], nrows=geo["nrows"],
                    tile_xml_path=str(tile_xml_path), added_utc=utc_now_iso(),
                ))
            except Exception as e:
                print(f"INDEX ERROR parsing {tile_xml_path}: {e}")
        return "ok"

    # ---- Otherwise fall back to API node listing (slower path) ----
    targets, root, granule_dir = select_targets(session, product_id, product_name, bands, bands_res)
    if not targets:
        print(f"NO TARGETS for {product_name}")
        return "error: no targets found"

    INCLUDE_SAFE = True
    PB_TAG = None

    present, missing = [], []
    for segs in targets:
        rel = relpath_for_segments(segs, include_safe_root=INCLUDE_SAFE, pb_tag=PB_TAG)
        full = os.path.join(output_dir, rel)
        (present if os.path.exists(full) else missing).append(segs)

    if not missing:
        print(f"SKIP: complete for {product_name} (all {len(targets)} files present)")
    else:
        print(f"{product_name}: {len(present)} present, {len(missing)} missing → downloading missing only")
        for segs in missing:
            status = download_node(session, product_id, segs, output_dir,
                                   include_safe_root=INCLUDE_SAFE, pb_tag=PB_TAG)
            if status != "ok":
                print("  ", segs, "->", status)

    # Indexing via tile XML (works whether we downloaded or already had it)
    tile_xml_rel = None
    for segs in targets:
        if isinstance(segs, tuple) and len(segs) >= 4 and segs[-1] == "MTD_TL.xml":
            tile_xml_rel = relpath_for_segments(segs, include_safe_root=True, pb_tag=None)
            break

    if not tile_xml_rel:
        print(f"INDEX SKIP: no tile XML for {product_name}")
        return "ok"

    tile_xml_path = os.path.join(output_dir, tile_xml_rel)
    if not os.path.exists(tile_xml_path):
        print(f"INDEX SKIP: tile XML missing on disk: {tile_xml_path}")
        return "ok"

    scene_id = granule_dir or product_name
    try:
        geo = parse_tile_xml(tile_xml_path, resolution=bands_res)
        append_scene_row(scene_csv, dict(
            scene_id=scene_id, product_name=product_name,
            epsg=geo["epsg"], xmin=geo["xmin"], ymin=geo["ymin"], xmax=geo["xmax"], ymax=geo["ymax"],
            xres=geo["xres"], yres=geo["yres"], ncols=geo["ncols"], nrows=geo["nrows"],
            tile_xml_path=tile_xml_path, added_utc=utc_now_iso(),
        ))
    except Exception as e:
        print(f"INDEX ERROR parsing {tile_xml_path}: {e}")

    return "ok"


# ---------------------------------------------------------------------
# MASK AND OFFSET VRT BUILDER
# ---------------------------------------------------------------------

DST_NODATA_DEFAULT = 65535

def parse_pb_from_path(p: Path) -> float | None:
    m = re.search(r"_N(\d{4})_", str(p))
    return float(m.group(1))/100 if m else None


def write_mask_and_offset_vrt(
    band_jp2: Path,
    scl_jp2: Path,
    out_vrt: Path,
    scl_classes: List[int],
    *,
    dst_nodata: int = 65535,
    dn_offset: int = 0,  # e.g., 1000; pass 0 if no PB offset
) -> None:
    # read grid/SRS from the band
    j = gdalinfo_json(band_jp2)
    w, h = j["size"]
    gt = j["geoTransform"]
    srs_wkt = j.get("coordinateSystem", {}).get("wkt", "")

    band_rel = os.path.relpath(band_jp2, out_vrt.parent)
    scl_rel  = os.path.relpath(scl_jp2,  out_vrt.parent)
    classes_csv = ",".join(map(str, sorted(set(scl_classes))))

    vrt_xml = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>{srs_wkt}</SRS>
  <GeoTransform>{gt[0]},{gt[1]},{gt[2]},{gt[3]},{gt[4]},{gt[5]}</GeoTransform>

  <VRTRasterBand dataType="UInt16" band="1" subClass="VRTDerivedRasterBand">
    <NoDataValue>{dst_nodata}</NoDataValue>

    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionType>mask_and_offset</PixelFunctionType>
    <PixelFunctionArguments nodata="{dst_nodata}" classes="{classes_csv}" dn_offset="{dn_offset}"/>
    <PixelFunctionCode><![CDATA[
import numpy as np, re
LUT = None

def _to_int(x, default=0):
    if x is None: 
        return default
    if isinstance(x, bytes): 
        x = x.decode('utf-8', 'ignore')
    s = str(x).strip().strip('"').strip("'")
    return default if s=="" else int(s)

def _to_csv_str(x):
    if x is None: 
        return ""
    if isinstance(x, bytes): 
        x = x.decode('utf-8', 'ignore')
    return str(x).strip().strip('"').strip("'")

def mask_and_offset(in_ar, out_ar, *args, **kwargs):
    A = in_ar[0].astype(np.uint16, copy=False)  # reflectance DN
    B = in_ar[1].astype(np.uint8, copy=False)  # SCL

    nd = _to_int(kwargs.get("nodata"), 65535)
    dn_off = _to_int(kwargs.get("dn_offset"), 0)
    classes = _to_csv_str(kwargs.get("classes"))

    global LUT
    if LUT is None:
        codes = [int(t) for t in re.findall(r"\d+", classes)]
        LUT = np.zeros(256, dtype=bool)
        LUT[codes] = True

    # Always mask SCL==0 and any code > 11
    mask = (B == 0) | (B > 11) | LUT[B]

    out = A.copy()
    out[mask] = nd

    if dn_off != 0:
        valid = (out != nd)
        if valid.any():
            tmp = out.astype(np.int32, copy=True)
            np.subtract(tmp, dn_off, out=tmp, where=valid)
            np.clip(tmp, 0, 65534, out=tmp)
            out[valid] = tmp[valid].astype(np.uint16)

    out_ar[:] = out
]]>
    </PixelFunctionCode>

    <!-- Reflectance -->
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{band_rel}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
    </SimpleSource>

    <!-- SCL -->
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{scl_rel}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
    </SimpleSource>

  </VRTRasterBand>
</VRTDataset>
"""
    ensure_dir(out_vrt.parent)
    tmp = out_vrt.with_suffix(out_vrt.suffix + ".tmp")
    tmp.write_text(vrt_xml)
    tmp.replace(out_vrt)


def warp_to_wgs84_vrt(
    src_vrt: Path,
    out_vrt: Path,
    *,
    dst_nodata: int = DST_NODATA_DEFAULT,
    target_srs: str = "EPSG:4326",
    tr: tuple[float,float] | None = None,
    te: tuple[float,float,float,float] | None = None,
    tap: bool = False
) -> None:
    """Create a VRT that is a WGS84-warped view of src_vrt (single-threaded; MPI-friendly)."""
    cmd = [
        "gdalwarp",
        "-of", "VRT",
        "-overwrite",
        "-t_srs", target_srs,
        "-r", "near",
        "-srcnodata", str(dst_nodata),
        "-dstnodata", str(dst_nodata),
        "-wo", "NUM_THREADS=1",
        "--config", "GDAL_VRT_ENABLE_PYTHON", "YES",
        "--config", "GDAL_NUM_THREADS", "1",
    ]
    if tr:
        cmd += ["-tr", str(tr[0]), str(tr[1])]
    if te:
        cmd += ["-te", str(te[0]), str(te[1]), str(te[2]), str(te[3]), "-te_srs", target_srs]
    if tap:
        cmd += ["-tap"]
    cmd += [str(src_vrt)]

    ensure_dir(out_vrt.parent)
    tmp = out_vrt.with_suffix(out_vrt.suffix + ".tmp")
    run(cmd + [str(tmp)])
    tmp.replace(out_vrt)


# ---------------------------------------------------------------------
# MEDIAN COMPOSITE AND CLIP TO GRIDCELL
# ---------------------------------------------------------------------

def list_wgs84_vrts(rank_tmp: Path) -> List[Path]:
    """Find Phase-1 warped scene VRTs under tmp/rank_*/wgs84_vrt/."""
    d = rank_tmp / "wgs84_vrt"
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*_masked_harmonized_wgs84.vrt") if p.is_file())


def ds_bounds(vrt_path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for a GDAL dataset (assumes north-up)."""
    ds = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"cannot open: {vrt_path}")
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    x0, px, rx, y0, ry, py = gt
    if rx != 0 or ry != 0:
        raise ValueError(f"rotation not supported for bounds: {vrt_path}")
    xs = [x0, x0 + w * px]
    ys = [y0, y0 + h * py]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ds = None
    return (xmin, ymin, xmax, ymax)


def bbox_intersects(a: Tuple[float,float,float,float],
                    b: Tuple[float,float,float,float]) -> bool:
    """Axis-aligned bbox intersection test (a,b=(xmin,ymin,xmax,ymax))."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def build_stack_vrt(candidates: Iterable[Path],
                    cell_bbox: tuple[float, float, float, float],
                    out_vrt: Path,
                    *,
                    nodata: int = 65535) -> Path:
    """
    Stack candidate scenes as separate bands in a VRT, cropped to cell_bbox.
    Uses gdalbuildvrt -separate; output written to a .tmp then atomically moved.
    """
    out_vrt.parent.mkdir(parents=True, exist_ok=True)

    # If nothing to stack, bail early
    candidates = [Path(p) for p in candidates]
    if not candidates:
        raise RuntimeError("build_stack_vrt: no candidate scenes")

    # Write the file list
    lst = out_vrt.with_suffix(out_vrt.suffix + ".list.txt")
    lst.write_text("\n".join(str(p) for p in candidates))

    # Build to a temp path, then rename
    tmp = out_vrt.with_suffix(out_vrt.suffix + ".tmp")

    cmd = [
        "gdalbuildvrt",
        "-overwrite",
        "-separate",
        "-srcnodata", str(nodata),
        "-vrtnodata", str(nodata),
        "-te", str(cell_bbox[0]), str(cell_bbox[1]), str(cell_bbox[2]), str(cell_bbox[3]),
        "-input_file_list", str(lst),
        str(tmp)  # OUTPUT LAST
    ]

    env = os.environ.copy()
    try:
        res = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        if not tmp.exists():
            # gdalbuildvrt returned 0 but no file was created
            raise RuntimeError(f"gdalbuildvrt produced no output: {tmp}\nSTDERR:\n{res.stderr}")
        os.replace(tmp, out_vrt)
        return out_vrt
    except subprocess.CalledProcessError as cpe:
        raise RuntimeError(f"gdalbuildvrt failed ({cpe.returncode}):\n{cpe.stderr}") from None
    finally:
        # cleanup aux files
        try: tmp.unlink()
        except: pass
        try: lst.unlink()
        except: pass


def build_median_vrt_from_stack(stack_vrt: Path,
                                out_vrt: Path,
                                *,
                                nodata_in: int = 65535,
                                nodata_out: float = -9999.0) -> Path:
    out_vrt.parent.mkdir(parents=True, exist_ok=True)

    ds = gdal.Open(str(stack_vrt), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"cannot open: {stack_vrt}")
    w, h = ds.RasterXSize, ds.RasterYSize
    srs_wkt = ds.GetProjection()
    gt = ds.GetGeoTransform()
    n_bands = ds.RasterCount
    ds = None

    src_rel = os.path.relpath(stack_vrt, out_vrt.parent)
    sources_xml = "\n".join(
        f"""    <SimpleSource>
        <SourceFilename relativeToVRT="1">{src_rel}</SourceFilename>
        <SourceBand>{i}</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
        <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
        </SimpleSource>"""
        for i in range(1, n_bands + 1)
    )

    vrt_xml = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>{srs_wkt}</SRS>
  <GeoTransform>{gt[0]},{gt[1]},{gt[2]},{gt[3]},{gt[4]},{gt[5]}</GeoTransform>

  <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
    <NoDataValue>{nodata_out}</NoDataValue>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionType>masked_median</PixelFunctionType>
    <PixelFunctionArguments nodata_in="{nodata_in}" nodata_out="{nodata_out}"/>
    <PixelFunctionCode><![CDATA[
import numpy as np
def _to_float(x, default):
    if x is None: return default
    if isinstance(x, bytes): x = x.decode('utf-8','ignore')
    s = str(x).strip().strip('"').strip("'")
    try: return float(s)
    except: return default

def masked_median(in_ar, out_ar, *args, **kwargs):
    # in_ar is a tuple of 2D arrays (one per SimpleSource band)
    nd_in  = _to_float(kwargs.get("nodata_in"),  65535.0)
    nd_out = _to_float(kwargs.get("nodata_out"), -9999.0)

    # Stack to (N, y, x) and work in float32
    arr = np.asarray(in_ar, dtype=np.float32)

    # Treat input nodata as NaN
    arr[arr == nd_in] = np.nan

    # Median across the stack, ignoring NaNs
    med = np.nanmedian(arr, axis=0)

    # Fill all-NaN pixels with output nodata and write
    np.nan_to_num(med, copy=False, nan=nd_out)
    out_ar[:] = med.astype(np.float32, copy=False)
]]></PixelFunctionCode>
{sources_xml}
  </VRTRasterBand>
</VRTDataset>
"""
    tmp = out_vrt.with_suffix(out_vrt.suffix + ".tmp")
    tmp.write_text(vrt_xml)
    tmp.replace(out_vrt)
    return out_vrt


def warp_cutline_wkt_py(
    src_path: Path,
    out_tif: Path,
    cutline_wkt: str,
    *,
    dst_nodata: float = -9999.0,
    resample: str = "near",  # e.g., "near","bilinear","cubic","cubicspline","lanczos","average","mode","max","min","med","q1","q3"
    num_threads: int = 1,
    creation_opts: Sequence[str] = ("TILED=YES","COMPRESS=ZSTD","PREDICTOR=2","BIGTIFF=YES"),
) -> Path:
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # Make a tiny cutline GeoJSON next to the output (same FS)
    geom = shp_wkt.loads(cutline_wkt)
    gj = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry": mapping(geom)}]}
    with tempfile.NamedTemporaryFile("w", suffix=".geojson", dir=str(out_tif.parent), delete=False) as tf:
        tf_path = Path(tf.name)
        json.dump(gj, tf)

    # Ensure Python VRTs execute inside this process
    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_VRT_ENABLE_PYTHON", "YES")
    gdal.SetConfigOption("GDAL_NUM_THREADS", str(num_threads))
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

    tmp_out = out_tif.with_name(f".tmp_{out_tif.name}")
    try:
        if tmp_out.exists():
            tmp_out.unlink()

        # Multithreading + cutline via WarpOptions
        opts = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=str(tf_path),
            cropToCutline=True,
            dstNodata=dst_nodata,
            resampleAlg=resample,
            multithread=True,                                # enable threaded warper
            warpOptions=[f"NUM_THREADS={num_threads}"],      # control thread count
            creationOptions=list(creation_opts),
        )

        # Run the warp
        ds = gdal.Warp(str(tmp_out), str(src_path), options=opts)
        if ds is None:
            # GetLastErrorMsg() usually has the reason (e.g., VRT pixel fn issues)
            raise RuntimeError(f"gdal.Warp returned None: {gdal.GetLastErrorMsg()}")
        ds = None

        os.replace(tmp_out, out_tif)
        return out_tif

    except Exception as e:
        msg = gdal.GetLastErrorMsg()
        raise RuntimeError(f"gdal.Warp failed: {e}\n{msg}") from None

    finally:
        try: tf_path.unlink()
        except: pass
        try:
            if tmp_out.exists():
                tmp_out.unlink()
        except: pass


# ---------------------------------------------------------------------
# SCENE FILE DISCOVERY
# ---------------------------------------------------------------------

def find_band_jp2s_by_res(output_root: Path, safe_names: Iterable[str],
                          band: str, res: str) -> list[Path]:
    """
    Return all JP2s for e.g. band='B03' at R{res}m inside the .SAFE directories.
    res is a string: "10" | "20" | "60"
    """
    out: list[Path] = []
    subdir = f"R{int(res)}m"
    patt = f"*_{band}_{res}m.jp2"
    for safe in sorted(set(safe_names)):
        safe_dir = Path(output_root) / (safe if safe.endswith(".SAFE")
                                        else f"{safe}.SAFE")
        for jp2 in safe_dir.glob(f"GRANULE/*/IMG_DATA/{subdir}/{patt}"):
            out.append(jp2)
    return out


def corresponding_scl_for_band(band_jp2: Path, band_res: str) -> Path:
    name = band_jp2.name
    # Try same res first
    pat = r'_(?:B\d{2}|B8A)_(?:10|20|60)m\.jp2$'
    same = band_jp2.with_name(re.sub(pat, f'_SCL_{band_res}m.jp2', name))
    if same.exists():
        return same
    # Fallback: most L2A deliveries include SCL at 20 m
    fallback = band_jp2.with_name(re.sub(pat, '_SCL_20m.jp2', name))
    return fallback


# ---------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------

def safe_remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] could not remove {path}: {e}")
