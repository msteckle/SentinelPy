#!/usr/bin/env python3
import argparse, os, sys, time
from pathlib import Path
import requests

EGM96_URL = "http://step.esa.int/auxdata/dem/egm96/ww15mgh_b.zip"

def egm96_path(aux_root: Path) -> Path:
    return aux_root / "auxdata" / "dem" / "egm96" / "ww15mgh_b.zip"

def ensure_dirs(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch_egm96(aux_root: Path, overwrite: bool=False) -> str:
    dst = egm96_path(aux_root)
    ensure_dirs(dst)

    # Skip if present and non-empty unless --overwrite
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return f"SKIP: {dst} (exists, {dst.stat().st_size} bytes)"

    # Resume if partial exists
    mode = "ab" if dst.exists() else "wb"
    headers = {}
    if dst.exists():
        headers["Range"] = f"bytes={dst.stat().st_size}-"

    with requests.get(EGM96_URL, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        got0 = dst.stat().st_size if dst.exists() else 0
        t0 = time.time()
        chunk = 1024 * 1024
        with open(dst, mode) as f:
            got = got0
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    f.write(blk)
                    got += len(blk)
                    if time.time() - t0 > 1.5:
                        print(f"… {got/1e6:.1f} MB", flush=True)
                        t0 = time.time()

    return f"OK: {dst} ({dst.stat().st_size} bytes)"

def main():
    ap = argparse.ArgumentParser(description="Manage SNAP auxdata cache (EGM96, etc.)")
    ap.add_argument("--aux-root", type=Path, required=True,
                    help="Persistent auxdata root (will contain auxdata/dem, auxdata/Orbits, …)")
    ap.add_argument("--overwrite", action="store_true", help="Re-download even if present")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-egm96", help="Download EGM96 ww15mgh_b.zip into auxdata/dem/egm96/")

    args = ap.parse_args()
    if args.cmd == "fetch-egm96":
        print(fetch_egm96(args.aux_root, overwrite=args.overwrite))

if __name__ == "__main__":
    main()
