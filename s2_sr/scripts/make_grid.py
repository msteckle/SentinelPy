#!/usr/bin/env python3
# python make_grid.py --out "path/to/csv.csv"

from pathlib import Path
import argparse

def fmt_slug(xmin, ymin):
    """
    Build a human-readable ID from the SW corner (xmin,ymin) at 0.25° grid.
    Example: xmin=-149.75, ymin=68.50 -> 'Q025_W149_75_N68_50'
    """
    lon_dir = "W" if xmin < 0 else "E"
    lat_dir = "S" if ymin < 0 else "N"
    lon_abs = abs(xmin)
    lat_abs = abs(ymin)
    lon_deg = int(lon_abs)  # 0..179
    lat_deg = int(lat_abs)  # 0..89 or 90 at edge (won't occur for ymin)
    lon_frac = int(round((lon_abs - lon_deg) * 100))  # 0,25,50,75
    lat_frac = int(round((lat_abs - lat_deg) * 100))  # 0,25,50,75
    return f"Q025_{lon_dir}{lon_deg:03d}_{lon_frac:02d}_{lat_dir}{lat_deg:02d}_{lat_frac:02d}"

def main():
    p = argparse.ArgumentParser(
        description="Write a 0.25° pan-Arctic grid CSV (EPSG:4326) with row/col and lat/lon IDs."
    )
    p.add_argument("-o", "--out", type=Path, default=Path("pan_arctic_q025_grid.csv"),
                   help="Output CSV path")
    p.add_argument("--lon-min", type=float, default=-180.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    p.add_argument("--lat-min", type=float, default=60.0)
    p.add_argument("--lat-max", type=float, default=90.0)
    p.add_argument("--cell", type=float, default=0.25, help="Cell size in degrees (default 0.25)")
    args = p.parse_args()

    # Build exact quarter-degree edges using integer math (avoid floating point drift)
    if abs(args.cell - 0.25) < 1e-12:
        den = 4  # 1 / 0.25
        lon_min_i = int(round(args.lon_min * den))
        lon_max_i = int(round(args.lon_max * den))
        lat_min_i = int(round(args.lat_min * den))
        lat_max_i = int(round(args.lat_max * den))

        ncols = lon_max_i - lon_min_i
        nrows = lat_max_i - lat_min_i
        assert ncols > 0 and nrows > 0, "Non-positive grid size."

        lon_edges = [ (lon_min_i + i) / den for i in range(ncols + 1) ]
        # lat edges from north (max) down to south (min)
        lat_edges = [ (lat_max_i - i) / den for i in range(nrows + 1) ]
    else:
        # Generic fallback (may have tiny flating point noise)
        ncols = int(round((args.lon_max - args.lon_min) / args.cell))
        nrows = int(round((args.lat_max - args.lat_min) / args.cell))
        lon_edges = [args.lon_min + i * args.cell for i in range(ncols + 1)]
        lat_edges = [args.lat_max - i * args.cell for i in range(nrows + 1)]

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("tile_id_rc,tile_id_geo,xmin,ymin,xmax,ymax,row,col\n")
        # row 0 = northernmost band; col 0 = westernmost column
        for r in range(len(lat_edges) - 1):
            y_top = lat_edges[r]
            y_bot = lat_edges[r + 1]
            for c in range(len(lon_edges) - 1):
                x_left = lon_edges[c]
                x_right = lon_edges[c + 1]
                tile_id_rc = f"Q025_R{r:03d}_C{c:04d}"
                tile_id_geo = fmt_slug(x_left, y_bot)  # SW corner
                f.write(f"{tile_id_rc},{tile_id_geo},{x_left:.6f},{y_bot:.6f},{x_right:.6f},{y_top:.6f},{r},{c}\n")

    print(f"Wrote {(len(lat_edges)-1)*(len(lon_edges)-1)} tiles to {args.out}")

if __name__ == "__main__":
    main()
