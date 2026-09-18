"""
Synthetic Copula Simulation -- ICON weather cell driven, L5 tile resolved
=========================================================================

Same model as ``synthetic_copula_demo.py``: a ``downscale_multihour`` call over
site positions and all hours, with ``copula_new`` as the backend. The only
difference is where the inputs come from -- the hardcoded lat/lon, cloud speed,
cloud direction and mean CSI are replaced by values read out of the QGIS
GeoPackages and the ICON forecast CSV.

The weather cell (``geo``) is walked one L5 tile at a time. Each L5 tile is
its own original-style ``downscale_multihour`` call: all hours, all PV units
inside that tile, same seed and params as the demo. After that tile is written
the next L5 tile is simulated. Sites inside one L5 tile stay fully correlated
as the model intends; neighbouring L5 tiles are separate draws that share only
the hourly weather of the parent L0 cell.

For a given ICON weather cell (``geo``) the script

1. reads the ``L0`` polygon for that ``geo`` from ``ICON_weather_tiles_D76.gpkg``,
2. collects the ``L5`` sub-tiles belonging to the same ``geo`` (levels 1-4 are
   ignored -- the L5 tiles fully tile their parent L0 cell),
3. selects every PV unit of ``D76 PV.gpkg`` that falls inside those L5 tiles and
   uses their lat/lon as the copula site positions,
4. reads the hourly ``CSI`` (mean_csi), ``wind_speed`` (cloud speed) and
   ``wind_direction`` (cloud direction) for the same ``geo`` from the ICON
   forecast CSV, in ascending time order, and
5. downscales to a 1-minute synthetic CSI series per PV unit.

Positions, hours and the random draw are passed to the backend exactly as they
come out of the data. Nothing is snapped, clipped or reseeded.

Results are written to ``simulation_outputs/geo_<geo>/`` as one CSV per PV
unit, grouped in a folder per L5 tile, alongside an index of the sites, the
hourly weather inputs that drove the run, and a run metadata file.

Cost
----
``copula_new`` factorises the space-time covariance matrix of one call, whose
side is ``n_sites_in_this_L5 * steps_per_hour``. Walking L5 by L5 keeps that
matrix at the size of one mini-tile instead of the whole weather cell. The
script prints the matrix size of each L5 tile before simulating it.

Usage
-----
    py demos\synthetic_copula_sim.py --list                    # show available geos
    py demos\synthetic_copula_sim.py 6329                   # one weather cell
    py demos\synthetic_copula_sim.py 6329 6331 6343           # several cells
    py demos\synthetic_copula_sim.py --all                   # every available cell
    py demos\synthetic_copula_sim.py 6338 --max-sites 60
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd

from solarspatialtools import spatial
from solarspatialtools.synthirrad.copula_new import (
    downscale_multihour,
    DEFAULT_PARAMS,
)

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

TILES_GPKG = os.path.join(BASE_DIR, "ICON_weather_tiles_D76.gpkg")
TILES_LAYER = "icon_weather_tiles"

PV_GPKG = os.path.join(BASE_DIR, "D76 PV.gpkg")
PV_LAYER = "d76_pv"

CSV_NAME = "36_DEXP_FC_2024032912_Bavaria_ens1_20240330_0000_to_20240330_2300.csv"
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, CSV_NAME),
    os.path.join(BASE_DIR, "solarspatialtools", "data", CSV_NAME),
]

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "simulation_outputs")

# PV attributes carried through to the site index
PV_COLUMNS = [
    "EinheitMastrNummer",
    "Gemeinde",
    "EinheitBetriebsstatus",
    "Bruttoleistung",
    "Nettonennleistung",
    "Laengengrad",
    "Breitengrad",
]

OPERATING_STATUS = "In Betrieb"


def _csv_path():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {CSV_NAME} in any of: {CSV_CANDIDATES}")


def _human_bytes(n):
    """Format a byte count, which for the covariance matrix can be enormous."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.1f} {unit}"
        n /= 1024


# ── input loading ───────────────────────────────────────────────────────────

def load_tiles():
    """Read the full ICON tile layer (all levels, ~5.5k polygons)."""
    tiles = gpd.read_file(TILES_GPKG, layer=TILES_LAYER)
    tiles["geo"] = tiles["geo"].astype(int)
    tiles["level"] = tiles["level"].astype(int)
    return tiles


def load_weather_table():
    """Read the hourly ICON forecast CSV."""
    df = pd.read_csv(_csv_path(),
                     usecols=["time", "geo", "CSI", "wind_speed",
                              "wind_direction"])
    df["time"] = pd.to_datetime(df["time"])
    df["geo"] = df["geo"].astype(int)
    return df


def available_geos(tiles, weather):
    """geo ids that have an L0 polygon, at least one L5 tile and CSV weather."""
    l0 = set(tiles.loc[tiles["level"] == 0, "geo"])
    l5 = set(tiles.loc[tiles["level"] == 5, "geo"])
    return sorted(l0 & l5 & set(weather["geo"]))


def select_sites(tiles, geo, only_operating=False, max_sites=None, seed=42):
    """
    Find the PV units inside the L5 sub-tiles of one ICON weather cell.

    Returns
    -------
    sites : pd.DataFrame
        One row per PV unit with ``site_id``, ``tile_id`` (the L5 tile it sits
        in), lat/lon and the carried PV attributes, ordered by L5 tile.
    l0 : GeoDataFrame
        The single-row L0 polygon of the weather cell.
    n_total : int
        Number of PV units found before any ``max_sites`` sampling.
    """
    l0 = tiles[(tiles["level"] == 0) & (tiles["geo"] == geo)]
    if l0.empty:
        raise ValueError(f"No L0 tile found for geo {geo}")
    if len(l0) > 1:
        warnings.warn(f"geo {geo} has {len(l0)} L0 polygons; using their union "
                      "as the parent cell.")

    l5 = tiles[(tiles["level"] == 5) & (tiles["geo"] == geo)]
    if l5.empty:
        raise ValueError(f"No L5 sub-tiles found for geo {geo}")

    # Read only the PV units in the L0 bounding box, then keep the ones that
    # actually fall inside an L5 polygon.
    pv = gpd.read_file(PV_GPKG, layer=PV_LAYER,
                       bbox=tuple(l0.total_bounds),
                       columns=PV_COLUMNS)
    if pv.empty:
        raise ValueError(f"No PV units in the bounding box of geo {geo}")
    if pv.crs != l5.crs:
        pv = pv.to_crs(l5.crs)

    joined = gpd.sjoin(pv, l5[["tile_id", "geometry"]], how="inner",
                       predicate="within")
    if joined.empty:
        raise ValueError(f"No PV units inside the L5 tiles of geo {geo}")

    if only_operating:
        joined = joined[joined["EinheitBetriebsstatus"] == OPERATING_STATUS]
        if joined.empty:
            raise ValueError(
                f"No operating PV units inside the L5 tiles of geo {geo}")

    sites = pd.DataFrame(joined.drop(columns=["geometry", "index_right"],
                                     errors="ignore"))
    sites = sites.rename(columns={"EinheitMastrNummer": "site_id",
                                  "Breitengrad": "lat",
                                  "Laengengrad": "lon"})
    sites = sites.dropna(subset=["lat", "lon"])
    n_total = len(sites)

    if max_sites is not None and n_total > max_sites:
        sites = sites.sample(n=max_sites, random_state=seed)

    sites = sites.sort_values(["tile_id", "site_id"]).reset_index(drop=True)
    return sites, l0, n_total


def select_weather(weather, geo):
    """Hourly drivers for one weather cell, in ascending time order."""
    cell = weather[weather["geo"] == geo].copy()
    if cell.empty:
        raise ValueError(f"No weather data in the CSV for geo {geo}")

    cell = (cell.drop_duplicates(subset="time")
                .sort_values("time")
                .reset_index(drop=True))

    steps = cell["time"].diff().dropna().unique()
    if len(steps) and not np.all(steps == np.timedelta64(1, "h")):
        warnings.warn(f"geo {geo}: hourly time axis is not contiguous "
                      f"({pd.to_timedelta(steps).unique()}); the run treats "
                      "the rows as consecutive hours anyway.")
    return cell


# ── simulation ──────────────────────────────────────────────────────────────

def site_positions(sites, l0):
    """
    East-North positions in metres, referenced to the L0 tile centroid.

    A single reference point per weather cell keeps every site in the same
    local frame, which is what the copula's distance matrix expects.
    """
    lat_ref = float(l0["tile_lat"].iloc[0])
    lon_ref = float(l0["tile_lon"].iloc[0])
    return spatial.lla2flat(sites["lat"].to_numpy(dtype=float),
                            sites["lon"].to_numpy(dtype=float),
                            lat_ref, lon_ref, method="tmerc")


def simulate_l5(tile_sites, l0, cell, seed=42, steps_per_hour=60):
    """
    Run the copula downscaling for every PV unit of one L5 tile.

    This is the same ``downscale_multihour`` call as in
    ``synthetic_copula_demo.py``: all hours of this weather cell, all sites of
    this one L5 tile. The hourly arrays are taken straight from the forecast
    CSV in ascending time order.

    Returns
    -------
    csi : np.ndarray, shape (n_hours * steps_per_hour, n_sites_in_tile)
    times : pd.DatetimeIndex
        The output time axis, one entry per row of ``csi``.
    e_pos, n_pos : np.ndarray
        Site positions in metres relative to the L0 tile centroid.
    """
    e_pos, n_pos = site_positions(tile_sites, l0)

    mean_csi = cell["CSI"].to_numpy(dtype=float)
    cloud_spd = cell["wind_speed"].to_numpy(dtype=float)
    # CSV wind_direction is in degrees, the model wants radians
    cloud_dir = np.deg2rad(cell["wind_direction"].to_numpy(dtype=float))

    n_hours = len(mean_csi)
    freq = pd.Timedelta(hours=1) / steps_per_hour

    # One hour of timestamps; the backend reuses it for each hour and treats
    # the hours as consecutive.
    hour_times = pd.date_range(start=cell["time"].iloc[0],
                               periods=steps_per_hour, freq=freq)

    csi = downscale_multihour(hour_times, e_pos, n_pos, cloud_spd, cloud_dir,
                              mean_csi, DEFAULT_PARAMS, seed=seed,
                              scale=True, noneg=True)

    times = pd.date_range(start=cell["time"].iloc[0],
                          periods=n_hours * steps_per_hour, freq=freq)
    return np.asarray(csi), times, e_pos, n_pos


# ── output writing ──────────────────────────────────────────────────────────

def _write_text(job):
    path, text = job
    with open(path, "w", newline="") as fh:
        fh.write(text)


def write_site_csvs(csi, times, sites, sites_dir, float_format="%.4f",
                    block=512, workers=8):
    """
    Write one CSV per PV unit, grouped into a folder per L4 tile.

    Sites are formatted a block at a time so the intermediate string array
    stays small. The writes themselves are the bottleneck (many small files),
    so they are handed to a thread pool.
    """
    time_col = times.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    header = "time,csi\n"

    for tile_id in sites["tile_id"].unique():
        os.makedirs(os.path.join(sites_dir, str(tile_id)), exist_ok=True)

    tile_ids = sites["tile_id"].to_numpy()
    site_ids = sites["site_id"].to_numpy()
    n_sites = len(sites)
    rel_paths = [f"sites/{tile_ids[i]}/{site_ids[i]}.csv"
                 for i in range(n_sites)]

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, n_sites, block):
            stop = min(start + block, n_sites)
            formatted = np.char.mod(float_format, csi[:, start:stop])
            jobs = []
            for j in range(stop - start):
                rel = rel_paths[start + j]
                text = header + "\n".join(
                    map(",".join, zip(time_col, formatted[:, j]))) + "\n"
                jobs.append((os.path.join(sites_dir, *rel.split("/")[1:]),
                             text))
            if pool is None:
                for job in jobs:
                    _write_text(job)
            else:
                list(pool.map(_write_text, jobs, chunksize=16))
    finally:
        if pool is not None:
            pool.shutdown()

    return rel_paths


def write_combined_csv(csi, times, sites, path, float_format="%.4f"):
    """Write a single wide CSV: rows are timestamps, columns are PV units."""
    frame = pd.DataFrame(csi, index=times, columns=sites["site_id"])
    frame.index.name = "time"
    frame.to_csv(path, float_format=float_format)


# ── per-cell driver ─────────────────────────────────────────────────────────

def run_geo(geo, tiles, weather, out_dir, seed=42, steps_per_hour=60,
            max_sites=None, only_operating=False, per_site=True,
            combined=False, float_format="%.4f", workers=8):
    """Load, simulate and write one ICON weather cell."""
    print(f"\n{'=' * 68}")
    print(f"  ICON weather cell geo = {geo}")
    print(f"{'=' * 68}")

    t_load = time.time()
    sites, l0, n_total = select_sites(tiles, geo, only_operating=only_operating,
                                      max_sites=max_sites, seed=seed)
    cell = select_weather(weather, geo)
    t_load = time.time() - t_load

    n_sites = len(sites)
    n_tiles = sites["tile_id"].nunique()
    n_l5_total = int(((tiles["level"] == 5) & (tiles["geo"] == geo)).sum())
    n_hours = len(cell)

    if n_sites < n_total:
        print(f"  PV units:      {n_sites} (sampled from {n_total})")
    else:
        print(f"  PV units:      {n_sites}")
    print(f"  L5 sub-tiles:  {n_tiles} with PV (of {n_l5_total} in the cell)")
    print(f"  Hourly steps:  {n_hours}  "
          f"({cell['time'].iloc[0]} to {cell['time'].iloc[-1]})")
    print(f"  CSI range:     [{cell['CSI'].min():.3f}, "
          f"{cell['CSI'].max():.3f}]")
    print(f"  Cloud speed:   [{cell['wind_speed'].min():.2f}, "
          f"{cell['wind_speed'].max():.2f}] m/s")
    print(f"  Input load:    {t_load:.1f} s")

    cell_dir = os.path.join(out_dir, f"geo_{geo}")
    sites_dir = os.path.join(cell_dir, "sites")
    os.makedirs(cell_dir, exist_ok=True)
    if per_site:
        os.makedirs(sites_dir, exist_ok=True)

    tile_ids = list(sites["tile_id"].unique())
    n_l5 = len(tile_ids)
    print(f"\n  Walking {n_l5} L5 tiles, one at a time "
          f"({n_hours} h x {steps_per_hour} steps) ...")

    e_all = np.full(n_sites, np.nan)
    n_all = np.full(n_sites, np.nan)
    rel_paths = [None] * n_sites
    combined_parts = [] if combined else None
    times = None
    skipped_tiles = []
    t_sim = t_write = 0.0

    for i, tile_id in enumerate(tile_ids, start=1):
        mask = sites["tile_id"] == tile_id
        tile_sites = sites.loc[mask]
        n_tile = len(tile_sites)
        d = n_tile * steps_per_hour
        print(f"\n  [{i}/{n_l5}] {tile_id}: {n_tile} PV units  "
              f"matrix {d} x {d} ({_human_bytes(d * d * 8.0)} per hour)")

        t0 = time.time()
        try:
            csi, times, e_pos, n_pos = simulate_l5(
                tile_sites, l0, cell, seed=seed,
                steps_per_hour=steps_per_hour)
        except (MemoryError, np.linalg.LinAlgError) as exc:
            print(f"    skipped: {exc}")
            skipped_tiles.append((tile_id, str(exc)))
            continue
        t_sim += time.time() - t0
        print(f"    simulated in {time.time() - t0:.1f} s")

        e_all[mask.to_numpy()] = e_pos
        n_all[mask.to_numpy()] = n_pos

        t0 = time.time()
        if per_site:
            tile_paths = write_site_csvs(
                csi, times, tile_sites, sites_dir,
                float_format=float_format, workers=workers)
            for idx, path in zip(np.flatnonzero(mask.to_numpy()), tile_paths):
                rel_paths[idx] = path
        if combined:
            combined_parts.append((tile_sites["site_id"].to_numpy(), csi))
        t_write += time.time() - t0

    if times is None:
        raise ValueError(
            f"No L4 tile of geo {geo} could be simulated"
            + (f" ({len(skipped_tiles)} skipped)" if skipped_tiles else ""))

    index = sites.copy()
    index.insert(0, "geo", geo)
    index["e_pos_m"] = np.round(e_all, 2)
    index["n_pos_m"] = np.round(n_all, 2)
    if per_site:
        index["output_file"] = rel_paths
    index.to_csv(os.path.join(cell_dir, "site_index.csv"), index=False)

    cell.to_csv(os.path.join(cell_dir, "weather_input.csv"), index=False)

    if combined and combined_parts:
        wide = np.concatenate([part[1] for part in combined_parts], axis=1)
        wide_ids = np.concatenate([part[0] for part in combined_parts])
        write_combined_csv(
            wide, times, pd.DataFrame({"site_id": wide_ids}),
            os.path.join(cell_dir, "all_sites_wide.csv"),
            float_format=float_format)

    metadata = {
        "geo": int(geo),
        "engine_module": "solarspatialtools.synthirrad.copula_new",
        "seed": seed,
        "n_sites": int(n_sites),
        "n_sites_in_cell": int(n_total),
        "n_l5_tiles_with_pv": int(n_tiles),
        "n_hours": int(n_hours),
        "steps_per_hour": int(steps_per_hour),
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "only_operating": bool(only_operating),
        "max_sites": max_sites,
        "l5_tiles_simulated": n_l5 - len(skipped_tiles),
        "l5_tiles_skipped": [
            {"tile_id": tid, "reason": reason}
            for tid, reason in skipped_tiles
        ],
        "reference_lat": float(l0["tile_lat"].iloc[0]),
        "reference_lon": float(l0["tile_lon"].iloc[0]),
        "params": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in DEFAULT_PARAMS.items()},
        "sources": {
            "tiles": os.path.basename(TILES_GPKG),
            "pv": os.path.basename(PV_GPKG),
            "weather": os.path.basename(_csv_path()),
        },
        "timings_s": {"load": round(t_load, 2), "simulate": round(t_sim, 2),
                      "write": round(t_write, 2)},
    }
    with open(os.path.join(cell_dir, "run_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n  Simulation:    {t_sim:.1f} s")
    print(f"  Writing:       {t_write:.1f} s")
    if skipped_tiles:
        print(f"  Skipped L5:    {len(skipped_tiles)} "
              f"({', '.join(tid for tid, _ in skipped_tiles)})")
    print(f"  Output:        {cell_dir}")
    if per_site:
        n_written = sum(1 for p in rel_paths if p is not None)
        print(f"                 {n_written} site CSVs in "
              f"{n_l5 - len(skipped_tiles)} L5 folders")

    return metadata


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Copula-based synthetic irradiance simulation for every PV "
                    "unit inside the L5 sub-tiles of an ICON weather cell.")
    parser.add_argument("geos", nargs="*", type=int,
                        help="ICON weather cell ids (geo). Omit with --all or "
                             "--list.")
    parser.add_argument("--all", action="store_true",
                        help="Run every geo present in both the GeoPackage and "
                             "the forecast CSV.")
    parser.add_argument("--list", action="store_true",
                        help="List the available geo ids and exit.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output root (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--steps-per-hour", type=int, default=60,
                        help="Output steps per hour (default: 60, i.e. 1 min).")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Randomly sample at most this many PV units per "
                             "weather cell before splitting them into L5 "
                             "tiles.")
    parser.add_argument("--only-operating", action="store_true",
                        help=f"Keep only PV units with "
                             f"EinheitBetriebsstatus == '{OPERATING_STATUS}'.")
    parser.add_argument("--no-per-site", action="store_true",
                        help="Skip the per-PV-unit CSVs.")
    parser.add_argument("--combined", action="store_true",
                        help="Also write one wide CSV with all sites as "
                             "columns.")
    parser.add_argument("--float-format", default="%.4f",
                        help="Number format for the CSI values "
                             "(default: %%.4f).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads used to write the per-site CSVs "
                             "(default: 8).")
    args = parser.parse_args()

    print("Loading ICON tiles and forecast weather ...")
    tiles = load_tiles()
    weather = load_weather_table()
    geos = available_geos(tiles, weather)

    if args.list:
        print(f"\n{len(geos)} geo ids with an L0 tile, L5 sub-tiles and "
              f"weather data:")
        print("  " + ", ".join(str(g) for g in geos))
        pv_counts = (tiles[(tiles["level"] == 5) & tiles["geo"].isin(geos)]
                     .groupby("geo")["pv_n"].sum())
        print("\n  geo    L5 tiles   PV units (from tile attributes)")
        for g in geos:
            n_l5 = int(((tiles["level"] == 5) & (tiles["geo"] == g)).sum())
            print(f"  {g:<6} {n_l5:<10} {int(pv_counts.get(g, 0))}")
        return

    targets = geos if args.all else args.geos
    if not targets:
        parser.error("Provide one or more geo ids, or use --all / --list.")

    unknown = [g for g in targets if g not in geos]
    if unknown:
        print(f"\nSkipping unavailable geo ids: {unknown}")
        print("Use --list to see what is available.")
        targets = [g for g in targets if g in geos]
    if not targets:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    summaries, failures = [], []
    t_all = time.time()
    for geo in targets:
        try:
            summaries.append(run_geo(
                geo, tiles, weather, args.out_dir,
                seed=args.seed, steps_per_hour=args.steps_per_hour,
                max_sites=args.max_sites, only_operating=args.only_operating,
                per_site=not args.no_per_site, combined=args.combined,
                float_format=args.float_format, workers=args.workers))
        except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
            print(f"  geo {geo} skipped: {exc}")
            failures.append((geo, str(exc)))

    print(f"\n{'=' * 68}")
    print(f"  Done: {len(summaries)} cell(s), "
          f"{sum(s['n_sites'] for s in summaries)} PV units, "
          f"{time.time() - t_all:.1f} s total")
    if failures:
        print(f"  {len(failures)} cell(s) skipped: "
              f"{', '.join(str(g) for g, _ in failures)}")
    print(f"  Results in {args.out_dir}")


if __name__ == "__main__":
    main()
