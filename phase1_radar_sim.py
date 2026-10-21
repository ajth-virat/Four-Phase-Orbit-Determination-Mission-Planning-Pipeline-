"""
PHASE 1 - Ground radar simulator (multi-satellite test bench)
=============================================================
WHAT IT IS (say this in the interview)
    A stand-in for a real tracking radar.  We do not own one, so we synthesise
    range / azimuth / elevation measurements of real satellites (orbits from public
    TLEs, propagated with SGP4) and add Gaussian noise at a known level.  Because we
    also keep the noiseless truth, Phase 2 can be graded honestly.

WHAT CHANGED vs the old version
    * Measurements and truth are written to TWO different files.  Phase 2 only ever
      sees radar_observations.csv, so it physically cannot cheat with true_* columns.
    * A TEST SET of satellites across LEO / MEO / GEO is generated, not just one.
    * Pass choice: highest-peak-elevation pass in the next 24 h, as one contiguous arc
      (the old code mixed several passes and used a 120-minute lookahead).
    * Works offline: live Celestrak -> local cache -> clearly labelled SYNTHETIC orbits.
    * Optional stress knobs (--range-bias-m, --noise-scale) to test a filter that is
      NOT perfectly tuned to the sensor.

OUTPUTS (in --out-dir, default ./data)
    radar_observations.csv  noisy measurements only  (input to Phase 2)
    truth_states.csv        SGP4 truth in ECI        (used ONLY for scoring)
    test_set.csv            which satellites, class, peak elevation, data source
    radar_simulation.png    overview plot

Run:  python phase1_radar_sim.py            (live TLE if reachable)
      python phase1_radar_sim.py --offline  (synthetic satellites, no network)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72, jday

import common as C

# ------------------------------------------------------------------ config
OBS_CADENCE_S = {"LEO": 10.0, "MEO": 30.0, "GEO": 30.0}   # radar ping interval
MAX_ARC_S = {"LEO": 1800.0, "MEO": 5400.0, "GEO": 5400.0}  # cap on one tracking arc
QUOTA = {"LEO": 5, "MEO": 3, "GEO": 2}                    # test-set size per class
SEARCH_HOURS = 24.0
SCAN_STEP_S = 30.0

CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"
LIVE_GROUPS = ["stations", "geodetic", "resource", "weather", "gps-ops", "geo"]
CACHE_FILE = "tle_cache.txt"


# ------------------------------------------------------------------ TLE I/O
def parse_tle_text(text: str):
    """Parse 3-line TLE text -> list of dicts.  Skips malformed entries."""
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                sat = Satrec.twoline2rv(l1, l2)
                out.append(dict(name=name, sat=sat, line1=l1, line2=l2,
                                norad=int(l1[2:7]), source="tle"))
            except Exception:
                pass
            i += 3
        else:
            i += 1
    return out


def fetch_live_tles(cache_path: str):
    """Download TLE groups from Celestrak.  Returns list or [] on total failure."""
    try:
        import requests
    except ImportError:
        print("   requests not installed - skipping live fetch")
        return []
    chunks, got = [], 0
    for g in LIVE_GROUPS:
        try:
            r = requests.get(CELESTRAK.format(g=g), timeout=15)
            r.raise_for_status()
            sats = parse_tle_text(r.text)
            if sats:
                chunks.append(r.text.strip())
                got += len(sats)
                print(f"   ok   {g:10s} {len(sats):4d} satellites")
        except Exception as ex:
            print(f"   FAIL {g:10s} {type(ex).__name__}: {str(ex)[:60]}")
    if not chunks:
        return []
    with open(cache_path, "w") as f:
        f.write("\n".join(chunks) + "\n")
    return parse_tle_text("\n".join(chunks))


def load_cache(cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return parse_tle_text(f.read())
    return []


# ------------------------------------------------------------------ synthetic satellites
def make_synthetic(name, alt_km, ecc, inc_deg, raan_deg, argp_deg, m_deg, epoch_jd, satnum):
    """Build a Satrec from classical elements (only used offline; labelled SYNTHETIC)."""
    mu72, re72 = 398600.8, 6378.135
    a = re72 + alt_km
    n_rad_min = np.sqrt(mu72 / a ** 3) * 60.0
    sat = Satrec()
    sat.sgp4init(WGS72, "i", satnum, epoch_jd - 2433281.5, 0.0, 0.0, 0.0,
                 ecc, argp_deg * C.DEG2RAD, inc_deg * C.DEG2RAD,
                 m_deg * C.DEG2RAD, n_rad_min, raan_deg * C.DEG2RAD)
    return dict(name=name, sat=sat, line1="", line2="", norad=satnum, source="SYNTHETIC")


def synthetic_catalog(t0: datetime):
    """A mixed LEO/MEO/GEO set.  Orbits are typical, NOT real catalog entries."""
    jd, fr = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    ejd = jd + fr
    theta0 = float(C.gmst_rad(ejd))

    def geo(name, lon_deg, satnum):
        # equatorial circular orbit with ECI longitude = GMST + station longitude
        m = np.degrees((theta0 + lon_deg * C.DEG2RAD) % (2 * np.pi))
        return make_synthetic(name, 35786.0, 0.0002, 0.05, 0.0, 0.0, m, ejd, satnum)

    cat = [
        make_synthetic("SYN-LEO-ISSLIKE", 420, 0.0006, 51.64, 120, 60, 0, ejd, 90001),
        make_synthetic("SYN-LEO-SSO-700", 700, 0.0010, 98.20, 200, 90, 40, ejd, 90002),
        make_synthetic("SYN-LEO-AJISAI-LIKE", 1490, 0.0010, 50.00, 80, 30, 100, ejd, 90003),
        make_synthetic("SYN-LEO-1200-87", 1200, 0.0008, 87.00, 300, 10, 200, ejd, 90004),
        make_synthetic("SYN-LEO-550-53", 550, 0.0002, 53.00, 15, 0, 300, ejd, 90005),
        make_synthetic("SYN-MEO-GPSLIKE-A", 20180, 0.0100, 55.0, 20, 0, 10, ejd, 90011),
        make_synthetic("SYN-MEO-GPSLIKE-B", 20180, 0.0100, 55.0, 140, 0, 130, ejd, 90012),
        make_synthetic("SYN-MEO-GPSLIKE-C", 20180, 0.0100, 55.0, 260, 0, 250, ejd, 90013),
        geo("SYN-GEO-74E", 74.0, 90021),
        geo("SYN-GEO-93E", 93.0, 90022),
    ]
    return cat


# ------------------------------------------------------------------ SGP4 helpers
def sgp4_eci(sat, jd_base: float, fr_base: float, offsets_s: np.ndarray):
    """Vectorised SGP4 at jd_base + fr_base + offsets.  Returns r (m), v (m/s), jd_full, ok."""
    fr = fr_base + offsets_s / 86400.0
    jd = np.full_like(fr, jd_base)
    e, r, v = sat.sgp4_array(jd, fr)
    ok = (e == 0)
    return r * 1000.0, v * 1000.0, jd + fr, ok


def elevation_series(r_eci, jd_full, st: C.Station):
    """Elevation (deg) of ECI positions as seen from the station; vectorised."""
    g = C.gmst_rad(jd_full)
    c, s = np.cos(g), np.sin(g)
    x, y, z = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
    ecef = np.stack([c * x + s * y, -s * x + c * y, z], axis=1)
    enu = (st.R_enu @ (ecef - st.ecef).T).T
    rng = np.linalg.norm(enu, axis=1)
    return np.degrees(np.arcsin(np.clip(enu[:, 2] / rng, -1, 1)))


def best_pass(entry, t0: datetime, st: C.Station):
    """Find the visible arc with the highest peak elevation in the next 24 h.

    Returns dict(start_off_s, end_off_s, peak_el_deg, peak_off_s) or None.
    """
    jd, fr = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    off = np.arange(0.0, SEARCH_HOURS * 3600.0, SCAN_STEP_S)
    r, _, jdf, ok = sgp4_eci(entry["sat"], jd, fr, off)
    if not ok.any():
        return None
    el = elevation_series(r, jdf, st)
    vis = (el >= C.ELEVATION_MASK_DEG) & ok
    if not vis.any():
        return None
    edges = np.diff(np.concatenate([[0], vis.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    best = None
    for s, e in zip(starts, ends):
        pk = el[s:e + 1].max()
        if best is None or pk > best["peak_el_deg"]:
            best = dict(start_off_s=off[s], end_off_s=off[e],
                        peak_el_deg=float(pk),
                        peak_off_s=float(off[s + int(np.argmax(el[s:e + 1]))]))
    return best


# ------------------------------------------------------------------ test-set choice
def entry_class(entry):
    sat = entry["sat"]
    period_min = 2 * np.pi / sat.no_kozai
    return C.classify_orbit(period_min, sat.ecco, np.degrees(sat.inclo))


def choose_test_set(catalog, t0, st: C.Station, quota=QUOTA):
    """Deterministic rule: per orbit class, take the passes with the highest peak elevation."""
    by_class = {k: [] for k in quota}
    for entry in catalog:
        cls = entry_class(entry)
        if cls not in quota:
            continue
        p = best_pass(entry, t0, st)
        if p is None:
            continue
        by_class[cls].append((p["peak_el_deg"], entry, p))
    chosen = []
    for cls, lst in by_class.items():
        lst.sort(key=lambda x: -x[0])
        for pk, entry, p in lst[:quota[cls]]:
            chosen.append(dict(entry=entry, cls=cls, pass_info=p))
    return chosen


# ------------------------------------------------------------------ measurement simulation
def simulate_arc(item, t0: datetime, st: C.Station, rng_gen, noise_scale=1.0, range_bias_m=0.0):
    entry, cls, p = item["entry"], item["cls"], item["pass_info"]
    step = OBS_CADENCE_S[cls]
    start = p["start_off_s"]
    # keep the arc centred on the pass peak if the pass is longer than the cap
    span = min(p["end_off_s"] - p["start_off_s"], MAX_ARC_S[cls])
    if p["end_off_s"] - p["start_off_s"] > span:
        start = max(p["start_off_s"], p["peak_off_s"] - span / 2)
    offs = np.arange(start, start + span + 1e-6, step)

    jd, fr = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    r, v, jdf, ok = sgp4_eci(entry["sat"], jd, fr, offs)
    el = elevation_series(r, jdf, st)
    keep = ok & (el >= C.ELEVATION_MASK_DEG)
    offs, r, v, jdf = offs[keep], r[keep], v[keep], jdf[keep]
    if len(offs) < 12:
        return None

    rows_obs, rows_truth = [], []
    for k in range(len(offs)):
        rng_t, az_t, el_t = st.rae_from_eci(r[k], jdf[k])
        rng_m = rng_t + range_bias_m + rng_gen.normal(0, C.SIGMA_RANGE_M * noise_scale)
        az_m = az_t + rng_gen.normal(0, C.SIGMA_AZ_DEG * noise_scale * C.DEG2RAD)
        el_m = el_t + rng_gen.normal(0, C.SIGMA_EL_DEG * noise_scale * C.DEG2RAD)
        ts = (t0 + timedelta(seconds=float(offs[k]))).isoformat()
        base = dict(sat_name=entry["name"], norad=entry["norad"], orbit_class=cls,
                    timestamp_utc=ts, jd_utc=float(jdf[k]),
                    elapsed_sec=float(offs[k] - offs[0]))
        rows_obs.append({**base, "range_m": rng_m, "azimuth_deg": az_m * C.RAD2DEG % 360.0,
                         "elevation_deg": el_m * C.RAD2DEG})
        rows_truth.append({**base, "x_m": r[k, 0], "y_m": r[k, 1], "z_m": r[k, 2],
                           "vx_ms": v[k, 0], "vy_ms": v[k, 1], "vz_ms": v[k, 2],
                           "true_range_m": rng_t, "true_az_deg": az_t * C.RAD2DEG,
                           "true_el_deg": el_t * C.RAD2DEG})
    return rows_obs, rows_truth


# ------------------------------------------------------------------ plots
def make_plot(obs_df, truth_df, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, g) in enumerate(truth_df.groupby("sat_name", sort=False)):
        r = g[["x_m", "y_m", "z_m"]].values
        gm = C.gmst_rad(g["jd_utc"].values)
        lon = np.degrees(np.arctan2(-np.sin(gm) * r[:, 0] + np.cos(gm) * r[:, 1],
                                    np.cos(gm) * r[:, 0] + np.sin(gm) * r[:, 1]))
        lat = np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1)))
        axes[0].plot(lon, lat, ".", ms=3, color=cmap(i % 10), label=name)
        axes[1].plot(g["elapsed_sec"] / 60, g["true_el_deg"], color=cmap(i % 10), label=name)
    axes[0].plot(C.GS_LON_DEG, C.GS_LAT_DEG, "r^", ms=10, label="Ground station")
    axes[0].set_xlabel("Longitude (deg)"); axes[0].set_ylabel("Latitude (deg)")
    axes[0].set_title("Sub-satellite points during tracking arcs"); axes[0].grid(alpha=.3)
    axes[1].axhline(C.ELEVATION_MASK_DEG, color="k", ls="--", lw=.8, label="mask")
    axes[1].set_xlabel("Minutes since arc start"); axes[1].set_ylabel("True elevation (deg)")
    axes[1].set_title("Tracking arcs (elevation)"); axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Phase 1 - radar simulation test set (truth shown; Phase 2 never sees it)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ main
def run(out_dir="data", offline=False, seed=42, start=None, noise_scale=1.0,
        range_bias_m=0.0, show=False, quota=None):
    os.makedirs(out_dir, exist_ok=True)
    t0 = (datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start
          else datetime.now(timezone.utc).replace(microsecond=0))
    st = C.Station()
    cache_path = os.path.join(out_dir, CACHE_FILE)

    print("=" * 60)
    print("PHASE 1  Ground radar simulator")
    print("=" * 60)
    print(f"Station {C.GS_LAT_DEG}N {C.GS_LON_DEG}E | search window starts {t0.isoformat()}")

    catalog, source = [], ""
    if not offline:
        print("Fetching live TLEs from Celestrak ...")
        catalog = fetch_live_tles(cache_path)
        source = "live Celestrak"
        if not catalog:
            catalog = load_cache(cache_path)
            source = "local TLE cache (may be stale)"
    if not catalog:
        catalog = synthetic_catalog(t0)
        source = "SYNTHETIC orbits (offline demo, not real satellites)"
    print(f"Catalog source: {source}  ({len(catalog)} satellites)")

    chosen = choose_test_set(catalog, t0, st, quota or QUOTA)
    if not chosen:
        raise RuntimeError("No satellite is visible from the station in the search window.")
    print(f"Test set: {len(chosen)} satellites with a visible pass")

    obs_rows, truth_rows, meta = [], [], []
    for i, item in enumerate(chosen):
        gen = np.random.default_rng(seed + i)
        res = simulate_arc(item, t0, st, gen, noise_scale, range_bias_m)
        if res is None:
            continue
        o, t = res
        obs_rows += o
        truth_rows += t
        e = item["entry"]
        meta.append(dict(sat_name=e["name"], norad=e["norad"], orbit_class=item["cls"],
                         peak_el_deg=round(item["pass_info"]["peak_el_deg"], 2),
                         n_obs=len(o), arc_s=o[-1]["elapsed_sec"], source=e["source"],
                         tle_line1=e["line1"], tle_line2=e["line2"]))
        print(f"  {e['name'][:24]:24s} {item['cls']}  peak {item['pass_info']['peak_el_deg']:5.1f} deg"
              f"  {len(o):4d} obs over {o[-1]['elapsed_sec']/60:5.1f} min")

    obs_df, truth_df, meta_df = map(pd.DataFrame, (obs_rows, truth_rows, meta))
    obs_df.to_csv(os.path.join(out_dir, "radar_observations.csv"), index=False)
    truth_df.to_csv(os.path.join(out_dir, "truth_states.csv"), index=False)
    meta_df.to_csv(os.path.join(out_dir, "test_set.csv"), index=False)
    make_plot(obs_df, truth_df, os.path.join(out_dir, "radar_simulation.png"), show)
    print(f"Wrote radar_observations.csv ({len(obs_df)} rows, NO truth columns), "
          f"truth_states.csv, test_set.csv -> {out_dir}/")
    return obs_df, truth_df, meta_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1 radar simulator")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--offline", action="store_true", help="skip network, use synthetic satellites")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default=None, help="UTC ISO start time (default: now)")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiply true noise (stress test)")
    ap.add_argument("--range-bias-m", type=float, default=0.0, help="unmodelled range bias (stress test)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.out_dir, a.offline, a.seed, a.start, a.noise_scale, a.range_bias_m, a.show)
