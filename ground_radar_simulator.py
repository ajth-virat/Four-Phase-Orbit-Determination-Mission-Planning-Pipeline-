"""
PROJECT 1: Ground Radar Simulator (v2 — Live TLE from Celestrak)
=================================================================
1. Fetches live TLEs for 150 active satellites from Celestrak (free, no login)
2. Scans all of them — finds which has the best upcoming pass over your location
3. Simulates a ground radar tracking that satellite
4. Injects realistic Gaussian noise into range / azimuth / elevation
5. Saves radar_observations.csv  →  this is Project 2's raw input

Beginner libraries used:
  numpy      — math, noise generation
  matplotlib — plots
  pandas     — CSV output
  sgp4       — TLE propagation
  requests   — live TLE fetch from Celestrak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday

# ─────────────────────────────────────────────
# CONFIGURATION  ← only change these
# ─────────────────────────────────────────────
GS_LAT_DEG         =  13.0827   # Your latitude  (Chennai, India)
GS_LON_DEG         =  80.2707   # Your longitude
GS_ALT_M           =  6.0       # Your altitude in metres

SIM_DURATION_MIN   = 100        # how long to simulate around the best pass
TIMESTEP_SEC       = 10         # radar ping every N seconds
ELEVATION_MASK_DEG = 5.0        # ignore satellite below this elevation
LOOKAHEAD_MIN      = 120        # scan next N minutes for best pass
SCAN_STEP_SEC      = 30         # coarse time step used during satellite scan

NOISE_RANGE_M      = 50.0       # ± 50 m  range noise  (realistic radar)
NOISE_AZ_DEG       = 0.05       # ± 0.05° azimuth noise
NOISE_EL_DEG       = 0.05       # ± 0.05° elevation noise

OUTPUT_CSV         = "radar_observations.csv"
OUTPUT_PNG         = "radar_simulation.png"

# Celestrak free TLE endpoints (no API key needed)
CELESTRAK_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
]

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
RE_M    = 6_378_137.0        # WGS84 Earth radius, metres
FE      = 1 / 298.257223563  # WGS84 flattening

# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """Geodetic (lat, lon, alt) → ECEF XYZ in metres."""
    lat = lat_deg * DEG2RAD
    lon = lon_deg * DEG2RAD
    N   = RE_M / np.sqrt(1 - FE * (2 - FE) * np.sin(lat) ** 2)
    x   = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y   = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z   = (N * (1 - FE) ** 2 + alt_m) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_enu(delta_ecef, lat_deg, lon_deg):
    """ECEF delta vector → local East-North-Up."""
    lat = lat_deg * DEG2RAD
    lon = lon_deg * DEG2RAD
    R = np.array([
        [-np.sin(lon),               np.cos(lon),              0.0        ],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    return R @ delta_ecef


def enu_to_azel(enu):
    """ENU → (azimuth_deg, elevation_deg, range_m)."""
    e, n, u = enu
    rng    = np.linalg.norm(enu)
    el_rad = np.arcsin(u / rng)
    az_rad = np.arctan2(e, n) % (2 * np.pi)
    return az_rad * RAD2DEG, el_rad * RAD2DEG, rng


def ecef_to_latlon(ecef):
    """ECEF → approximate geodetic (lat_deg, lon_deg) for plotting."""
    x, y, z = ecef
    lon = np.arctan2(y, x) * RAD2DEG
    p   = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - FE)) * RAD2DEG
    return lat, lon


def gmst_rad(jd):
    """Greenwich Mean Sidereal Time in radians from Julian Date."""
    T        = (jd - 2_451_545.0) / 36_525.0
    gmst_sec = (67310.54841
                + (876_600 * 3600 + 8_640_184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6   * T ** 3)
    return (gmst_sec % 86_400) / 86_400 * 2 * np.pi


def teme_to_ecef(r_teme_m, jd):
    """Rotate TEME position vector → ECEF using GMST."""
    theta = gmst_rad(jd)
    R = np.array([
        [ np.cos(theta), np.sin(theta), 0.0],
        [-np.sin(theta), np.cos(theta), 0.0],
        [ 0.0,           0.0,           1.0],
    ])
    return R @ r_teme_m


def propagate(sat, t):
    """Propagate a Satrec object to time t → (r_ecef_m, jd_full) or (None, None)."""
    jd, fr = jday(t.year, t.month, t.day,
                  t.hour, t.minute, t.second + t.microsecond / 1e6)
    e, r_km, _ = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    r_ecef = teme_to_ecef(np.array(r_km) * 1000.0, jd + fr)
    return r_ecef, jd + fr

# ─────────────────────────────────────────────
# STEP 1: FETCH LIVE TLEs
# ─────────────────────────────────────────────

def fetch_tles(max_sats=150):
    print("\n[1/3] Fetching live TLEs from Celestrak...")
    satellites = []
    for url in CELESTRAK_URLS:
        try:
            resp  = requests.get(url, timeout=15)
            resp.raise_for_status()
            lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
            for i in range(0, len(lines) - 2, 3):
                name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
                if line1.startswith("1") and line2.startswith("2"):
                    try:
                        sat = Satrec.twoline2rv(line1, line2)
                        satellites.append({"name": name, "sat": sat})
                    except Exception:
                        pass
            group = url.split("GROUP=")[1].split("&")[0]
            print(f"   ✓ {group:12s} → {len(satellites)} satellites loaded so far")
            if len(satellites) >= max_sats:
                break
        except Exception as ex:
            print(f"   ✗ Failed ({url}): {ex}")

    satellites = satellites[:max_sats]
    print(f"   Using {len(satellites)} satellites for pass prediction.")
    return satellites

# ─────────────────────────────────────────────
# STEP 2: FIND BEST PASS
# ─────────────────────────────────────────────

def find_best_satellite(satellites, gs_ecef, t_now):
    print(f"\n[2/3] Scanning {len(satellites)} sats — finding best pass over your location...")
    best_entry  = None
    best_el     = -999.0
    best_t_peak = None

    n_steps = int(LOOKAHEAD_MIN * 60 / SCAN_STEP_SEC)

    for entry in satellites:
        sat       = entry["sat"]
        peak_el   = -999.0
        peak_time = None
        for i in range(n_steps):
            t      = t_now + timedelta(seconds=i * SCAN_STEP_SEC)
            r_ecef, _ = propagate(sat, t)
            if r_ecef is None:
                continue
            enu         = ecef_to_enu(r_ecef - gs_ecef, GS_LAT_DEG, GS_LON_DEG)
            _, el, _    = enu_to_azel(enu)
            if el > peak_el:
                peak_el   = el
                peak_time = t
        if peak_el > best_el:
            best_el     = peak_el
            best_entry  = entry
            best_t_peak = peak_time

    print(f"   ✓ Winner     : {best_entry['name']}")
    print(f"   ✓ Peak elev  : {best_el:.1f}°")
    print(f"   ✓ Peak time  : {best_t_peak.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    return best_entry, best_t_peak, best_el

# ─────────────────────────────────────────────
# STEP 3: SIMULATE RADAR
# ─────────────────────────────────────────────

def simulate_radar(best_entry, t_peak, gs_ecef):
    print(f"\n[3/3] Simulating radar tracking of {best_entry['name']}...")
    sat     = best_entry["sat"]
    t_start = t_peak - timedelta(minutes=45)   # start 45 min before peak
    n_steps = int(SIM_DURATION_MIN * 60 / TIMESTEP_SEC)

    records                                   = []
    sat_lats, sat_lons                        = [], []
    true_rng, true_el_list, true_az_list      = [], [], []
    noisy_rng, noisy_el_list, noisy_az_list   = [], [], []

    np.random.seed(42)

    for i in range(n_steps):
        t      = t_start + timedelta(seconds=i * TIMESTEP_SEC)
        r_ecef, _ = propagate(sat, t)
        if r_ecef is None:
            continue

        s_lat, s_lon = ecef_to_latlon(r_ecef)
        sat_lats.append(s_lat)
        sat_lons.append(s_lon)

        enu          = ecef_to_enu(r_ecef - gs_ecef, GS_LAT_DEG, GS_LON_DEG)
        az_t, el_t, rng_t = enu_to_azel(enu)

        if el_t < ELEVATION_MASK_DEG:
            continue

        # Inject Gaussian noise
        az_n  = az_t  + np.random.normal(0, NOISE_AZ_DEG)
        el_n  = el_t  + np.random.normal(0, NOISE_EL_DEG)
        rng_n = rng_t + np.random.normal(0, NOISE_RANGE_M)

        true_az_list.append(az_t);   true_el_list.append(el_t);   true_rng.append(rng_t)
        noisy_az_list.append(az_n);  noisy_el_list.append(el_n);  noisy_rng.append(rng_n)

        records.append({
            "timestamp_utc"      : t.isoformat(),
            "elapsed_sec"        : i * TIMESTEP_SEC,
            "azimuth_deg"        : round(az_n,  4),
            "elevation_deg"      : round(el_n,  4),
            "range_m"            : round(rng_n, 2),
            "true_azimuth_deg"   : round(az_t,  4),
            "true_elevation_deg" : round(el_t,  4),
            "true_range_m"       : round(rng_t, 2),
            "sat_lat_deg"        : round(s_lat, 4),
            "sat_lon_deg"        : round(s_lon, 4),
        })

    print(f"   Visible observations : {len(records)}")
    return records, sat_lats, sat_lons, true_rng, true_el_list, noisy_rng, noisy_el_list

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def make_plots(records, sat_lats, sat_lons, true_rng, true_el,
               noisy_rng, noisy_el, sat_name):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    lkw = dict(color="#cccccc", fontsize=9)
    tkw = dict(colors="#aaaaaa")

    # ── Ground track ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0d1117")
    ax1.scatter(sat_lons, sat_lats, s=1, c="#00aaff", alpha=0.3, label="Full orbit")
    ax1.scatter([r["sat_lon_deg"] for r in records],
                [r["sat_lat_deg"] for r in records],
                s=8, c="#ff6600", zorder=3, label="Visible to GS")
    ax1.scatter(GS_LON_DEG, GS_LAT_DEG, s=150, c="#ff0055",
                marker="^", zorder=5, label="Ground Station (Chennai)")
    ax1.set_xlim(-180, 180); ax1.set_ylim(-90, 90)
    ax1.set_xlabel("Longitude (°)", **lkw); ax1.set_ylabel("Latitude (°)", **lkw)
    ax1.set_title(f"Ground Track — {sat_name}", color="#ffffff", fontsize=11)
    ax1.tick_params(**tkw)
    ax1.grid(True, color="#222233", linewidth=0.5)
    ax1.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc", loc="lower left")

    # ── Range ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0d1117")
    ax2.plot(true_rng,  color="#00aaff", lw=1.5, label="True range")
    ax2.plot(noisy_rng, color="#ff6600", lw=0.8, alpha=0.7, label="Noisy range")
    ax2.set_xlabel("Observation #", **lkw); ax2.set_ylabel("Range (m)", **lkw)
    ax2.set_title("Range: True vs Noisy", color="#ffffff", fontsize=10)
    ax2.tick_params(**tkw); ax2.grid(True, color="#222233", linewidth=0.5)
    ax2.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc")

    # ── Elevation ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0d1117")
    ax3.plot(true_el,  color="#00ff88", lw=1.5, label="True elevation")
    ax3.plot(noisy_el, color="#ffcc00", lw=0.8, alpha=0.7, label="Noisy elevation")
    ax3.axhline(ELEVATION_MASK_DEG, color="#ff0055", lw=0.8, ls="--",
                label=f"Mask {ELEVATION_MASK_DEG}°")
    ax3.set_xlabel("Observation #", **lkw); ax3.set_ylabel("Elevation (°)", **lkw)
    ax3.set_title("Elevation: True vs Noisy", color="#ffffff", fontsize=10)
    ax3.tick_params(**tkw); ax3.grid(True, color="#222233", linewidth=0.5)
    ax3.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc")

    plt.suptitle("Project 1 — Ground Radar Simulator (Live TLE)",
                 color="#ffffff", fontsize=13, y=1.01)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"   Plot saved : {OUTPUT_PNG}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  GROUND RADAR SIMULATOR v2 — Live TLE Edition")
    print("=" * 55)
    print(f"  Station : Chennai ({GS_LAT_DEG}°N, {GS_LON_DEG}°E, {GS_ALT_M}m)")

    gs_ecef    = geodetic_to_ecef(GS_LAT_DEG, GS_LON_DEG, GS_ALT_M)
    t_now      = datetime.now(timezone.utc)

    satellites = fetch_tles(max_sats=150)
    if not satellites:
        print("\n[!] No TLEs fetched. Check internet connection.")
        return

    best_entry, t_peak, peak_el = find_best_satellite(satellites, gs_ecef, t_now)

    records, sat_lats, sat_lons, true_rng, true_el, noisy_rng, noisy_el = \
        simulate_radar(best_entry, t_peak, gs_ecef)

    if not records:
        print("\n[!] Zero visible observations generated.")
        print("    Try: increase LOOKAHEAD_MIN, or lower ELEVATION_MASK_DEG to 0.")
        return

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"   CSV saved  : {OUTPUT_CSV}  ({len(df)} rows)")

    make_plots(records, sat_lats, sat_lons, true_rng, true_el,
               noisy_rng, noisy_el, best_entry["name"])

    print("\n" + "=" * 55)
    print("  ✓  Project 1 complete.")
    print(f"  Feed '{OUTPUT_CSV}' into Project 2 (Kalman Filter).")
    print("=" * 55)


if __name__ == "__main__":
    main()
