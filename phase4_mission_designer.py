"""
PHASE 4 - Earth -> Mars mission designer (departs from the Phase 3 parking orbit)
=================================================================================
INPUT   data/parking_orbit.json     radius + inclination of the orbit Phases 1-3 produced
OUTPUT  data/mission_summary.json   headline numbers
        data/mission_trajectory.csv the transfer arc (Sun-centred, ecliptic) sampled in time
        data/porkchop.csv           C3 and delta-v over the whole (departure, TOF) grid
        data/mission_plots.png

WHAT CHANGED vs the old version
    * Departure burn is computed from the radius of the orbit PHASE 3 produced, not a
      generic 200 km orbit.  This is the real link between the four phases.
    * Correct ephemeris (ephemeris.py) and a verified Lambert solver (lambert.py).
    * Self-check: the Lambert answer is propagated forward with a numerical two-body
      integrator and must land on Mars; the miss distance is printed.  (The old arc
      visibly stopped short of Mars; the old miss was ~272 million km.)
    * Optimiser: scipy differential evolution (global) + Nelder-Mead polish, compared with
      the brute-force grid minimum.
    * Removed: the JAX "sensitivity" (it differentiated a Hohmann proxy, not the real
      C3), the CasADi stub (it failed and still printed "complete"), and the pygmo
      dependency.  Sensitivity is now a finite difference of the REAL C3 function, and
      the launch-window width is read off the porkchop.

THE MODEL (patched conics, impulsive burns - say this up front)
    heliocentric leg  : Lambert arc between Earth at departure and Mars at arrival
    departure         : v_inf = v_transfer - v_Earth;  C3 = |v_inf|^2
                        burn from circular parking orbit radius r_p:
                        dv = sqrt(v_inf^2 + 2 mu/r_p) - sqrt(mu/r_p)
    arrival           : capture burn into a circular Mars orbit (default 400 km alt);
                        this is an UPPER bound (an elliptical capture orbit is cheaper)
    ignored           : gravity losses, finite burn time, plane-change to match the
                        launch asymptote, mid-course corrections, Mars/Earth ephemeris error

Run:  python phase4_mission_designer.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

import common as C
from ephemeris import AU, MU_SUN, Ephemeris, R_EQ2ECL
from lambert import lambert_universal

MU_EARTH = C.MU_EARTH
MU_MARS = 4.282837e13
R_MARS = 3389.5e3
MARS_CAPTURE_ALT_M = 400e3


def jd_to_date(jd):
    from astropy.time import Time
    return Time(jd, format="jd", scale="utc").iso[:10]


def date_to_jd(s):
    from astropy.time import Time
    return float(Time(s, scale="utc").jd)


# ---------------------------------------------------------------- core transfer evaluation
def transfer(eph: Ephemeris, jd_dep, tof_days, r_park, r_capture):
    """Everything about one departure/TOF pair.  Returns dict or None if Lambert fails."""
    r1, ve = eph.state("earth", jd_dep)
    r2, vm = eph.state("mars", jd_dep + tof_days)
    try:
        v1, v2 = lambert_universal(r1, r2, tof_days * 86400.0, MU_SUN, prograde=True)
    except ValueError:
        return None
    vinf_d, vinf_a = v1 - ve, v2 - vm
    c3 = float(vinf_d @ vinf_d)
    va2 = float(vinf_a @ vinf_a)
    dv_dep = np.sqrt(c3 + 2 * MU_EARTH / r_park) - np.sqrt(MU_EARTH / r_park)
    dv_cap = np.sqrt(va2 + 2 * MU_MARS / r_capture) - np.sqrt(MU_MARS / r_capture)
    return dict(c3=c3 / 1e6, vinf_dep=np.sqrt(c3) / 1e3, vinf_arr=np.sqrt(va2) / 1e3,
                dv_dep=dv_dep / 1e3, dv_cap=dv_cap / 1e3, dv_total=(dv_dep + dv_cap) / 1e3,
                r1=r1, v1=v1, r2=r2, v2=v2, vinf_dep_vec=vinf_d, ve=ve, vm=vm)


def porkchop(eph, jd0, span, tofs, n_dep, r_park, r_cap):
    deps = np.linspace(jd0, jd0 + span, n_dep)
    C3 = np.full((len(tofs), n_dep), np.nan)
    DV = np.full_like(C3, np.nan)
    for j, jd in enumerate(deps):
        for i, tof in enumerate(tofs):
            t = transfer(eph, jd, tof, r_park, r_cap)
            if t is not None:
                C3[i, j], DV[i, j] = t["c3"], t["dv_total"]
    return deps, C3, DV


def optimise(eph, bounds, r_park, r_cap, seed=3):
    """Global DE then local polish on total delta-v (km/s)."""
    def obj(x):
        t = transfer(eph, x[0], x[1], r_park, r_cap)
        return 1e3 if t is None else t["dv_total"]
    de = differential_evolution(obj, bounds, seed=seed, popsize=24, maxiter=80, tol=1e-9,
                                polish=False, updating="deferred", workers=1)
    nm = minimize(obj, de.x, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10))
    return nm.x, nm.fun, de.nfev + nm.nfev


# ---------------------------------------------------------------- verification
def verify_by_propagation(t):
    """Integrate the heliocentric two-body ODE from Earth with the Lambert v1 and
    return the distance (km) between the arrival point and where Mars is."""
    def rhs(_, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_SUN * r / np.linalg.norm(r) ** 3])
    return rhs


def miss_distance_km(eph, t, tof_days, jd_dep):
    rhs = verify_by_propagation(t)
    sol = solve_ivp(rhs, (0, tof_days * 86400.0), np.concatenate([t["r1"], t["v1"]]),
                    method="DOP853", rtol=1e-12, atol=1e-3, dense_output=True)
    r_end = sol.y[:3, -1]
    r_mars, _ = eph.state("mars", jd_dep + tof_days)
    return float(np.linalg.norm(r_end - r_mars) / 1e3), sol


# ---------------------------------------------------------------- plots
def make_plots(path, eph, deps, tofs, C3, DV, best, jd_best, tof_best, sol, park, win, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from astropy.time import Time

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    dep_dt = Time(deps, format="jd").to_datetime()
    a0 = ax[0, 0]
    lv = [c for c in (5, 7.5, 10, 12.5, 15, 20, 30, 40, 60) if np.nanmin(C3) < c < 80]
    cf = a0.contourf(dep_dt, tofs, np.clip(C3, 0, 60), levels=np.linspace(max(np.nanmin(C3) - .5, 0), 60, 40),
                     cmap="viridis_r", extend="max")
    cs = a0.contour(dep_dt, tofs, C3, levels=lv, colors="w", linewidths=.6)
    a0.clabel(cs, fmt="%g", fontsize=7)
    plt.colorbar(cf, ax=a0, label="C3 (km$^2$/s$^2$)")
    a0.plot(Time(jd_best, format="jd").to_datetime(), tof_best, "r*", ms=16, label=f"optimum C3={best['c3']:.1f}")
    a0.set_xlabel("departure date"); a0.set_ylabel("time of flight (days)")
    a0.set_title("Earth->Mars porkchop (C3)"); a0.legend(fontsize=8)
    a0.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); a0.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    a1 = ax[0, 1]
    ts = np.linspace(0, sol.t[-1], 400)
    arc = sol.sol(ts)[:3].T / AU
    jd_ts = jd_best + ts / 86400
    for body, col in (("earth", "tab:blue"), ("mars", "tab:red")):
        per = 365.25 if body == "earth" else 687.0
        orb = eph.positions(body, np.linspace(jd_best, jd_best + per * 0.999, 300)) \
            if jd_best + per < eph.jd1 else eph.positions(body, np.linspace(eph.jd0, eph.jd1, 300))
        a1.plot(orb[:, 0] / AU, orb[:, 1] / AU, color=col, lw=.8, alpha=.5)
    a1.plot(arc[:, 0], arc[:, 1], "k-", lw=2, label="transfer (Lambert, propagated)")
    pe = eph.positions("earth", [jd_best])[0] / AU
    pm = eph.positions("mars", [jd_best + tof_best])[0] / AU
    pm0 = eph.positions("mars", [jd_best])[0] / AU
    a1.plot(*pe[:2], "o", color="tab:blue", ms=9, label="Earth at departure")
    a1.plot(*pm0[:2], "o", mfc="none", color="tab:red", ms=9, label="Mars at departure")
    a1.plot(*pm[:2], "o", color="tab:red", ms=9, label="Mars at arrival")
    a1.plot(0, 0, "o", color="gold", ms=12)
    a1.set_aspect("equal"); a1.legend(fontsize=7, loc="upper right"); a1.grid(alpha=.3)
    a1.set_xlabel("x (AU)"); a1.set_ylabel("y (AU)"); a1.set_title("Heliocentric transfer (ecliptic J2000)")

    a2 = ax[0, 2]
    c3min = np.nanmin(C3, axis=0)
    a2.plot(dep_dt, c3min)
    a2.axhline(win["c3_limit"], color="r", ls="--", lw=.8, label=f"C3* + {win['margin']:.0f}")
    a2.axvline(Time(jd_best, format="jd").to_datetime(), color="k", lw=.8)
    a2.set_title(f"Best C3 per departure date - window ~{win['width_days']:.0f} days wide")
    a2.set_ylabel("min over TOF of C3"); a2.legend(); a2.grid(alpha=.3)
    a2.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    a3 = ax[1, 0]
    a3.bar(["departure\nburn", "Mars capture\n(400 km circ)", "total"],
           [best["dv_dep"], best["dv_cap"], best["dv_total"]], color=["#4c72b0", "#55a868", "#8172b2"])
    for i, v in enumerate([best["dv_dep"], best["dv_cap"], best["dv_total"]]):
        a3.text(i, v + .03, f"{v:.2f}", ha="center")
    a3.set_ylabel("km/s"); a3.set_title("Impulsive delta-v budget")

    a4 = ax[1, 1]
    cf2 = a4.contourf(dep_dt, tofs, np.clip(DV, 0, 9), levels=30, cmap="magma_r", extend="max")
    plt.colorbar(cf2, ax=a4, label="total dv (km/s)")
    a4.plot(Time(jd_best, format="jd").to_datetime(), tof_best, "c*", ms=14)
    a4.set_title("Total delta-v (departure from Phase 3 orbit + capture)")
    a4.set_xlabel("departure date"); a4.set_ylabel("TOF (days)")
    a4.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); a4.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    a5 = ax[1, 2]
    a5.axis("off")
    txt = (f"MISSION SUMMARY\n{'-' * 38}\n"
           f"Departure   {jd_to_date(jd_best)}\n"
           f"Arrival     {jd_to_date(jd_best + tof_best)}\n"
           f"TOF         {tof_best:.1f} d\n"
           f"C3          {best['c3']:.2f} km2/s2   (v_inf {best['vinf_dep']:.2f} km/s)\n"
           f"Arr v_inf   {best['vinf_arr']:.2f} km/s\n"
           f"Dep burn    {best['dv_dep']:.3f} km/s  from r = {park['r_park_m'] / 1e3:.0f} km\n"
           f"Capture     {best['dv_cap']:.3f} km/s\n"
           f"TOTAL       {best['dv_total']:.3f} km/s\n\n"
           f"Parking orbit from Phase 3:\n  {park['source_sat']}\n  alt {park['alt_park_km']:.0f} km, i = {park['i_park_deg']:.1f} deg\n"
           f"Launch asymptote dec {win['dla_deg']:+.1f} deg\n"
           f"  -> needs i >= {abs(win['dla_deg']):.1f} deg: "
           f"{'OK' if park['i_park_deg'] >= abs(win['dla_deg']) else 'PLANE CHANGE NEEDED'}\n\n"
           f"Lambert check: miss {win['miss_km']:.0f} km\nEphemeris: {eph.kind}")
    a5.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=10)
    fig.suptitle("Phase 4 - Earth to Mars mission design (patched conics, impulsive)")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- main
def run(data_dir="data", dep_start="2026-08-15", dep_span_days=240, tof_min=130, tof_max=420,
        n_dep=90, n_tof=75, kernel=None, show=False):
    print("=" * 60)
    print("PHASE 4  Earth -> Mars mission designer")
    print("=" * 60)
    pk = os.path.join(data_dir, "parking_orbit.json")
    if os.path.exists(pk):
        with open(pk) as f:
            park = json.load(f)
        print(f"Parking orbit from Phase 3: {park['source_sat']}, alt {park['alt_park_km']:.1f} km, "
              f"i = {park['i_park_deg']:.2f} deg")
    else:
        park = dict(source_sat="DEFAULT (Phase 3 output not found)", r_park_m=C.RE_M + 200e3,
                    alt_park_km=200.0, i_park_deg=28.5)
        print("WARNING: parking_orbit.json not found - using a generic 200 km orbit")
    r_park = park["r_park_m"]
    r_cap = R_MARS + MARS_CAPTURE_ALT_M

    jd0 = date_to_jd(dep_start)
    eph = Ephemeris(jd0 - 5, jd0 + dep_span_days + tof_max + 10, kernel=kernel)
    print(f"Ephemeris: {eph.kind}")

    tofs = np.linspace(tof_min, tof_max, n_tof)
    print(f"Porkchop grid {n_dep} x {n_tof} ...")
    deps, C3, DV = porkchop(eph, jd0, dep_span_days, tofs, n_dep, r_park, r_cap)
    ig = np.unravel_index(np.nanargmin(DV), DV.shape)
    ic = np.unravel_index(np.nanargmin(C3), C3.shape)
    print(f"  grid min total dv : {DV[ig]:.3f} km/s  dep {jd_to_date(deps[ig[1]])}  TOF {tofs[ig[0]]:.0f} d  (C3 {C3[ig]:.2f})")
    print(f"  grid min C3       : {C3[ic]:.2f} km2/s2 dep {jd_to_date(deps[ic[1]])}  TOF {tofs[ic[0]]:.0f} d")

    print("Differential evolution + Nelder-Mead on total delta-v ...")
    (jd_best, tof_best), dv_best, nfev = optimise(
        eph, [(jd0, jd0 + dep_span_days), (tof_min, tof_max)], r_park, r_cap)
    best = transfer(eph, jd_best, tof_best, r_park, r_cap)
    print(f"  optimum ({nfev} evaluations): dep {jd_to_date(jd_best)}  TOF {tof_best:.1f} d  "
          f"C3 {best['c3']:.2f}  total dv {best['dv_total']:.3f} km/s")
    if dv_best > DV[ig] + 1e-3:
        print("  NOTE: optimiser did not beat the grid minimum; reporting the better of the two")
        jd_best, tof_best = deps[ig[1]], tofs[ig[0]]
        best = transfer(eph, jd_best, tof_best, r_park, r_cap)

    miss, sol = miss_distance_km(eph, best, tof_best, jd_best)
    print(f"Lambert self-check: propagated arrival misses Mars by {miss:.1f} km "
          f"(old code: ~272,000,000 km)")

    # sensitivity of the REAL c3 (central difference) and window width from the porkchop
    h = 1.0
    cp = transfer(eph, jd_best + h, tof_best, r_park, r_cap)["c3"]
    cm = transfer(eph, jd_best - h, tof_best, r_park, r_cap)["c3"]
    dc3_dt = (cp - cm) / (2 * h)
    margin = 2.0
    c3min_per_dep = np.nanmin(C3, axis=0)
    ok = c3min_per_dep <= best["c3"] + margin
    k0 = int(np.argmin(np.abs(deps - jd_best)))
    lo = hi = k0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(deps) - 1 and ok[hi + 1]:
        hi += 1
    width = float(deps[hi] - deps[lo])
    vinf_ecl = best["vinf_dep_vec"]
    vinf_eq = R_EQ2ECL.T @ vinf_ecl
    dla = float(np.degrees(np.arcsin(vinf_eq[2] / np.linalg.norm(vinf_eq))))
    win = dict(width_days=width, c3_limit=best["c3"] + margin, margin=margin, dla_deg=dla, miss_km=miss)
    print(f"Window: C3 within +{margin:.0f} km2/s2 of optimum for ~{width:.0f} days "
          f"({jd_to_date(deps[lo])} to {jd_to_date(deps[hi])}); dC3/dt_dep = {dc3_dt:+.3f} (km2/s2)/day")
    print(f"Launch asymptote declination {dla:+.1f} deg vs parking inclination {park['i_park_deg']:.1f} deg -> "
          f"{'OK' if park['i_park_deg'] >= abs(dla) else 'PLANE CHANGE NEEDED (not costed)'}")

    # outputs
    ts = np.linspace(0, tof_best * 86400.0, 300)
    st = sol.sol(ts)
    pe = eph.positions("earth", jd_best + ts / 86400)
    pm = eph.positions("mars", jd_best + ts / 86400)
    pd.DataFrame(dict(t_days=ts / 86400, jd_utc=jd_best + ts / 86400,
                      x_au=st[0] / AU, y_au=st[1] / AU, z_au=st[2] / AU,
                      earth_x_au=pe[:, 0] / AU, earth_y_au=pe[:, 1] / AU,
                      mars_x_au=pm[:, 0] / AU, mars_y_au=pm[:, 1] / AU)
                 ).to_csv(os.path.join(data_dir, "mission_trajectory.csv"), index=False)
    grid_rows = [dict(departure_jd=deps[j], departure=jd_to_date(deps[j]), tof_days=tofs[i],
                      c3_km2s2=C3[i, j], total_dv_kms=DV[i, j])
                 for i in range(len(tofs)) for j in range(len(deps)) if not np.isnan(C3[i, j])]
    pd.DataFrame(grid_rows).to_csv(os.path.join(data_dir, "porkchop.csv"), index=False)
    summary = dict(departure=jd_to_date(jd_best), arrival=jd_to_date(jd_best + tof_best),
                   tof_days=float(tof_best), c3_km2s2=best["c3"], vinf_dep_kms=best["vinf_dep"],
                   vinf_arr_kms=best["vinf_arr"], dv_departure_kms=best["dv_dep"],
                   dv_capture_kms=best["dv_cap"], dv_total_kms=best["dv_total"],
                   parking_source=park["source_sat"], parking_alt_km=park["alt_park_km"],
                   parking_incl_deg=park["i_park_deg"], dla_deg=dla,
                   window_width_days=width, dc3_per_day=float(dc3_dt),
                   lambert_miss_km=miss, ephemeris=eph.kind, optimiser="scipy DE + Nelder-Mead")
    with open(os.path.join(data_dir, "mission_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_plots(os.path.join(data_dir, "mission_plots.png"), eph, deps, tofs, C3, DV, best,
               jd_best, tof_best, sol, park, win, show)
    print(f"Wrote mission_summary.json, mission_trajectory.csv, porkchop.csv, mission_plots.png -> {data_dir}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dep-start", default="2026-08-15")
    ap.add_argument("--dep-span-days", type=float, default=240)
    ap.add_argument("--tof-min", type=float, default=130)
    ap.add_argument("--tof-max", type=float, default=420)
    ap.add_argument("--kernel", default=None, help="optional JPL .bsp kernel (needs spiceypy)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.dep_start, a.dep_span_days, a.tof_min, a.tof_max, kernel=a.kernel, show=a.show)
