"""
PHASE 3 - Maneuver planner (reads the orbit Phase 2 estimated)
==============================================================
INPUT   data/selected_orbit.json    ECI state + covariance of the satellite Phase 2 chose
OUTPUT  data/maneuver_plan.csv      burns needed to reach the parking orbit
        data/parking_orbit.json     THE HAND-OFF to Phase 4 (parking-orbit radius, inclination)
        data/maneuver_plots.png

WHAT CHANGED vs the old version
    * No more hard-coded AJISAI target.  The orbit comes from the EKF estimate.
    * Delta-v comes with an UNCERTAINTY: we draw samples from the EKF covariance, push
      each through the planner, and report mean +/- sigma.  A burn of 10 m/s computed
      from an orbit known to +/-0.5 km means little unless you say how well it is known.
    * Phase 4 now actually consumes this phase's output (parking-orbit radius).
    * Drag removed: the old exponential-atmosphere numbers were not defensible.  J2
      propagation only, and that is stated plainly.

WHAT IT DOES
    1. state -> Keplerian elements (Vallado Algorithm 9) with first-order sigmas
    2. propagate one period with J2 (sanity plot, not used for the burns)
    3. plan a two-impulse Hohmann-type transfer from the estimated orbit to a
       CIRCULAR parking orbit (default: circular at the estimated semi-major axis),
       plus an optional plane change if you request a different inclination
    4. write the parking orbit for Phase 4

Run:  python phase3_maneuver_planner.py --data-dir data
      python phase3_maneuver_planner.py --target-alt-km 500 --target-inc-deg 28.5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import common as C
import phase2_ekf_od as P2

MU = C.MU_EARTH


# ---------------------------------------------------------------- maneuver math
def vis_viva(r, a):
    return np.sqrt(MU * (2.0 / r - 1.0 / a))


def hohmann_circular(r1, r2):
    """Textbook Hohmann between circular coplanar orbits (Vallado Alg. 36).

    Returns dict with dv1, dv2, total (m/s) and transfer time (s).
    """
    at = 0.5 * (r1 + r2)
    dv1 = np.sqrt(MU / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1.0)
    dv2 = np.sqrt(MU / r2) * (1.0 - np.sqrt(2 * r1 / (r1 + r2)))
    return dict(dv1=dv1, dv2=dv2, total=abs(dv1) + abs(dv2),
                tof_s=np.pi * np.sqrt(at ** 3 / MU), a_transfer=at)


def plane_change_dv(v, d_inc_rad):
    """Pure inclination change at a node: dv = 2 v sin(di/2)."""
    return 2.0 * v * np.sin(abs(d_inc_rad) / 2.0)


def transfer_to_circular(a, e, r_t):
    """Two burns from an ellipse (a, e) to a circular orbit of radius r_t.

    Burn 1 at periapsis sets the opposite apsis to r_t, burn 2 circularises.
    Works whether r_t is above or below the periapsis radius.
    """
    rp = a * (1.0 - e)
    v_p = vis_viva(rp, a)
    at = 0.5 * (rp + r_t)
    dv1 = vis_viva(rp, at) - v_p
    dv2 = np.sqrt(MU / r_t) - vis_viva(r_t, at)
    return dict(dv1=dv1, dv2=dv2, total=abs(dv1) + abs(dv2), a_transfer=at,
                tof_s=np.pi * np.sqrt(at ** 3 / MU), rp=rp)


def plan(r_eci, v_eci, r_target=None, inc_target_deg=None):
    """Full plan from a state.  r_target default = estimated semi-major axis (circular)."""
    el = C.state_to_keplerian(r_eci, v_eci)
    rt = el["a"] if r_target is None else r_target
    tr = transfer_to_circular(el["a"], el["e"], rt)
    d_inc = 0.0 if inc_target_deg is None else np.radians(inc_target_deg - el["i_deg"])
    dv_pc = plane_change_dv(np.sqrt(MU / rt), d_inc)
    return dict(el=el, r_target=rt, transfer=tr, dv_plane=dv_pc, d_inc_deg=np.degrees(d_inc),
                dv_total=tr["total"] + dv_pc)


def _elements_vec(x):
    el = C.state_to_keplerian(x[:3], x[3:])
    return np.array([el["a"], el["e"], np.radians(el["i_deg"])])


def element_sigmas(x, P):
    """First-order (a, e, i) uncertainty from the state covariance."""
    J = np.zeros((3, 6))
    steps = np.array([1.0] * 3 + [1e-3] * 3)
    for k in range(6):
        d = np.zeros(6)
        d[k] = steps[k]
        J[:, k] = (_elements_vec(x + d) - _elements_vec(x - d)) / (2 * steps[k])
    cov = J @ P @ J.T
    s = np.sqrt(np.diag(cov))
    return dict(sigma_a_m=s[0], sigma_e=s[1], sigma_i_deg=np.degrees(s[2]))


def dv_uncertainty(x, P, r_target=None, inc_target_deg=None, n=400, seed=7):
    """Monte Carlo: sample the state from N(x, P) and re-plan.  Returns mean, std of total dv."""
    gen = np.random.default_rng(seed)
    L = np.linalg.cholesky(P + 1e-9 * np.eye(6) * np.trace(P) / 6)
    out = []
    for _ in range(n):
        xs = x + L @ gen.standard_normal(6)
        out.append(plan(xs[:3], xs[3:], r_target, inc_target_deg)["dv_total"])
    return float(np.mean(out)), float(np.std(out))


def time_to_periapsis(el):
    """Seconds from the estimate epoch to the next periapsis passage."""
    e, nu = el["e"], np.radians(el["nu_deg"])
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = (E - e * np.sin(E)) % (2 * np.pi)
    n = np.sqrt(MU / el["a"] ** 3)
    return ((2 * np.pi - M) % (2 * np.pi)) / n


# ---------------------------------------------------------------- propagation for the sanity plot
def propagate_one_period(x, epoch_jd, period_s, n=240):
    times = np.linspace(0.0, period_s, n)
    X = P2.propagate_states(x, times)
    return times, X


# ---------------------------------------------------------------- plots
def make_plots(path, el, sig, pl, dv_mean, dv_std, times, X, epoch_jd, r_t, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    alt = (np.linalg.norm(X[:, :3], axis=1) - C.RE_M) / 1000
    ax[0, 0].plot(times / 60, alt)
    ax[0, 0].set_xlabel("min"); ax[0, 0].set_ylabel("altitude (km)")
    ax[0, 0].set_title("One period, J2 propagation of the ESTIMATED orbit")

    lat, lon = [], []
    for k in range(len(times)):
        g = C.gmst_rad(epoch_jd + times[k] / 86400.0)
        re = C.rot_eci_to_ecef(g) @ X[k, :3]
        la, lo, _ = C.ecef_to_geodetic_approx(re)
        lat.append(la); lon.append(lo)
    ax[0, 1].scatter(lon, lat, c=times / 60, s=6, cmap="plasma")
    ax[0, 1].plot(C.GS_LON_DEG, C.GS_LAT_DEG, "r^", ms=9)
    ax[0, 1].set_xlim(-180, 180); ax[0, 1].set_ylim(-90, 90)
    ax[0, 1].set_title("Ground track (one period)"); ax[0, 1].set_xlabel("lon"); ax[0, 1].set_ylabel("lat")

    # orbit-plane diagram, Earth to scale
    a_ax = ax[0, 2]
    th = np.linspace(0, 2 * np.pi, 400)
    a_ax.fill(C.RE_M / 1e3 * np.cos(th), C.RE_M / 1e3 * np.sin(th), color="#4a7fb5", alpha=.5, label="Earth")
    e, a = el["e"], el["a"]
    p = a * (1 - e ** 2)
    a_ax.plot(p / (1 + e * np.cos(th)) * np.cos(th) / 1e3, p / (1 + e * np.cos(th)) * np.sin(th) / 1e3,
              label="estimated orbit")
    a_ax.plot(r_t / 1e3 * np.cos(th), r_t / 1e3 * np.sin(th), "g--", label="parking orbit (target)")
    tr = pl["transfer"]
    at, rp = tr["a_transfer"], tr["rp"]
    et = abs(r_t - rp) / (r_t + rp)
    sgn = -1 if r_t > rp else 1
    rr = at * (1 - et ** 2) / (1 + sgn * et * np.cos(th))
    a_ax.plot(rr * np.cos(th) / 1e3, rr * np.sin(th) / 1e3, "r:", label="transfer ellipse")
    a_ax.set_aspect("equal"); a_ax.legend(fontsize=7)
    a_ax.set_title("Orbit plane (km); eccentricity of estimate is within noise"
                   if e < sig["sigma_e"] * 3 else "Orbit plane (km)", fontsize=9)

    names = ["burn 1\n(periapsis)", "burn 2\n(circularise)", "plane\nchange", "TOTAL"]
    vals = [abs(tr["dv1"]), abs(tr["dv2"]), pl["dv_plane"], pl["dv_total"]]
    ax[1, 0].bar(names, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax[1, 0].errorbar([3], [dv_mean], yerr=[dv_std], color="k", capsize=6)
    ax[1, 0].set_ylabel("m/s")
    ax[1, 0].set_title(f"Delta-v: total {dv_mean:.1f} +/- {dv_std:.1f} m/s (1-sigma from OD covariance)", fontsize=9)

    ax[1, 1].axis("off")
    txt = (f"ESTIMATED ORBIT (from Phase 2)\n{'-' * 34}\n"
           f"a   = {el['a'] / 1e3:10.2f} km  +/- {sig['sigma_a_m'] / 1e3:.2f}\n"
           f"e   = {el['e']:10.5f}     +/- {sig['sigma_e']:.5f}\n"
           f"i   = {el['i_deg']:10.3f} deg +/- {sig['sigma_i_deg']:.3f}\n"
           f"RAAN= {el['raan_deg']:10.3f} deg\n"
           f"alt = {el['alt_km']:10.1f} km\n"
           f"T   = {el['period_s'] / 60:10.2f} min\n\n"
           f"PARKING ORBIT -> Phase 4\n{'-' * 34}\n"
           f"radius   = {r_t / 1e3:.2f} km\n"
           f"altitude = {(r_t - C.RE_M) / 1e3:.1f} km\n"
           f"incl.    = {el['i_deg'] + pl['d_inc_deg']:.2f} deg")
    ax[1, 1].text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10)

    ax[1, 2].axis("off")
    ax[1, 2].text(0.02, 0.98,
                  "LIMITS (say these first)\n" + "-" * 34 +
                  "\n- J2 only: no drag, SRP, J3+\n- e is poorly observed from one pass,\n  so burn-1 location is uncertain\n"
                  "- plane change done at the target\n  radius (not optimised)\n- circularisation dv is mostly\n  OD uncertainty, not real need",
                  va="top", family="monospace", fontsize=10)
    for a_ in ax[0, :2].tolist() + [ax[1, 0]]:
        a_.grid(alpha=.3)
    fig.suptitle("Phase 3 - maneuver planner")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- main
def run(data_dir="data", target_alt_km=None, target_inc_deg=None, show=False):
    print("=" * 60)
    print("PHASE 3  Maneuver planner")
    print("=" * 60)
    with open(os.path.join(data_dir, "selected_orbit.json")) as f:
        sel = json.load(f)
    x = np.array(sel["r_eci_m"] + sel["v_eci_ms"])
    P = np.array(sel["P_eci"])
    print(f"Orbit source: {sel['sat_name']} ({sel['orbit_class']}); {sel['selection_reason']}")

    el = C.state_to_keplerian(x[:3], x[3:])
    sig = element_sigmas(x, P)
    print(f"a = {el['a'] / 1e3:.2f} +/- {sig['sigma_a_m'] / 1e3:.2f} km | e = {el['e']:.5f} +/- {sig['sigma_e']:.5f} | "
          f"i = {el['i_deg']:.3f} +/- {sig['sigma_i_deg']:.3f} deg | alt = {el['alt_km']:.1f} km")
    if el["e"] >= 1 or el["a"] < C.RE_M:
        raise ValueError("Estimated orbit is not a valid bound orbit - check Phase 2 output")

    r_t = None if target_alt_km is None else C.RE_M + target_alt_km * 1e3
    pl = plan(x[:3], x[3:], r_t, target_inc_deg)
    r_t = pl["r_target"]
    dv_mean, dv_std = dv_uncertainty(x, P, r_t, target_inc_deg)
    tr = pl["transfer"]
    t1 = time_to_periapsis(el)
    t2 = t1 + tr["tof_s"]
    print(f"Parking orbit: circular r = {r_t / 1e3:.1f} km (alt {(r_t - C.RE_M) / 1e3:.1f} km), "
          f"i = {el['i_deg'] + pl['d_inc_deg']:.2f} deg")
    print(f"Burn 1 {tr['dv1']:+.2f} m/s at periapsis (T+{t1 / 60:.1f} min); "
          f"burn 2 {tr['dv2']:+.2f} m/s (T+{t2 / 60:.1f} min); plane change {pl['dv_plane']:.2f} m/s")
    print(f"Total delta-v = {pl['dv_total']:.2f} m/s  (MC over OD covariance: {dv_mean:.2f} +/- {dv_std:.2f})")

    plan_df = pd.DataFrame([
        dict(maneuver="burn_1_periapsis", dv_ms=tr["dv1"], t_after_epoch_min=t1 / 60,
             radius_before_m=tr["rp"], radius_after_m=tr["rp"]),
        dict(maneuver="burn_2_circularise", dv_ms=tr["dv2"], t_after_epoch_min=t2 / 60,
             radius_before_m=r_t, radius_after_m=r_t),
        dict(maneuver="plane_change", dv_ms=pl["dv_plane"], t_after_epoch_min=np.nan,
             radius_before_m=r_t, radius_after_m=r_t),
    ])
    plan_df.to_csv(os.path.join(data_dir, "maneuver_plan.csv"), index=False)

    park = dict(source_sat=sel["sat_name"], orbit_class=sel["orbit_class"],
                epoch_iso_utc=sel["epoch_iso_utc"],
                a_est_m=el["a"], a_sigma_m=sig["sigma_a_m"], e_est=el["e"], e_sigma=sig["sigma_e"],
                r_park_m=r_t, alt_park_km=(r_t - C.RE_M) / 1e3,
                i_park_deg=el["i_deg"] + pl["d_inc_deg"], raan_deg=el["raan_deg"],
                plan_dv_total_ms=pl["dv_total"], plan_dv_mean_ms=dv_mean, plan_dv_sigma_ms=dv_std)
    with open(os.path.join(data_dir, "parking_orbit.json"), "w") as f:
        json.dump(park, f, indent=2)

    times, X = propagate_one_period(x, sel["epoch_jd_utc"], el["period_s"])
    make_plots(os.path.join(data_dir, "maneuver_plots.png"), el, sig, pl, dv_mean, dv_std,
               times, X, sel["epoch_jd_utc"], r_t, show)

    # grading only (never used above): compare with SGP4 truth if present
    tp = os.path.join(data_dir, "truth_states.csv")
    if os.path.exists(tp):
        tr_df = pd.read_csv(tp)
        g = tr_df[tr_df.sat_name == sel["sat_name"]].iloc[-1]
        et = C.state_to_keplerian(g[["x_m", "y_m", "z_m"]].values.astype(float),
                                  g[["vx_ms", "vy_ms", "vz_ms"]].values.astype(float))
        print(f"[grading only] true a = {et['a'] / 1e3:.2f} km -> estimate error {(el['a'] - et['a']) / 1e3:+.2f} km "
              f"({(el['a'] - et['a']) / sig['sigma_a_m']:+.1f} sigma); true e = {et['e']:.5f}; true i = {et['i_deg']:.3f}")
    print(f"Wrote maneuver_plan.csv, parking_orbit.json, maneuver_plots.png -> {data_dir}/")
    return park


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--target-alt-km", type=float, default=None, help="circular parking altitude (default: estimated a)")
    ap.add_argument("--target-inc-deg", type=float, default=None, help="parking inclination (default: keep)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.target_alt_km, a.target_inc_deg, a.show)
