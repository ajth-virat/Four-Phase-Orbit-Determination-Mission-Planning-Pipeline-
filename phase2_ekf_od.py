"""
PHASE 2 - Orbit determination with an Extended Kalman Filter (ECI frame)
========================================================================
INPUT   data/radar_observations.csv   noisy range/az/el ONLY (no truth columns)
        data/truth_states.csv         used ONLY by score_against_truth() for grading
OUTPUT  data/ekf_results.csv          per-time estimate for every satellite
        data/od_summary.csv           per-satellite accuracy + consistency table
        data/selected_orbit.json      the ONE satellite handed to Phase 3
        data/ekf_plots.png

WHAT WAS WRONG BEFORE (and is fixed here)
    The old filter integrated the dynamics in ECEF (a rotating frame) while using
    inertial gravity equations.  That omits the Coriolis + centrifugal accelerations
    (~1 m/s^2), so the filter's physics was wrong and it was ~290x overconfident.
    Here the state lives in ECI (TEME, same frame as SGP4).  The Earth's rotation
    enters only in the MEASUREMENT model, where it belongs:
        r_ecef = R3(gmst) r_eci  ->  ENU at the station  ->  (range, az, el)

STATE      x = [x, y, z, vx, vy, vz]  ECI, metres and m/s
DYNAMICS   two-body + J2, integrated with DOP853 together with the state-transition
           matrix Phi (so the covariance prediction uses the true linearisation,
           not the crude F = I + A*dt of the old code)
SEED       Herrick-Gibbs on three NOISY measurements (never the true_* columns)
PROCESS    continuous white-acceleration noise sigma_a covers what we do not model
NOISE      (J3/J4, drag, SRP, third bodies)
CHECKS     NIS (needs no truth) and NEES (needs truth) - is the filter honest about
           its own uncertainty?
SELECTION  choose ONE satellite for Phase 3 using NIS + final covariance only,
           never the truth (real data has no truth).

Run:  python phase2_ekf_od.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import chi2

import common as C

# process-noise level (m/s^2, 1-sigma white acceleration) by orbit class.
# LEO sees drag and stronger higher-order gravity; MEO/GEO are smoother.
SIGMA_A = {"LEO": 1e-4, "MEO": 2e-5, "GEO": 2e-5}
HG_SPACING_S = {"LEO": 30.0, "MEO": 300.0, "GEO": 300.0}


# ---------------------------------------------------------------- dynamics
def accel(r):
    """Two-body + J2 acceleration in ECI (m/s^2)."""
    rm = np.linalg.norm(r)
    x, y, z = r
    a = -C.MU_EARTH / rm ** 3 * r
    f = 1.5 * C.J2 * C.MU_EARTH * C.RE_M ** 2 / rm ** 5
    k = 5.0 * z * z / (rm * rm)
    return a + f * np.array([x * (k - 1.0), y * (k - 1.0), z * (k - 3.0)])


def grav_gradient(r, eps=10.0):
    """d(accel)/d(position), 3x3, by central differences (robust for the J2 cross terms)."""
    G = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        G[:, k] = (accel(r + d) - accel(r - d)) / (2 * eps)
    return G


def _rhs(t, y):
    G = grav_gradient(y[:3])
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = G
    Phi = y[6:].reshape(6, 6)
    return np.concatenate([y[3:6], accel(y[:3]), (A @ Phi).ravel()])


def propagate(x0, dt):
    """Propagate state dt seconds (negative allowed).  Returns (x, Phi)."""
    if dt == 0.0:
        return x0.copy(), np.eye(6)
    y0 = np.concatenate([x0, np.eye(6).ravel()])
    sol = solve_ivp(_rhs, (0.0, dt), y0, method="DOP853", rtol=1e-10, atol=1e-6)
    yf = sol.y[:, -1]
    return yf[:6], yf[6:].reshape(6, 6)


def propagate_states(x0, times):
    """Propagate to a list of times (s) in one integration (no STM).  Used for plots/tests."""
    def rhs(t, y):
        return np.concatenate([y[3:6], accel(y[:3])])
    sol = solve_ivp(rhs, (0.0, times[-1]), x0, method="DOP853", t_eval=times,
                    rtol=1e-10, atol=1e-6)
    return sol.y.T


# ---------------------------------------------------------------- measurement model
def measurement(x, jd, st: C.Station, want_jacobian=True):
    """h(x) = [range, az, el] and H = dh/dx (3x6).  Only the position block is non-zero."""
    R = C.rot_eci_to_ecef(C.gmst_rad(jd))
    enu = st.R_enu @ (R @ x[:3] - st.ecef)
    e, n, u = enu
    rng = np.linalg.norm(enu)
    rh = np.hypot(e, n)
    h = np.array([rng, np.arctan2(e, n) % (2 * np.pi), np.arcsin(u / rng)])
    if not want_jacobian:
        return h
    d_rng = enu / rng
    d_az = np.array([n / rh ** 2, -e / rh ** 2, 0.0])
    d_el = np.array([-e * u / (rng ** 2 * rh), -n * u / (rng ** 2 * rh), rh / rng ** 2])
    T = st.R_enu @ R                    # d(enu)/d(r_eci)
    H = np.zeros((3, 6))
    H[0, :3], H[1, :3], H[2, :3] = d_rng @ T, d_az @ T, d_el @ T
    return h, H


# ---------------------------------------------------------------- Herrick-Gibbs seed
def herrick_gibbs(r1, r2, r3, t1, t2, t3, mu=C.MU_EARTH):
    """Vallado Algorithm 55: velocity at the middle of three close position fixes (ECI)."""
    d21, d31, d32 = t2 - t1, t3 - t1, t3 - t2
    n1, n2, n3 = (np.linalg.norm(r) for r in (r1, r2, r3))
    return (-d32 * (1 / (d21 * d31) + mu / (12 * n1 ** 3)) * r1
            + (d32 - d21) * (1 / (d21 * d32) + mu / (12 * n2 ** 3)) * r2
            + d21 * (1 / (d32 * d31) + mu / (12 * n3 ** 3)) * r3)


def seed_from_measurements(obs, st: C.Station, cls: str):
    """Initial state + covariance from the first noisy measurements only.

    Returns (x_seed, P_seed, idx_mid) where the state is valid at observation idx_mid.
    """
    t = obs["elapsed_sec"].values
    spacing = HG_SPACING_S[cls]
    k = int(max(1, round(spacing / max(np.median(np.diff(t)), 1e-9))))
    if 2 * k >= len(obs):
        k = max(1, (len(obs) - 1) // 2)
    i1, i2, i3 = 0, k, 2 * k

    def pos(i):
        row = obs.iloc[i]
        return st.eci_from_rae(row["range_m"], row["azimuth_deg"] * C.DEG2RAD,
                               row["elevation_deg"] * C.DEG2RAD, row["jd_utc"])

    r1, r2, r3 = pos(i1), pos(i2), pos(i3)
    v2 = herrick_gibbs(r1, r2, r3, t[i1], t[i2], t[i3])

    # seed uncertainty from sensor noise, inflated x3 because H-G is only 2nd order
    rng_mean = obs["range_m"].iloc[[i1, i2, i3]].mean()
    s_ang = np.sqrt(2.0) * C.SIGMA_AZ_DEG * C.DEG2RAD * rng_mean
    s_pos = np.sqrt(C.SIGMA_RANGE_M ** 2 + s_ang ** 2)
    dt_span = t[i3] - t[i1]
    s_vel = np.sqrt(2.0) * s_pos / dt_span * 2.0
    P = np.diag([(3 * s_pos) ** 2] * 3 + [(3 * s_vel) ** 2] * 3)
    return np.concatenate([r2, v2]), P, i2


# ---------------------------------------------------------------- the filter
def run_ekf(obs: pd.DataFrame, st: C.Station, cls: str, sigma_a=None,
            sigma_scale=1.0, range_bias_assumed=0.0):
    """Run the EKF over one satellite's observations (sorted by time).

    sigma_scale multiplies the assumed measurement sigmas (use >1 to deliberately
    de-tune the filter for stress tests).  Returns dict of arrays.
    """
    obs = obs.sort_values("elapsed_sec").reset_index(drop=True)
    qa = (SIGMA_A[cls] if sigma_a is None else sigma_a) ** 2
    R = np.diag([(C.SIGMA_RANGE_M * sigma_scale) ** 2,
                 (C.SIGMA_AZ_DEG * C.DEG2RAD * sigma_scale) ** 2,
                 (C.SIGMA_EL_DEG * C.DEG2RAD * sigma_scale) ** 2])

    x, P, imid = seed_from_measurements(obs, st, cls)
    t = obs["elapsed_sec"].values
    # move the seed back to the first observation time, then run forward over ALL data
    x, Phi = propagate(x, t[0] - t[imid])
    P = Phi @ P @ Phi.T

    n = len(obs)
    X = np.zeros((n, 6)); Pd = np.zeros((n, 6, 6))
    Y = np.full((n, 3), np.nan); NIS = np.zeros(n); DOF = np.zeros(n, int)
    t_prev = t[0]
    for k in range(n):
        dt = t[k] - t_prev
        if dt != 0.0:
            x, Phi = propagate(x, dt)
            Q = qa * np.block([[dt ** 3 / 3 * np.eye(3), dt ** 2 / 2 * np.eye(3)],
                               [dt ** 2 / 2 * np.eye(3), dt * np.eye(3)]])
            P = Phi @ P @ Phi.T + Q
        t_prev = t[k]

        row = obs.iloc[k]
        z = np.array([row["range_m"] - range_bias_assumed,
                      row["azimuth_deg"] * C.DEG2RAD, row["elevation_deg"] * C.DEG2RAD])
        h, H = measurement(x, row["jd_utc"], st)
        y = z - h
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi      # wrap azimuth innovation

        rows = [0, 1, 2]
        if h[2] > np.radians(85.0):                       # azimuth is singular near zenith
            rows = [0, 2]
        yv, Hv, Rv = y[rows], H[rows], R[np.ix_(rows, rows)]
        S = Hv @ P @ Hv.T + Rv
        K = P @ Hv.T @ np.linalg.inv(S)
        x = x + K @ yv
        IKH = np.eye(6) - K @ Hv
        P = IKH @ P @ IKH.T + K @ Rv @ K.T                # Joseph form
        P = 0.5 * (P + P.T)

        X[k], Pd[k] = x, P
        Y[k, rows] = yv
        NIS[k] = float(yv @ np.linalg.solve(S, yv))
        DOF[k] = len(rows)

    return dict(obs=obs, X=X, P=Pd, Y=Y, NIS=NIS, DOF=DOF)


# ---------------------------------------------------------------- scoring (uses truth)
def score_against_truth(res, truth: pd.DataFrame):
    """Compare estimate with SGP4 truth.  FOR GRADING ONLY - never feeds selection."""
    obs = res["obs"]
    tr = truth.set_index("elapsed_sec").loc[obs["elapsed_sec"].values]
    xt = tr[["x_m", "y_m", "z_m", "vx_ms", "vy_ms", "vz_ms"]].values
    err = res["X"] - xt
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(err, res["P"])])
    pos_err = np.linalg.norm(err[:, :3], axis=1)
    vel_err = np.linalg.norm(err[:, 3:], axis=1)
    sig_pos = np.sqrt(np.trace(res["P"][:, :3, :3], axis1=1, axis2=2))
    ek, et = C.state_to_keplerian(res["X"][-1, :3], res["X"][-1, 3:]), \
        C.state_to_keplerian(xt[-1, :3], xt[-1, 3:])
    return dict(err=err, nees=nees, pos_err=pos_err, vel_err=vel_err, sig_pos=sig_pos,
                a_err_m=ek["a"] - et["a"], e_est=ek["e"], e_true=et["e"],
                i_err_deg=ek["i_deg"] - et["i_deg"], xt=xt)


def nis_bounds(dof_total, level=0.95):
    lo, hi = chi2.ppf([(1 - level) / 2, 1 - (1 - level) / 2], dof_total)
    return lo, hi


# ---------------------------------------------------------------- orchestration
def run_all(obs_df, truth_df=None, **kw):
    st = C.Station()
    results, rows = {}, []
    for name, g in obs_df.groupby("sat_name", sort=False):
        cls = g["orbit_class"].iloc[0]
        res = run_ekf(g, st, cls, **kw)
        D = int(res["DOF"].sum())
        nis_sum = float(res["NIS"].sum())
        lo, hi = nis_bounds(D)
        Pf = res["P"][-1]
        row = dict(sat_name=name, orbit_class=cls, n_obs=len(g),
                   arc_min=round(g["elapsed_sec"].max() / 60, 1),
                   nis_mean=nis_sum / D * 3, nis_sum=nis_sum, nis_lo=lo, nis_hi=hi,
                   nis_consistent=bool(lo <= nis_sum <= hi),
                   final_sigma_pos_m=float(np.sqrt(np.trace(Pf[:3, :3]) / 3)),
                   final_sigma_vel_ms=float(np.sqrt(np.trace(Pf[3:, 3:]) / 3)))
        if truth_df is not None:
            sc = score_against_truth(res, truth_df[truth_df["sat_name"] == name])
            res["score"] = sc
            row.update(final_pos_err_m=float(sc["pos_err"][-1]),
                       final_vel_err_ms=float(sc["vel_err"][-1]),
                       rms_pos_err_m=float(np.sqrt(np.mean(sc["pos_err"] ** 2))),
                       a_err_km=sc["a_err_m"] / 1000, i_err_deg=sc["i_err_deg"],
                       mean_nees=float(sc["nees"].mean()),
                       frac_pos_in_3sigma=float(np.mean(sc["pos_err"] <= 3 * sc["sig_pos"] * np.sqrt(3))))
        results[name] = res
        rows.append(row)
    return results, pd.DataFrame(rows)


def select_satellite(summary: pd.DataFrame):
    """Pick ONE satellite using only truth-free quantities.

    Rule: among filters whose NIS passes the 95% chi-square consistency test,
    take the smallest final position sigma.  If none pass, take the NIS closest
    to its expected value and say so.  Returns (name, reason).
    """
    ok = summary[summary["nis_consistent"]]
    if len(ok):
        best = ok.sort_values("final_sigma_pos_m").iloc[0]
        return best["sat_name"], (f"NIS consistent (95% chi-square) and smallest final "
                                  f"position sigma ({best['final_sigma_pos_m']:.0f} m)")
    dev = (summary["nis_sum"] / (summary["nis_lo"] + summary["nis_hi"]) * 2 - 1).abs()
    best = summary.loc[dev.idxmin()]
    return best["sat_name"], "WARNING: no filter passed NIS test; picked the least inconsistent"


def write_selected(res, row, reason, path):
    obs = res["obs"]
    k = len(obs) - 1
    out = dict(sat_name=row["sat_name"], orbit_class=row["orbit_class"],
               epoch_jd_utc=float(obs["jd_utc"].iloc[k]),
               epoch_iso_utc=obs["timestamp_utc"].iloc[k],
               r_eci_m=res["X"][k, :3].tolist(), v_eci_ms=res["X"][k, 3:].tolist(),
               P_eci=res["P"][k].tolist(), selection_reason=reason,
               nis_sum=row["nis_sum"], nis_bounds=[row["nis_lo"], row["nis_hi"]],
               n_obs=int(row["n_obs"]), arc_min=float(row["arc_min"]))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------- plots
def make_plots(results, summary, selected, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = results[selected]
    t = res["obs"]["elapsed_sec"].values / 60
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    if "score" in res:
        sc = res["score"]
        ax[0, 0].semilogy(t, sc["pos_err"], label="true position error")
        ax[0, 0].semilogy(t, 3 * sc["sig_pos"] * np.sqrt(3), "--", label="3-sigma (filter's claim)")
        ax[0, 0].set_title(f"{selected}: position error vs filter 3-sigma"); ax[0, 0].legend()
        ax[0, 1].semilogy(t, sc["vel_err"], label="true velocity error")
        sv = np.sqrt(np.trace(res["P"][:, 3:, 3:], axis1=1, axis2=2))
        ax[0, 1].semilogy(t, 3 * sv, "--", label="3-sigma (trace)")
        ax[0, 1].set_title("Velocity error vs filter claim"); ax[0, 1].legend()
        ax[0, 2].plot(t, sc["nees"]); ax[0, 2].axhline(chi2.ppf(.95, 6), color="r", ls="--", label="95% bound (6 dof)")
        ax[0, 2].set_title("NEES (needs truth)"); ax[0, 2].legend()
    Y = res["Y"]
    ax[1, 0].plot(t, Y[:, 0], ".-", ms=3); ax[1, 0].axhline(0, color="k", lw=.5)
    ax[1, 0].set_title(f"Range innovation (m); sensor sigma = {C.SIGMA_RANGE_M:.0f} m")
    ax[1, 1].plot(t, np.degrees(Y[:, 2]), ".-", ms=3); ax[1, 1].axhline(0, color="k", lw=.5)
    ax[1, 1].set_title(f"Elevation innovation (deg); sensor sigma = {C.SIGMA_EL_DEG} deg")
    if "final_pos_err_m" in summary:
        s = summary.sort_values("orbit_class")
        cols = {"LEO": "tab:blue", "MEO": "tab:orange", "GEO": "tab:green"}
        ax[1, 2].barh(s["sat_name"], s["final_pos_err_m"] / 1000, color=[cols[c] for c in s["orbit_class"]])
        ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("final position error (km, log)")
        ax[1, 2].set_title("All test satellites (scored vs SGP4 truth)")
        ax[1, 2].tick_params(axis="y", labelsize=6)
    for a in ax.ravel():
        a.grid(alpha=.3)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]):
        a.set_xlabel("minutes since arc start")
    fig.suptitle("Phase 2 - EKF orbit determination (ECI frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def run(data_dir="data", show=False, **kw):
    print("=" * 60)
    print("PHASE 2  EKF orbit determination (ECI)")
    print("=" * 60)
    obs_df = pd.read_csv(os.path.join(data_dir, "radar_observations.csv"))
    assert not any(c.startswith("true") for c in obs_df.columns), "observations must not contain truth"
    truth_path = os.path.join(data_dir, "truth_states.csv")
    truth_df = pd.read_csv(truth_path) if os.path.exists(truth_path) else None

    results, summary = run_all(obs_df, truth_df, **kw)
    selected, reason = select_satellite(summary)
    summary["selected"] = summary["sat_name"] == selected

    pd.set_option("display.width", 200)
    cols = ["sat_name", "orbit_class", "n_obs", "nis_consistent", "final_sigma_pos_m"]
    if truth_df is not None:
        cols += ["final_pos_err_m", "final_vel_err_ms", "a_err_km", "mean_nees"]
    print(summary[cols + ["selected"]].round(2).to_string(index=False))
    print(f"\nSelected for Phase 3: {selected}\n  reason: {reason}")

    # per-time output for every satellite
    rows = []
    for name, res in results.items():
        o = res["obs"]
        for k in range(len(o)):
            g = C.gmst_rad(o["jd_utc"].iloc[k])
            re, ve = C.eci_to_ecef_state(res["X"][k, :3], res["X"][k, 3:], g)
            rows.append(dict(sat_name=name, elapsed_sec=o["elapsed_sec"].iloc[k],
                             timestamp_utc=o["timestamp_utc"].iloc[k], jd_utc=o["jd_utc"].iloc[k],
                             x_m=res["X"][k, 0], y_m=res["X"][k, 1], z_m=res["X"][k, 2],
                             vx_ms=res["X"][k, 3], vy_ms=res["X"][k, 4], vz_ms=res["X"][k, 5],
                             x_ecef_m=re[0], y_ecef_m=re[1], z_ecef_m=re[2],
                             sigma_pos_m=float(np.sqrt(np.trace(res["P"][k, :3, :3]) / 3)),
                             sigma_vel_ms=float(np.sqrt(np.trace(res["P"][k, 3:, 3:]) / 3)),
                             nis=res["NIS"][k]))
    pd.DataFrame(rows).to_csv(os.path.join(data_dir, "ekf_results.csv"), index=False)
    summary.to_csv(os.path.join(data_dir, "od_summary.csv"), index=False)
    row = summary[summary["sat_name"] == selected].iloc[0]
    write_selected(results[selected], row, reason, os.path.join(data_dir, "selected_orbit.json"))
    make_plots(results, summary, selected, os.path.join(data_dir, "ekf_plots.png"), show)
    print(f"Wrote ekf_results.csv, od_summary.csv, selected_orbit.json, ekf_plots.png -> {data_dir}/")
    return results, summary, selected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.show)
