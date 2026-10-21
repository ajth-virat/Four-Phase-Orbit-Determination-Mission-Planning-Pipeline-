"""
validate_filter.py - Monte Carlo consistency test for the Phase 2 EKF
=====================================================================
A single run proves little: one lucky noise draw can look fine.  The standard test
(Bar-Shalom) is to run the SAME truth arc with many different noise draws and check
that the average NEES ("normalised estimation error squared") matches its theory:

    NEES_k = (x_hat - x_true)^T  P^-1  (x_hat - x_true)    ~  chi-square, 6 dof
    averaged over N independent runs -> bounds chi2(6N)/N

   * average NEES inside the bounds  -> the filter's covariance is honest
   * average NEES far above           -> overconfident (the OLD filter: ~290x)
   * average NEES far below           -> too pessimistic (Q or R inflated)

Usage:  python validate_filter.py --data-dir data --runs 30 [--sat NAME]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

import common as C
import phase2_ekf_od as P2


def renoise(truth_sat: pd.DataFrame, seed: int, noise_scale=1.0, range_bias_m=0.0):
    """Rebuild a noisy observation table from one satellite's truth rows."""
    st = C.Station()
    gen = np.random.default_rng(seed)
    rows = []
    for _, r in truth_sat.iterrows():
        rng = r["true_range_m"] + range_bias_m + gen.normal(0, C.SIGMA_RANGE_M * noise_scale)
        az = np.radians(r["true_az_deg"]) + gen.normal(0, np.radians(C.SIGMA_AZ_DEG) * noise_scale)
        el = np.radians(r["true_el_deg"]) + gen.normal(0, np.radians(C.SIGMA_EL_DEG) * noise_scale)
        rows.append(dict(sat_name=r["sat_name"], orbit_class=r["orbit_class"],
                         timestamp_utc=r["timestamp_utc"], jd_utc=r["jd_utc"],
                         elapsed_sec=r["elapsed_sec"], range_m=rng,
                         azimuth_deg=np.degrees(az) % 360.0, elevation_deg=np.degrees(el)))
    return pd.DataFrame(rows)


def monte_carlo(truth_sat: pd.DataFrame, runs=30, seed0=1000, **noise_kw):
    st = C.Station()
    cls = truth_sat["orbit_class"].iloc[0]
    nees_all, pos_final, vel_final = [], [], []
    for i in range(runs):
        obs = renoise(truth_sat, seed0 + i, **noise_kw.get("noise", {}))
        res = P2.run_ekf(obs, st, cls, **noise_kw.get("filter", {}))
        sc = P2.score_against_truth(res, truth_sat)
        nees_all.append(sc["nees"])
        pos_final.append(sc["pos_err"][-1])
        vel_final.append(sc["vel_err"][-1])
    nees = np.array(nees_all)                       # runs x time
    ens = nees.mean(axis=0)                         # ensemble-average NEES per time step
    lo, hi = chi2.ppf([0.025, 0.975], 6 * runs) / runs
    return dict(ens_nees=ens, lo=lo, hi=hi, mean_nees=float(ens.mean()),
                frac_inside=float(np.mean((ens >= lo) & (ens <= hi))),
                pos_final_rms=float(np.sqrt(np.mean(np.square(pos_final)))),
                vel_final_rms=float(np.sqrt(np.mean(np.square(vel_final)))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--sat", default=None, help="satellite name (default: one LEO, one MEO, one GEO)")
    a = ap.parse_args()
    truth = pd.read_csv(os.path.join(a.data_dir, "truth_states.csv"))
    names = [a.sat] if a.sat else [truth[truth.orbit_class == c]["sat_name"].iloc[0]
                                   for c in ("LEO", "MEO", "GEO") if (truth.orbit_class == c).any()]
    print(f"{'satellite':24s} {'mean NEES':>9s} {'95% bounds':>16s} {'% steps inside':>14s} "
          f"{'final pos RMS':>14s} {'vel RMS':>9s}")
    for n in names:
        r = monte_carlo(truth[truth.sat_name == n], runs=a.runs)
        print(f"{n:24s} {r['mean_nees']:9.2f} {r['lo']:7.2f}-{r['hi']:<7.2f} "
              f"{100 * r['frac_inside']:13.0f}% {r['pos_final_rms']:13.0f}m {r['vel_final_rms']:8.2f}")
    print("\nExpected NEES = 6 (state dimension).  The pre-fix ECEF filter was ~290x overconfident.")
