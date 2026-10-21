"""
run_pipeline.py - run all four phases in order
==============================================
    python run_pipeline.py                 live TLE if reachable, else cache, else synthetic
    python run_pipeline.py --offline       synthetic satellites (no network)
    python run_pipeline.py --offline --start 2026-10-21T04:30:00

Data flow (every arrow is a file in --data-dir):
    Phase 1  radar_observations.csv (noisy only) + truth_states.csv (grading only)
      |
    Phase 2  EKF on all test satellites -> picks ONE by a truth-free rule -> selected_orbit.json
      |
    Phase 3  plans burns to a circular parking orbit            -> parking_orbit.json
      |
    Phase 4  Earth->Mars Lambert / porkchop / optimiser, departing from that parking orbit
"""

import argparse

import phase1_radar_sim as P1
import phase2_ekf_od as P2
import phase3_maneuver_planner as P3
import phase4_mission_designer as P4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--start", default=None, help="UTC ISO start of the 24 h pass search (default: now)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--range-bias-m", type=float, default=0.0)
    ap.add_argument("--target-alt-km", type=float, default=None)
    ap.add_argument("--target-inc-deg", type=float, default=None)
    ap.add_argument("--dep-start", default="2026-08-15")
    ap.add_argument("--dep-span-days", type=float, default=240)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    P1.run(a.data_dir, a.offline, a.seed, a.start, a.noise_scale, a.range_bias_m, a.show)
    P2.run(a.data_dir, a.show)
    P3.run(a.data_dir, a.target_alt_km, a.target_inc_deg, a.show)
    s = P4.run(a.data_dir, a.dep_start, a.dep_span_days, show=a.show)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  parking orbit from {s['parking_source']} at {s['parking_alt_km']:.0f} km")
    print(f"  depart {s['departure']}, arrive {s['arrival']} ({s['tof_days']:.0f} d), "
          f"C3 {s['c3_km2s2']:.2f}, total dv {s['dv_total_kms']:.2f} km/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
