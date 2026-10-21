"""
Verification suite.  Every test checks the code against something INDEPENDENT of the
code: a textbook worked example, SGP4 truth, astropy, or a physical invariant.

    pytest -q            (the Monte Carlo test takes ~15 s)
"""
import numpy as np
import pandas as pd
import pytest

import common as C
import ephemeris as EPH
import lambert as LB
import phase1_radar_sim as P1
import phase2_ekf_od as P2
import phase3_maneuver_planner as P3
import phase4_mission_designer as P4
import validate_filter as VF

START = "2026-10-21T04:30:00"


# ------------------------------------------------------------------ shared fixture
@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    obs, truth, meta = P1.run(str(d), offline=True, start=START)
    return dict(dir=str(d), obs=obs, truth=truth, meta=meta)


# ------------------------------------------------------------------ frames
def test_eci_ecef_roundtrip():
    r = np.array([6.8e6, 1.2e6, 2.1e6]); v = np.array([-1.1e3, 7.0e3, 1.5e3])
    g = 1.234
    r2, v2 = C.ecef_to_eci_state(*C.eci_to_ecef_state(r, v, g), g)
    assert np.allclose(r, r2) and np.allclose(v, v2)


def test_ecef_velocity_includes_rotation():
    # a point fixed in ECI must appear to move at omega*r in ECEF
    r = np.array([7e6, 0, 0]); v = np.zeros(3)
    _, v_ecef = C.eci_to_ecef_state(r, v, 0.0)
    assert abs(np.linalg.norm(v_ecef) - C.OMEGA_E * 7e6) < 1e-6


# ------------------------------------------------------------------ textbook checks
def test_keplerian_vallado_example_2_5():
    r = np.array([6524.834, 6862.875, 6448.296]) * 1e3
    v = np.array([4.901327, 5.533756, -1.976341]) * 1e3
    el = C.state_to_keplerian(r, v)
    assert el["a"] / 1e3 == pytest.approx(36127.343, abs=0.5)
    assert el["e"] == pytest.approx(0.832853, abs=1e-5)
    assert el["i_deg"] == pytest.approx(87.870, abs=1e-2)
    assert el["raan_deg"] == pytest.approx(227.89, abs=1e-2)
    assert el["aop_deg"] == pytest.approx(53.38, abs=1e-2)
    assert el["nu_deg"] == pytest.approx(92.335, abs=1e-2)


def test_hohmann_vallado_example_6_1():
    h = P3.hohmann_circular((6378.137 + 191.34411) * 1e3, 42164e3)
    assert h["total"] / 1e3 == pytest.approx(3.935, abs=0.005)
    assert h["tof_s"] / 3600 == pytest.approx(5.25, abs=0.02)


def test_lambert_curtis_example_5_2():
    v1, v2 = LB.lambert_universal([5000, 10000, 2100], [-14600, 2500, 7000], 3600.0, 398600.0)
    assert np.allclose(v1, [-5.9925, 1.9254, 3.2456], atol=2e-4)
    assert np.allclose(v2, [-3.3125, -4.1966, -0.38529], atol=2e-4)


# ------------------------------------------------------------------ ephemeris + Mars
@pytest.fixture(scope="module")
def eph():
    return EPH.Ephemeris(2461200, 2461900)


def test_ephemeris_geometry(eph):
    for jd in np.linspace(2461300, 2461800, 9):
        re, _ = eph.state("earth", jd); rm, _ = eph.state("mars", jd)
        assert 0.98 < np.linalg.norm(re) / EPH.AU < 1.02
        assert 1.37 < np.linalg.norm(rm) / EPH.AU < 1.67
        assert abs(re[2]) / EPH.AU < 1e-3          # Earth is in the ecliptic plane


def test_ephemeris_earth_velocity_matches_astropy(eph):
    jd = 2461333.37
    _, v = eph.state("earth", jd)
    _, va = EPH._astropy_states("earth", [jd])
    assert np.linalg.norm(v - va[0]) < 0.05        # m/s


def test_lambert_hits_mars_by_propagation(eph):
    """The test the OLD solver failed in 60/60 cases (median miss 272 million km)."""
    worst = 0.0
    for dep in np.linspace(2461300, 2461420, 5):
        for tof in (180, 250, 300, 350):
            t = P4.transfer(eph, dep, tof, C.RE_M + 500e3, 3.8e6)
            assert t is not None
            miss, _ = P4.miss_distance_km(eph, t, tof, dep)
            worst = max(worst, miss)
    assert worst < 100.0                           # km (Mars is ~1e8 km away)


def test_mars_window_matches_known_2026_opportunity(eph):
    """Independent expectation: best C3 ~9 km2/s2, late Oct - Nov 2026, TOF ~ 290 d."""
    jds = np.arange(2461290, 2461400, 2.0)
    best = min((P4.transfer(eph, j, tf, C.RE_M + 500e3, 3.8e6)["c3"], j, tf)
               for j in jds for tf in range(240, 340, 10)
               if P4.transfer(eph, j, tf, C.RE_M + 500e3, 3.8e6))
    assert 8.0 < best[0] < 11.0
    assert 2461340 < best[1] < 2461360             # ~ 25 Oct - 14 Nov 2026


# ------------------------------------------------------------------ Phase 1
def test_observations_contain_no_truth(sim):
    assert not [c for c in sim["obs"].columns if "true" in c or c in ("x_m", "y_m", "z_m")]


def test_noise_level_matches_settings(sim):
    o = sim["obs"].set_index(["sat_name", "elapsed_sec"])
    t = sim["truth"].set_index(["sat_name", "elapsed_sec"]).loc[o.index]
    assert np.std(o["range_m"] - t["true_range_m"]) == pytest.approx(C.SIGMA_RANGE_M, rel=0.15)
    d_el = o["elevation_deg"] - t["true_el_deg"]
    assert np.std(d_el) == pytest.approx(C.SIGMA_EL_DEG, rel=0.15)


def test_test_set_covers_orbit_classes(sim):
    assert set(sim["meta"]["orbit_class"]) == {"LEO", "MEO", "GEO"}


# ------------------------------------------------------------------ Phase 2
def test_measurement_jacobian_matches_numerical():
    st = C.Station()
    x = np.array([5.2e6, 3.9e6, 3.1e6, -3e3, 4e3, 5e3]); jd = 2461335.02
    h, H = P2.measurement(x, jd, st)
    Hn = np.zeros((3, 6))
    for k in range(3):
        d = np.zeros(6); d[k] = 1.0
        dh = (P2.measurement(x + d, jd, st, False) - P2.measurement(x - d, jd, st, False)) / 2
        dh[1] = (dh[1] + np.pi) % (2 * np.pi) - np.pi
        Hn[:, k] = dh
    assert np.allclose(H[:, :3], Hn[:, :3], rtol=1e-5, atol=1e-12)


def test_dynamics_agree_with_sgp4_truth(sim):
    """The old ECEF filter was 36 m / 7 m/s wrong after ONE 10 s step."""
    g = sim["truth"][sim["truth"].sat_name == "SYN-LEO-AJISAI-LIKE"].reset_index(drop=True)
    cols = ["x_m", "y_m", "z_m", "vx_ms", "vy_ms", "vz_ms"]
    x1, _ = P2.propagate(g.loc[10, cols].values.astype(float), 10.0)
    xt = g.loc[11, cols].values.astype(float)
    assert np.linalg.norm(x1[:3] - xt[:3]) < 1.0
    assert np.linalg.norm(x1[3:] - xt[3:]) < 0.01


def test_stm_matches_finite_difference():
    x0 = np.array([6.9e6, 1.0e6, 2.0e6, -1.5e3, 6.5e3, 3.0e3])
    _, Phi = P2.propagate(x0, 30.0)
    for k, eps in ((0, 10.0), (4, 0.01)):
        d = np.zeros(6); d[k] = eps
        col = (P2.propagate(x0 + d, 30.0)[0] - P2.propagate(x0 - d, 30.0)[0]) / (2 * eps)
        assert np.allclose(Phi[:, k], col, rtol=1e-4, atol=1e-6)


def test_ekf_accuracy_and_no_truth_needed(sim):
    results, summary = P2.run_all(sim["obs"], sim["truth"])
    assert summary["nis_consistent"].all()
    leo = summary[summary.orbit_class == "LEO"]
    assert (leo["final_pos_err_m"] < 3000).all()       # old filter: 107,000 m
    assert (leo["final_vel_err_ms"] < 10).all()        # old filter: 364 m/s
    assert (summary["final_pos_err_m"] < 8000).all()
    # estimate must still run on a table with no truth columns at all
    P2.run_all(sim["obs"], None)


def test_ekf_is_statistically_consistent_monte_carlo(sim):
    """Average NEES over independent noise draws must be ~6 (= state dimension)."""
    tr = sim["truth"][sim["truth"].sat_name == "SYN-LEO-1200-87"]
    r = VF.monte_carlo(tr, runs=12)
    assert 4.0 < r["mean_nees"] < 9.0                  # old filter: ~290x too confident


def test_selection_never_uses_truth(sim):
    _, summary = P2.run_all(sim["obs"], None)
    name, _ = P2.select_satellite(summary)
    assert name in set(summary.sat_name)


# ------------------------------------------------------------------ Phases 3 + 4 and the hand-off
def test_full_chain_and_handoff(sim):
    P2.run(sim["dir"])
    park = P3.run(sim["dir"], target_alt_km=500.0, target_inc_deg=None)
    assert park["alt_park_km"] == pytest.approx(500.0)
    s = P4.run(sim["dir"], dep_start="2026-09-15", dep_span_days=120, tof_min=200, tof_max=380,
               n_dep=30, n_tof=30)
    # Phase 4 really used Phase 3's orbit
    assert s["parking_alt_km"] == pytest.approx(500.0)
    assert s["lambert_miss_km"] < 100.0
    assert 8.0 < s["c3_km2s2"] < 12.0
    assert 3.0 < s["dv_departure_kms"] < 4.0


def test_departure_burn_depends_on_parking_orbit(eph):
    lo = P4.transfer(eph, 2461345.0, 290, C.RE_M + 300e3, 3.8e6)["dv_dep"]
    hi = P4.transfer(eph, 2461345.0, 290, 42164e3, 3.8e6)["dv_dep"]
    assert lo > hi + 0.5       # leaving from GEO is much cheaper than from LEO
