# Orbit Mission Pipeline (fixed) - radar -> EKF -> maneuvers -> Earth-Mars transfer

One chain, four phases. Every arrow is a file, and each phase really uses the previous one.

```
Phase 1  radar_observations.csv (noisy only) + truth_states.csv (grading only)
   |
Phase 2  EKF on every test satellite -> picks ONE by a truth-free rule -> selected_orbit.json
   |
Phase 3  burns to a circular parking orbit, with delta-v uncertainty  -> parking_orbit.json
   |
Phase 4  Lambert + porkchop + optimiser, departing from THAT parking orbit
```

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py                 # live Celestrak TLEs (cache, then synthetic, if offline)
python run_pipeline.py --offline       # synthetic LEO/MEO/GEO demo set, no network
python validate_filter.py --runs 30    # Monte Carlo filter-consistency test
pytest -q tests                        # 20 checks against textbook values, SGP4 and astropy
```

Each phase also runs alone: `python phase2_ekf_od.py --data-dir data`, etc.
Stress knobs: `--range-bias-m 100`, `--noise-scale 3`. Parking orbit: `--target-alt-km 500 --target-inc-deg 28.5`.

## Files

| file | job |
|---|---|
| `common.py` | constants, ONE GMST, frame transforms, Keplerian conversion, `Station` |
| `phase1_radar_sim.py` | TLE -> SGP4 -> noisy range/az/el for a LEO/MEO/GEO test set |
| `phase2_ekf_od.py` | ECI EKF, Herrick-Gibbs seed, NIS/NEES, truth-free selection |
| `phase3_maneuver_planner.py` | Keplerian elements +/- sigma, Hohmann-type burns, delta-v uncertainty |
| `phase4_mission_designer.py` | porkchop, optimiser, propagation self-check, departure burn |
| `lambert.py`, `ephemeris.py` | universal-variable Lambert; Earth/Mars states |
| `validate_filter.py` | Monte Carlo NEES test |
| `tests/test_pipeline.py` | verification suite |

## Results (offline demo set, synthetic satellites, 2026-10-21)

| check | result |
|---|---|
| EKF one-step dynamics error vs SGP4 | 0.07 m (old ECEF filter: 36 m) |
| EKF final position error, LEO | 22 - 771 m (old filter: ~107 km) |
| EKF final velocity error, LEO | 0.1 - 2.2 m/s (old filter: 364 m/s) |
| Monte Carlo mean NEES, 30 runs (expected 6.0, 95% band 4.8-7.3) | LEO 5.96, MEO 6.28, GEO 6.25 |
| Phase 3 semi-major axis of selected sat | 7078.46 +/- 0.21 km (truth 7078.69, error -1.2 sigma) |
| Lambert vs Curtis Example 5.2 | matches to 4-5 digits |
| Lambert arrival miss vs Mars (propagated) | ~0 km (old solver: median 272 million km) |
| Best Mars window (min C3) | C3 9.19 km2/s2, depart ~1 Nov 2026, TOF ~291 d |
| Chosen (min total delta-v) | depart 2026-11-01, arrive 2027-09-07, C3 9.27, 5.58 km/s total |

## What was wrong before, and the fix

1. **EKF in a rotating frame (ECEF) with inertial physics.** It left out ~1 m/s2 of Coriolis/centrifugal acceleration and was ~290x overconfident. Fixed: the state lives in ECI (TEME); Earth rotation appears only in the measurement model.
2. **EKF seed read `true_*` columns.** It could not run on real data. Fixed: Herrick-Gibbs on noisy measurements; truth lives in a separate file used only for grading. `run_all` asserts the observation file has no truth columns.
3. **Crude covariance prediction (`F = I + A dt`).** Fixed: the state-transition matrix is integrated with the state.
4. **Phase 3 hard-coded to AJISAI**, fed by a broken EKF. Fixed: reads the estimated orbit, reports uncertainty, drag removed (the old exponential atmosphere was not defensible).
5. **Wrong planet positions** (longitude of perihelion left out: Earth 103 deg off, Mars 24 deg off). Fixed: astropy built-in ephemeris, optional SPICE kernel.
6. **Lambert solver was wrong** (missed a textbook case by 0.1 km/s). Fixed: universal-variable solver verified by textbook example and by propagation.
7. **Phase 4 ignored Phase 3** and assumed a 200 km orbit. Fixed: the departure burn uses the Phase 3 parking-orbit radius.
8. **CasADi failed but printed "complete"; JAX differentiated a Hohmann proxy.** Both removed. Sensitivity is a finite difference of the real C3; the window width is read from the porkchop.

## How to explain it (short answers)

- **Why simulate the radar?** No radar is available, so measurements are synthesised from a real orbit with known noise. Keeping the truth lets the estimator be graded.
- **What does the EKF extract?** The full 6-D state (position and velocity) from range/az/el. Velocity is never measured. Filtering noise is a side effect.
- **Why "extended"?** Range/az/el is nonlinear in position, so the filter linearises with the Jacobian each step.
- **Where do passes come from?** The TLE gives the orbit, SGP4 propagates it, and the pass is the stretch above the 5 deg elevation mask.
- **How do you know the filter is honest?** NIS (no truth needed) and NEES over many noise draws: mean NEES ~6 = state dimension.
- **Why doesn't the TLE make Phases 1-2 pointless?** A TLE is a catalog product with ~1 km error and no covariance. Phases 1-2 show the raw-measurement-to-state chain with uncertainty.

## Limitations (say these first)

- **Circularity.** Truth is SGP4 and the noise is Gaussian at exactly the level the filter assumes. A good result proves the filter matches this simulator, not real radar.
- **NIS misses small biases.** A 300 m unmodelled range bias raised the LEO error from 215 m to 502 m and the semi-major-axis error from 0.24 km to 1.65 km, yet all 10 filters still passed NIS. Noise 3x larger than assumed was caught (0/10 passed).
- **One pass gives a poor eccentricity**, so the periapsis burn location is uncertain, and the "circularisation" delta-v is mostly orbit-determination uncertainty.
- **Dynamics are two-body + J2.** No drag, SRP, J3+ or third bodies; the process noise absorbs them. Fine over a pass, not over days.
- **Patched conics, impulsive burns.** No gravity losses, launch-asymptote plane matching, or mid-course corrections. Mars capture is an upper bound (circular 400 km).
- **Ephemeris accuracy** is porkchop-grade (Mars position uncertain at about the 1000 km level with astropy's built-in model). Use a SPICE kernel (`--kernel de440s.bsp`) for design-grade work.
- **Single-revolution Lambert only**; no multi-revolution or gravity-assist options.
- The shipped `data/` folder comes from SYNTHETIC satellites. Live Celestrak was not reachable where this was built, so the live-download path is untested end to end (TLE parsing and the pass/test-set code were tested on a real-format TLE).
