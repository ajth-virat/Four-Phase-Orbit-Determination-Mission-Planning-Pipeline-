# Orbit Mission Pipeline

**From a radar blip to a Mars trip, in four steps.**

This project is a small, honest simulation of a real space-engineering workflow:

1. A ground radar watches a satellite pass overhead.
2. Software works out exactly where the satellite is and how fast it is moving.
3. Software plans the tiny engine burns needed to put that satellite on a clean circular orbit.
4. Software then asks: *starting from that orbit, when should a spacecraft leave Earth to reach Mars, and how much fuel does it take?*

Each step writes a file, and the next step reads it. The steps really are chained: change the output of step 2 and the answers in steps 3 and 4 change too.

> **Read this first:** every satellite in this run is **synthetic** (made up by the code to look like typical low, medium and high orbits). No real radar data was used. The project shows that the *method* works, not that it works on real hardware. See [What this does and does not prove](#what-this-does-and-does-not-prove).

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py                 # live Celestrak TLEs (cache, then synthetic, if offline)
python run_pipeline.py --offline       # synthetic LEO/MEO/GEO demo set, no network
python validate_filter.py --runs 30    # Monte Carlo filter-consistency test
pytest -q tests                        # 20 checks against textbook values, SGP4 and astropy
```

Each phase also runs alone, for example `python phase2_ekf_od.py --data-dir data`.
Useful options:

- `--noise-scale 3` makes the radar three times noisier than the filter assumes.
- `--range-bias-m 100` adds an unmodelled range error.
- `--target-alt-km 500 --target-inc-deg 28.5` sets a different parking orbit.

# Files

| File | Job |
|---|---|
| `common.py` | Constants, time and frame conversions, orbit-element conversion, ground station |
| `phase1_radar_sim.py` | Creates satellites and noisy radar measurements |
| `phase2_ekf_od.py` | Kalman filter, consistency checks, satellite selection |
| `phase3_maneuver_planner.py` | Orbit elements with uncertainty, burn plan, fuel uncertainty |
| `phase4_mission_designer.py` | Porkchop plot, optimizer, self-check, departure burn |
| `lambert.py` | Lambert's problem solver |
| `ephemeris.py` | Earth and Mars positions |
| `validate_filter.py` | Repeated-run statistical test of the filter |
| `tests/test_pipeline.py` | 20 verification tests |

**Output files in `data/`:** `radar_observations.csv`, `truth_states.csv` (grading only), `test_set.csv`, `ekf_results.csv`, `od_summary.csv`, `selected_orbit.json`, `maneuver_plan.csv`, `parking_orbit.json`, `porkchop.csv`, `mission_trajectory.csv`, `mission_summary.json`, plus one overview image per phase.

---

## Who this is for

You do not need any space background. The next section explains the ideas in plain words, and the [glossary](#glossary) at the end defines every technical term used.

---

## The big idea in plain language

### Where is it, and where is it going?

To know an orbit, you need two things at one moment: the object's **position** (where it is) and its **velocity** (how fast, and in which direction). From those two, physics tells you the whole path, the same way that if you know where a thrown ball is and how it is moving, you can predict where it lands.

A radar does **not** measure velocity. It measures only:

- **Range**: how far away the object is.
- **Azimuth**: the compass direction to it (0° = north, 90° = east).
- **Elevation**: how high above the horizon it is.

Every measurement is slightly wrong (noise). So the job is to take many noisy position readings over a few minutes and *infer* a precise position and velocity.

### The tool that does this: a Kalman filter

A Kalman filter is a "predict, then correct" loop:

1. **Predict:** use the laws of gravity to guess where the satellite should be a few seconds from now.
2. **Compare:** look at the radar reading. How far off was the guess?
3. **Correct:** nudge the estimate toward the reading, trusting the radar more or less depending on how noisy it is.

Repeat for every reading. The estimate gets sharper over time. The "Extended" in **EKF** just means the filter handles the fact that the radar's measurements (angles and distance) are not simple straight-line functions of position.

Crucially, the filter also reports **how sure it is** (its *covariance*, an uncertainty estimate). A good filter is not just accurate; it is *honestly* uncertain, neither overconfident nor timid.

---

## The four phases

### Phase 1: Simulated radar (`phase1_radar_sim.py`)

We do not own a radar, so we fake one. The code creates 10 satellites (5 low, 3 medium, 2 high orbit), flies them with a standard orbit-prediction model (SGP4), and "measures" them from a ground station in Chennai with realistic noise (50 m in range, 0.05° in angles).

The perfect, noise-free positions are saved in a **separate file** that later phases may only use to *grade* themselves, never to make decisions. The code checks that the measurement file contains no truth columns.

**This run:** 10 satellites, 1,285 measurements in total.

| Orbit type | Height | Satellites | Typical tracking time |
|---|---|---|---|
| LEO (low Earth orbit) | ~500-1,500 km | 5 | 7.5-20.5 min |
| MEO (medium Earth orbit, like GPS) | ~20,000 km | 3 | 90 min |
| GEO (geostationary, hovers over one spot) | ~35,800 km | 2 | 90 min |

### Phase 2: Working out the orbit (`phase2_ekf_od.py`)

An Extended Kalman Filter runs on each of the 10 satellites. It starts from a rough guess built from just the first few measurements, then refines it. Its physics model is gravity from a spherical Earth plus the Earth's slight bulge at the equator (called J2).

It is then graded against the hidden truth, and it also grades itself with a truth-free consistency test (NIS, see glossary).

Finally, it picks **one** satellite to hand to Phase 3, using a rule that needs no truth (as in real life): among filters that pass the self-consistency test, take the one with the smallest claimed uncertainty.

**Results of this run (all 10 passed the truth-free consistency test):**

| Satellite | Type | Final position error | Final speed error | Filter's claimed position uncertainty |
|---|---|---|---|---|
| SYN-LEO-AJISAI-LIKE (**selected**) | LEO | **35 m** | **0.15 m/s** | 192 m |
| SYN-LEO-ISSLIKE | LEO | 410 m | 0.96 m/s | 198 m |
| SYN-LEO-1200-87 | LEO | 474 m | 1.18 m/s | 235 m |
| SYN-LEO-SSO-700 | LEO | 523 m | 1.64 m/s | 243 m |
| SYN-LEO-550-53 | LEO | 792 m | 2.39 m/s | 195 m |
| SYN-MEO-GPSLIKE-B | MEO | 760 m | 0.37 m/s | 754 m |
| SYN-MEO-GPSLIKE-A | MEO | 1,134 m | 0.44 m/s | 755 m |
| SYN-MEO-GPSLIKE-C | MEO | 2,444 m | 1.34 m/s | 756 m |
| SYN-GEO-74E | GEO | 1,847 m | 0.56 m/s | 1,911 m |
| SYN-GEO-93E | GEO | 1,801 m | 0.11 m/s | 1,854 m |

*Note:* "claimed uncertainty" is a combined (root-mean-square) figure across the three position axes, while "final error" is the straight-line miss distance at the last measurement.

**How to read this honestly:**

- For the selected satellite, the error shrinks from several kilometres at the start to tens of metres at the end. It was best because it had the longest, highest pass (peak elevation 83°, 20.5 minutes), which gives the filter the most geometry to work with.
- The "35 m" is the error at the final instant only. Averaged over the whole pass, the selected satellite's position error was about 1.1 km (RMS), because it started poorly.
- The geostationary satellites barely move across the sky, so the radar sees almost no change in angle. They are the hardest to pin down, and the ~1.8 km errors show it.
- **The honesty check is mixed, not perfect.** An ideal filter scores about 6 on the NEES consistency number (see glossary). This run's single-pass values range from 1.8 to 9.6. The selected satellite scored 2.4, meaning the filter was *too cautious* (it claimed ~190 m uncertainty when it was actually off by 35 m). Two MEO satellites scored about 9, meaning slightly *overconfident*. A single run is a noisy test. The separate Monte Carlo test (`validate_filter.py`) is the proper check and was not re-run for this report.

### Phase 3: Planning small burns (`phase3_maneuver_planner.py`)

It takes the orbit from Phase 2, converts it into the standard six orbit numbers (size, shape, tilt, and so on), and plans two small engine burns to make the orbit perfectly circular at its average height. It also draws thousands of random samples from the filter's uncertainty to see how much the fuel estimate could vary.

**Results of this run:**

| Quantity | Value |
|---|---|
| Semi-major axis (orbit size) | 7,868.24 km, uncertainty 0.07 km |
| Altitude | about 1,490 km |
| Inclination (tilt to the equator) | 49.999°, uncertainty 0.004° |
| Orbital period | about 115.8 min |
| Burn 1 (at lowest point, 4.6 min after epoch) | -2.87 m/s (slowing down) |
| Burn 2 (62.4 min later) | +2.88 m/s (speeding up) |
| Plane change | 0 m/s (none requested) |
| **Total** | **5.75 m/s, plus or minus 0.05 m/s** |

For scale, a walking pace is about 1.4 m/s. These burns are tiny because the orbit is already almost circular (eccentricity 0.0016).

The numbers in this step are passed on to Phase 4 as `parking_orbit.json`.

### Phase 4: Earth to Mars (`phase4_mission_designer.py`)

Planets move, so you cannot just aim at Mars; you must aim at where Mars *will be* when you arrive. The code:

1. Looks up where Earth and Mars are on any date.
2. For each departure date and flight duration, solves **Lambert's problem**: find the path around the Sun that connects the two positions in exactly that time.
3. Computes the **C3** (a measure of how hard you must throw the spacecraft out of Earth's gravity) and the total fuel need ("delta-v").
4. Draws a **porkchop plot**: a map of the cost for every date and duration. The cheapest region looks like a blob, and the contours resemble a pork chop.
5. Uses an optimizer to find the cheapest point, then double-checks it by actually flying the path forward numerically and measuring how far it misses Mars.

The departure burn starts from the **orbit that Phase 3 produced** (1,490 km up), not a generic assumed one.

**Results of this run:**

| Quantity | Value |
|---|---|
| Best departure | 2026-11-01 |
| Arrival at Mars | 2027-09-07 |
| Flight time | 310 days |
| C3 | 9.27 km²/s² |
| Speed needed beyond Earth's pull (v-infinity) | 3.04 km/s |
| Departure burn (from the 1,490 km orbit) | 3.40 km/s |
| Mars capture burn (into a 400 km circular orbit) | 2.04 km/s |
| **Total** | **5.44 km/s** |
| Width of the good launch window (C3 within +2 of best) | about 49 days |
| Cost of leaving 1 day late | about +0.02 km²/s² of C3 |
| Miss distance of the double-check flight | 0.0006 km (about 0.6 m) |
| Launch direction needed | 29.4° north of the equator; the orbit's 50° tilt covers this |

**How to read this honestly:**

- The ~0.6 m miss only proves the solver is internally consistent (the path it computes really does connect the two planets). It says nothing about how accurate the planet positions are.
- The "orbit tilt is OK" check compares tilts only. It does not check whether the orbit is rotated to the right orientation on launch day. A real mission would need to wait for, or pay for, that alignment, and this is not costed.
- 2026-11-01 sits in the real late-2026 Mars launch season, which the test suite also checks against an independent expectation.

---

## Why bother with this project?

- It shows an entire estimation-to-mission chain where each stage consumes the previous stage's *uncertain* output.
- It is built to be checked: every key piece is tested against textbook worked examples, an independent orbit propagator, or astropy.
- It documents its own earlier bugs (see below), which is the more valuable part.

---

## What was wrong in the first version (and the fix)

| # | Problem | Fix |
|---|---|---|
| 1 | The filter worked in the Earth-fixed rotating frame but used non-rotating physics, ignoring ~1 m/s² of apparent acceleration. It was about 290 times overconfident. | State now lives in a non-rotating frame; Earth's rotation appears only in the measurement model. |
| 2 | The filter's starting guess secretly read the true answer. | Starting guess is built from noisy measurements only; truth lives in a separate grading file. |
| 3 | Uncertainty was propagated with a crude approximation. | The exact sensitivity matrix is integrated along with the state. |
| 4 | Phase 3 was hard-coded to one real satellite. | It now reads Phase 2's output and reports uncertainty. Drag modelling was removed because it was not defensible. |
| 5 | Earth and Mars were in the wrong places (Earth off by ~103°, Mars by ~24°). | Positions now come from astropy's planetary ephemeris (an optional JPL kernel is supported). |
| 6 | The Lambert solver was hand-rolled and wrong (missed Mars by ~272 million km). | Replaced by the textbook universal-variable method, verified by a textbook example and by flying the answer. |
| 7 | Phase 4 ignored Phase 3 and assumed a 200 km orbit. | The departure burn uses the Phase 3 orbit radius. |
| 8 | A broken optimizer printed "complete"; a sensitivity step differentiated a proxy, not the real C3. | Both removed; sensitivity is now a finite difference of the real C3. |

---

## What this does and does not prove

**It does show:**

- The filter, the burn planner and the Mars solver each agree with independent references (textbook examples, SGP4, astropy).
- The chain of files really propagates: Phase 4's departure cost depends on Phase 3's orbit, which depends on Phase 2's estimate.

**It does not show:**

- **Real-radar performance.** The truth is the same orbit model that generated the data, and the noise is exactly the Gaussian noise the filter assumes. A good score proves the filter matches *this simulator*, not a real sensor with biases and surprises.
- **Robustness to small errors.** In an earlier stress test, a 300 m unmodelled range bias doubled LEO errors, yet the filters still passed the truth-free consistency test. That test is blind to small biases.
- **Real catalog behaviour.** All ten satellites here are synthetic. The code can pull live public orbit data (TLEs) from Celestrak, but this run did not use it.

## Other limitations

- **The "selected" satellite is largely a pass-geometry winner.** The selection rule picks the smallest claimed uncertainty, and that favours the longest, highest pass. It is a reasonable truth-free rule, but it is not choosing the "best orbit" in any deeper sense.
- **The parking orbit is whatever satellite was chosen.** A 1,490 km parking orbit is higher than a typical departure orbit (a few hundred km). That is fine as a demonstration of the hand-off but is not a realistic launch scenario.
- **The two Phase 3 burns come from the orbit's real, estimated shape, not from noise.** The estimate's eccentricity (0.0016) is about 100 times its stated uncertainty, so the 2.9 m/s burns are driven by the estimated orbit shape, not by measurement error. The estimated value is an instantaneous ("osculating") shape that includes short-period wobble from the Earth's bulge, so it can differ from a long-term average shape. (The warning text drawn on the Phase 3 plot says the opposite; that text is a general caution and does not apply to this run.)
- **Physics is simplified.** Orbit model: gravity plus the equatorial bulge only (no drag, solar pressure, or higher gravity terms). Transfer model: "patched conics" with instant burns (no gravity losses, no mid-course corrections). Mars capture assumes a circular 400 km orbit, which is an upper bound on cost.
- **Planet-position accuracy is porkchop-grade.** The built-in ephemeris is good to roughly 1,000 km at Mars. Use a JPL kernel (`--kernel de440s.bsp`) for design-grade work.
- **Single-revolution transfers only.** No multi-loop or gravity-assist options.
- **Frame simplification.** Polar motion and nutation are ignored (about a 10 m effect, small next to the 50 m range noise).

## Glossary

- **Orbit:** the path an object follows around a planet under gravity.
- **LEO / MEO / GEO:** low (hundreds to ~2,000 km), medium (e.g. GPS, ~20,000 km) and geostationary (~35,800 km, matches Earth's rotation so it appears fixed in the sky) Earth orbits.
- **TLE:** a compact public text format describing a satellite's orbit.
- **SGP4:** the standard algorithm that turns a TLE into positions over time.
- **Range, azimuth, elevation:** distance, compass bearing and height above the horizon, as seen from a ground station.
- **ECI / ECEF:** a coordinate frame that does not rotate with Earth (inertial) versus one that does (Earth-fixed).
- **EKF (Extended Kalman Filter):** a predict-and-correct estimator for nonlinear problems.
- **Covariance:** the filter's own estimate of how uncertain it is.
- **NIS:** a check, needing no truth, that the filter's surprises match its stated uncertainty.
- **NEES:** the same idea but using the known truth. Its average should equal the number of state variables, which is 6 here.
- **J2:** the leading effect of Earth being slightly squashed at the poles.
- **Delta-v:** the change in speed a maneuver needs; the universal "fuel cost" measure in spaceflight.
- **Semi-major axis (a):** half the long width of an orbit, a measure of its size.
- **Eccentricity (e):** how stretched an orbit is. 0 is a perfect circle.
- **Inclination (i):** the tilt of an orbit relative to Earth's equator.
- **Parking orbit:** a temporary orbit a spacecraft waits in before departing.
- **Hohmann transfer:** the standard two-burn way to move between circular orbits.
- **Lambert's problem:** given two positions and a travel time, find the path that connects them.
- **C3:** departure energy; smaller is cheaper.
- **v-infinity:** the speed relative to Earth once the spacecraft has fully escaped its gravity.
- **Porkchop plot:** a map of mission cost versus departure date and flight time.
- **Launch window:** the span of dates when a transfer is reasonably cheap.
- **Patched conics:** a simplification that treats each planet's gravity and the Sun's gravity one at a time.
- **Ephemeris:** a table or model of where the planets are at any date.
