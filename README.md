# Orbital Mechanics & Mission Design Suite
## A 4-Phase End-to-End Astrodynamics Pipeline

A complete Python framework for satellite tracking, orbit determination, maneuver planning, and interplanetary mission design. This suite integrates radar simulation, state estimation, propulsive maneuvering, and trajectory optimization into a unified workflow.

---

## Project Overview

This is a 4-phase project that models the real workflow of satellite operations and mission planning:

```
Phase 1              Phase 2              Phase 3              Phase 4
Radar Sim      →     Orbit Est.     →     Maneuvers      →     Mission Design
(noise)              (EKF)                (Hohmann)            (Lambert + Opt.)
```

### The Workflow

1. **Ground radar observes a satellite** → noisy range/azimuth/elevation measurements
2. **Kalman filter cleans the data** → reconstructs full 6D state vector (position + velocity)
3. **Planner designs orbit corrections** → computes delta-v for station-keeping
4. **Mission designer extends to interplanetary** → Earth-Mars transfer with global optimization

---

## Phase 1: Ground Radar Simulator
**File:** `ground_radar_simulator.py`

### What It Does
- Fetches live TLEs for 150+ active satellites from Celestrak (no API key needed)
- Identifies the best upcoming pass over your location
- Simulates realistic ground-based radar tracking
- Injects Gaussian noise into range, azimuth, elevation measurements
- **Output:** `radar_observations.csv` (Project 2's input)

### Key Features
- **Live TLE data**: Uses Celestrak free endpoints
- **SGP4 propagation**: Accurate satellite position prediction
- **Configurable site location**: Set your ground station coordinates
- **Elevation masking**: Ignores satellites below horizon
- **Realistic noise**: Range ±50m, azimuth ±0.05°, elevation ±0.05°

### Configuration (Edit These)
```python
GS_LAT_DEG = 13.0827      # Ground station latitude
GS_LON_DEG = 80.2707      # Ground station longitude
GS_ALT_M = 6.0            # Altitude in metres
SIM_DURATION_MIN = 100    # How long to simulate
NOISE_RANGE_M = 50.0      # Range measurement noise
```

### Libraries Used
- `numpy`, `pandas`, `matplotlib` — data handling & plotting
- `sgp4` — TLE-based satellite propagation
- `requests` — live TLE fetching from Celestrak

---

## Phase 2: Orbit Determination via Extended Kalman Filter
**File:** `ekf_orbit_determination.py`

### What It Does
Takes noisy radar observations (range, azimuth, elevation) and reconstructs the satellite's **full state vector**:
```
x = [X, Y, Z, Vx, Vy, Vz]  (position + velocity in ECEF)
```

This solves an **inverse problem** — recovering 6D state from 3D measurements using Gaussian optimal estimation.

### The EKF Algorithm
**Two-step recursive process each timestep:**

1. **PREDICT**: Propagate state forward using two-body gravity dynamics
   ```
   x_pred = f(x)                  # integrate equations of motion
   P_pred = F @ P @ F.T + Q       # covariance propagation
   ```

2. **UPDATE**: Correct prediction with new noisy measurement
   ```
   K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)  # Kalman Gain
   x_new = x_pred + K @ (z - h(x_pred))          # state update
   P_new = (I - K @ H) @ P_pred                   # covariance update
   ```

### Why "Extended"?
The measurement model `h(x)` (position → range/azimuth/elevation) is **nonlinear**. The EKF linearizes it at each step using the Jacobian H. Standard KF assumes linearity.

### Key Features
- **Two-body gravity model** with realistic error dynamics
- **Jacobian-based linearization** for nonlinear measurements
- **Process & measurement noise tuning** for convergence
- **State and covariance output** for downstream analysis
- **Output:** `ekf_results.csv` (Project 3's input)

### Libraries Used
- `numpy`, `scipy` — linear algebra & ODE integration
- `pandas`, `matplotlib` — results handling & plotting

---

## Phase 3: Station-Keeping & Maneuver Planner
**File:** `maneuver_planner.py`

### What It Does
Takes EKF-estimated state vectors and plans **propulsive maneuvers** to maintain/correct the orbit:

1. Converts ECEF state → Keplerian orbital elements (Vallado Algorithm 9)
2. Propagates orbit under J2 perturbation + atmospheric drag
3. Plans **Hohmann transfer** to restore semi-major axis
4. Plans **inclination correction** maneuver at ascending node
5. Outputs full delta-v budget and maneuver windows
6. **Output:** `maneuver_plan.csv` (Project 4's input)

### The Math

**Keplerian Elements** (from Vallado):
```
h = r × v                          (angular momentum)
e = (1/μ)(v×h) - r/|r|             (eccentricity)
i = arccos(h_z/|h|)                (inclination)
a = -μ / (2E)  where E = v²/2 - μ/r (semi-major axis)
```

**Hohmann Transfer** (coplanar altitude change):
```
a_t = (a1 + a2) / 2                (transfer SMA)
Δv1 = √(μ/a1) * (√(2a2/(a1+a2)) - 1)   (raise orbit)
Δv2 = √(μ/a2) * (1 - √(2a1/(a1+a2)))   (circularize)
```

**Inclination Change** (at ascending node):
```
Δv = 2 * v_circ * sin(Δi/2)        (delta-v for plane change)
```

### Key Features
- **J2 perturbation modeling** (oblateness of Earth)
- **Atmospheric drag** using exponential model
- **Automatic maneuver window detection**
- **Delta-v accounting** for station-keeping budget
- **Orbital element validation** with robust error handling

### Libraries Used
- `poliastro` — orbit propagation & Hohmann transfers
- `spiceypy` — kernel-based coordinate transformations
- `astropy` — time handling & coordinate systems
- `numpy`, `scipy`, `matplotlib` — numerics & plotting

---

## Phase 4: Interplanetary Mission Designer
**File:** `mission_designer.py`

### What It Does
Scales maneuver planning to **interplanetary scope** — designs optimized Earth-Mars transfer missions:

1. Loads maneuver plan from Project 3 (station-keeping context)
2. Computes **porkchop plot** via Lambert solver grid
3. Uses **pygmo global optimization** to find best launch window
4. Computes **launch window sensitivity** with JAX autodiff
5. Refines with **CasADi NLP** for high-precision trajectory
6. **Outputs:** `mission_trajectory.csv` + `mission_plots.png`

### The Math

**Lambert's Problem** (Izzo 2015):
```
Given: r1 (departure), r2 (arrival), TOF (time of flight)
Find: v1, v2 (velocity vectors of connecting conic)

Solution: Universal variable x iterated until t(x) = TOF
```

**C3 (Characteristic Energy)**:
```
C3 = |v_∞_dep|²   (km²/s²)
v_∞ = v_transfer - v_planet    (velocity relative to planet)
```

**Porkchop Plot**:
```
Grid over (departure_time, TOF) → compute C3 for each pair
Contours show launch windows; shape resembles porkchop cut
Minimum = optimal launch date and travel time
```

### Optimization Roles

- **JAX**: Autodiff through C3 function → ∂C3/∂t_dep (sensitivity analysis)
- **pygmo (DE1220)**: Global optimization over thousands of (t_dep, TOF) pairs
- **CasADi NLP**: Refine pygmo result + low-thrust trajectory optimal control

### Key Features
- **Multi-objective porkchop plot** (C3 vs date vs duration)
- **Global optimization** without local minima traps
- **Sensitivity quantification** (launch window robustness)
- **Low-thrust trajectory support** (CasADi OCP)
- **Automatic best-window detection**

### Libraries Used
- `jax` — automatic differentiation (sensitivity)
- `pygmo` — global population-based optimization
- `casadi` — nonlinear programming & optimal control
- `spiceypy` — ephemeris & coordinate frames
- `poliastro` — orbital mechanics
- `numpy`, `pandas`, `matplotlib` — core numerics & plots

---

## Installation

### Requirements
- Python 3.9+
- pip

### Quick Install
```bash
pip install numpy pandas matplotlib scipy astropy
pip install poliastro spiceypy sgp4
pip install pygmo casadi jax
pip install requests
```

### Optional (for advanced features)
```bash
pip install numba  # faster numerical loops
```

---

## File Structure
```
.
├── ground_radar_simulator.py       # Phase 1: Radar observations
├── ekf_orbit_determination.py      # Phase 2: State estimation
├── maneuver_planner.py             # Phase 3: Orbit corrections
├── mission_designer.py             # Phase 4: Interplanetary design
├── radar_observations.csv           # Phase 1 output → Phase 2 input
├── ekf_results.csv                  # Phase 2 output → Phase 3 input
├── maneuver_plan.csv                # Phase 3 output → Phase 4 input
├── mission_trajectory.csv           # Phase 4 final output
├── radar_simulation.png             # Phase 1 plots
├── ekf_plots.png                    # Phase 2 plots
├── maneuver_plots.png               # Phase 3 plots
└── mission_plots.png                # Phase 4 plots
```

---

## Quick Start

### Step 1: Run Ground Radar Simulator
```bash
python ground_radar_simulator.py
```
**Output:** `radar_observations.csv` + `radar_simulation.png`

Edit `GS_LAT_DEG` and `GS_LON_DEG` to set your location first.

### Step 2: Run Orbit Determination
```bash
python ekf_orbit_determination.py
```
**Input:** `radar_observations.csv`  
**Output:** `ekf_results.csv` + `ekf_plots.png`

### Step 3: Run Maneuver Planner
```bash
python maneuver_planner.py
```
**Input:** `ekf_results.csv`  
**Output:** `maneuver_plan.csv` + `maneuver_plots.png`

### Step 4: Run Mission Designer
```bash
python mission_designer.py
```
**Input:** `maneuver_plan.csv` + `ekf_results.csv`  
**Output:** `mission_trajectory.csv` + `mission_plots.png`

---

## Configuration

Each phase has a **CONFIGURATION** section at the top. Edit these to customize behavior:

### Phase 1 (Radar)
```python
GS_LAT_DEG = 13.0827       # Your latitude
GS_LON_DEG = 80.2707       # Your longitude
NOISE_RANGE_M = 50.0       # Range noise (metres)
NOISE_AZ_DEG = 0.05        # Azimuth noise (degrees)
```

### Phase 2 (EKF)
```python
Q_SCALE = 0.1              # Process noise intensity
R_RANGE = 50.0             # Range measurement std dev
R_ANGLE = 0.05             # Angle measurement std dev
```

### Phase 3 (Maneuver)
```python
TARGET_SMA_M = 6800000     # Target semi-major axis
TARGET_INC_DEG = 51.6      # Target inclination
```

### Phase 4 (Mission)
```python
DEPARTURE_JD_START = ...   # Mission start window
DEPARTURE_JD_END = ...     # Mission end window
TOF_MIN_DAYS = 200         # Minimum flight time
TOF_MAX_DAYS = 400         # Maximum flight time
```

---

## Mathematical Background

### Orbital Mechanics Fundamentals

**Two-Body Problem:**
The fundamental equation of motion under gravitational attraction:
```
d²r/dt² = -μ/|r|³ * r
```
where μ = GM (gravitational parameter), r = position vector.

**Keplerian Elements:**
Six parameters describing an orbit:
- `a` — semi-major axis (size)
- `e` — eccentricity (shape: 0=circle, 1=parabola)
- `i` — inclination (tilt)
- `Ω` — right ascension of ascending node (longitude of node)
- `ω` — argument of perigee (rotation within orbit plane)
- `ν` — true anomaly (current position in orbit)

**State Vector (Cartesian):**
Alternative representation as position + velocity:
```
x = [X, Y, Z, Vx, Vy, Vz]  (6 dimensions)
```

### Kalman Filtering

The Kalman Filter is an optimal recursive estimator for linear systems with Gaussian noise. The Extended Kalman Filter (EKF) extends this to **nonlinear** systems by linearizing at each timestep.

**Why optimal?** It minimizes the mean-square error given the assumed noise model.

### Lambert's Problem

Classical problem: given two positions and a transfer time, find the unique elliptical (or hyperbolic) orbit connecting them. Solved iteratively using the universal variable formulation.

### Porkchop Plots

Contour plot of C3 (launch energy) vs. (departure date, time of flight). Reveals the **launch window** — the optimal date and trajectory duration for a mission. Named for the characteristic shape.

---

## Example Output

### Phase 1: Radar Simulation
```
Satellite: ISS (NORAD ID 25544)
Ground Station: 13.08°N, 80.27°E
Pass Duration: 100 minutes
Timestep: 10 seconds
Observations: 600 samples
CSV columns: timestamp, range_m, azimuth_deg, elevation_deg, range_noise_m, ...
```

### Phase 2: EKF Results
```
Initial covariance: 10 km, 100 m/s (high uncertainty)
After filtering: 0.5 km, 1 m/s (refined estimate)
CSV columns: time, X_m, Y_m, Z_m, Vx, Vy, Vz, P_diag (covariance)
```

### Phase 3: Maneuver Plan
```
SMA Error: -5 km → Hohmann transfer Δv1 + Δv2 = 120 m/s
Inc Error: +0.2° → Inclination change Δv = 80 m/s
Total Budget: ~200 m/s
Maneuver windows: [timestamp_1, timestamp_2, ...]
CSV: maneuver_type, epoch, dv_ms, duration_sec, ...
```

### Phase 4: Mission Trajectory
```
Earth → Mars Transfer
Departure: 2024-07-15 (example)
Arrival: 2025-02-10 (example)
Flight Time: 210 days
Minimum C3: 4.2 km²/s²
Sensitivity ∂C3/∂t: 0.015 km²/s² per day
```

---

## Troubleshooting

### Celestrak Not Responding (Phase 1)
The TLE fetch requires internet. Check your connection:
```bash
curl https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle
```

### EKF Divergence (Phase 2)
Covariance grows instead of shrinking? Reduce `Q_SCALE` or increase `R_*` (trust measurements more).

### No Feasible Maneuvers (Phase 3)
If delta-v is NaN, check:
- State vector sanity (radius > Earth radius)
- Keplerian element bounds (a > 0, 0 ≤ e < 1)

### Lambert Solver Failure (Phase 4)
Some (t_dep, TOF) pairs have no solution. The grid search will skip these. If too many fail, increase TOF range.

---

## References

- **Vallado, Curtis, Prussing** — *Fundamentals of Astrodynamics and Applications* (4th ed.)
- **Izzo, Dario** — "Lambert's Problem: A Fast and Robust Solver" (2015)
- **Welch & Bishop** — "An Introduction to the Kalman Filter" (2006)
- **Celestrak** — https://celestrak.org (TLE data)
- **JPL SPICE** — https://naif.jpl.nasa.gov/naif/

---

## License

This project is provided as-is for educational and research purposes.

---

## Author Notes

This suite demonstrates the **full pipeline** from raw sensor observations to mission-ready trajectory designs. Each phase builds on the previous, showing how satellite operations engineers tackle real problems:

1. **Observe** → Get noisy measurements
2. **Estimate** → Clean up with Kalman filtering
3. **Plan** → Design maneuvers within fuel budgets
4. **Optimize** → Find the best interplanetary route

The code emphasizes **readability over performance**. Production systems would use compiled numerics (C/Fortran), but these implementations prioritize clarity for learning.
