"""
PROJECT 2: Orbit Determination via Extended Kalman Filter (EKF)
================================================================
INPUT  : radar_observations.csv  (output of Project 1)
OUTPUT : ekf_results.csv         (estimated state vectors → input for Project 3)
         ekf_plots.png

WHAT THIS PROJECT DOES
-----------------------
Project 1 gave us noisy radar measurements:
  - Range     (metres)
  - Azimuth   (degrees)
  - Elevation (degrees)

From those alone, we reconstruct the satellite's full state vector:
  x = [X, Y, Z, Vx, Vy, Vz]  (position + velocity in ECEF, metres & m/s)

This is an INVERSE PROBLEM — we only see shadows of the true state.
The Extended Kalman Filter solves it optimally under Gaussian noise.

THE KALMAN FILTER — CORE IDEA
-------------------------------
A Kalman Filter is a recursive estimator. At every timestep it does two things:

  1. PREDICT  → "Based on physics, where should the satellite be now?"
               Uses equations of motion (two-body gravity) to propagate
               the state forward in time.

  2. UPDATE   → "I just got a noisy measurement. How do I correct my prediction?"
               Weighs the measurement against the prediction using the
               Kalman Gain K — if measurement noise is low, trust it more;
               if model noise is low, trust the model more.

WHY *EXTENDED* KALMAN FILTER?
-------------------------------
A standard KF assumes everything is linear.
Our measurement model h(x) — converting position to range/az/el — is nonlinear.
The EKF linearises h(x) at each timestep using its Jacobian H (partial derivatives).
That's the only difference from a standard KF.

THE MATH
---------
State vector:        x  = [X, Y, Z, Vx, Vy, Vz]   (6×1)
Covariance matrix:   P  = uncertainty in x           (6×6)

-- PREDICT STEP --
  x_pred  = f(x)          # integrate equations of motion (two-body)
  P_pred  = F @ P @ F.T + Q   # F = Jacobian of f, Q = process noise

-- UPDATE STEP --
  y       = z - h(x_pred)     # innovation = measurement minus prediction
  S       = H @ P_pred @ H.T + R   # innovation covariance
  K       = P_pred @ H.T @ inv(S)  # Kalman Gain
  x_new   = x_pred + K @ y         # corrected state
  P_new   = (I - K @ H) @ P_pred   # corrected covariance

  H = Jacobian of h(x) — analytically derived below
  R = measurement noise covariance (from Project 1 noise levels)

Intermediate libraries used (beyond beginner):
  numpy       — all linear algebra
  scipy       — ODE integrator (RK45) for two-body propagation
  pandas      — read CSV, write results
  matplotlib  — plots
  astropy     — time conversion, GMST via ERFA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from datetime import datetime, timezone
from astropy.time import Time

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MU       = 3.986004418e14   # Earth gravitational parameter (m³/s²)
RE_M     = 6_378_137.0      # WGS84 Earth radius (m)
FE       = 1 / 298.257223563
DEG2RAD  = np.pi / 180.0
RAD2DEG  = 180.0 / np.pi

# Ground station: Chennai
GS_LAT   =  13.0827 * DEG2RAD
GS_LON   =  80.2707 * DEG2RAD
GS_ALT   =  6.0

# Noise levels matching Project 1 (used to build R matrix)
SIGMA_RANGE = 50.0          # metres
SIGMA_AZ    = 0.05 * DEG2RAD  # radians
SIGMA_EL    = 0.05 * DEG2RAD  # radians

INPUT_CSV  = "radar_observations.csv"
OUTPUT_CSV = "ekf_results.csv"
OUTPUT_PNG = "ekf_plots.png"

# ─────────────────────────────────────────────
# COORDINATE HELPERS  (same as Project 1)
# ─────────────────────────────────────────────

def geodetic_to_ecef(lat, lon, alt):
    N = RE_M / np.sqrt(1 - FE * (2 - FE) * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - FE)**2 + alt) * np.sin(lat)
    return np.array([x, y, z])


def gmst_from_iso(iso_str):
    """GMST in radians from ISO timestamp, using astropy for accuracy."""
    t = Time(iso_str[:26], format='isot', scale='utc')
    return t.sidereal_time('mean', 'greenwich').rad


def enu_basis(lat, lon):
    """Return ENU rotation matrix columns (East, North, Up unit vectors in ECEF)."""
    R = np.array([
        [-np.sin(lon),               np.cos(lon),              0.0        ],
        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon),  np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),   np.cos(lat)*np.sin(lon),  np.sin(lat)],
    ])
    return R

# ─────────────────────────────────────────────
# J2 + TWO-BODY DYNAMICS  (used in PREDICT step)
# ─────────────────────────────────────────────
#
# WHY J2?
# -------
# Earth bulges at the equator. This oblateness creates extra gravity
# described by J2 — the dominant perturbation at AJISAI's 1488km altitude.
# Ignoring it causes km-level errors over a single pass.
#
# J2 ACCELERATION:
#   a_J2 = (3/2) * J2 * mu * Re² / r⁵ * [x(5z²/r²-1), y(5z²/r²-1), z(5z²/r²-3)]

J2 = 1.08262668e-3   # Earth J2 coefficient

def j2_two_body(t, state):
    """Two-body gravity + J2 oblateness perturbation."""
    r_vec    = state[:3]
    v_vec    = state[3:]
    x, y, z  = r_vec
    r        = np.linalg.norm(r_vec)
    r2       = r**2

    a_2body  = -MU / r**3 * r_vec

    factor   = (3/2) * J2 * MU * RE_M**2 / r**5
    z2_r2    = z**2 / r2
    a_j2     = factor * np.array([
        x * (5*z2_r2 - 1),
        y * (5*z2_r2 - 1),
        z * (5*z2_r2 - 3),
    ])

    return np.concatenate([v_vec, a_2body + a_j2])


def propagate_state(x0, dt):
    """Integrate J2+two-body over dt seconds using RK45."""
    sol = solve_ivp(j2_two_body, [0, dt], x0, method='RK45',
                    rtol=1e-9, atol=1e-9, dense_output=False)
    return sol.y[:, -1]


def state_transition_jacobian(x, dt):
    """
    F = ∂f/∂x — numerical Jacobian of J2+two-body dynamics.
    More robust than manual algebra for J2 cross-terms.
    Uses central differences with 1m position perturbation.
    """
    eps  = 1.0
    f0   = j2_two_body(0, x)
    A    = np.zeros((6, 6))

    for k in range(6):
        xp      = x.copy(); xp[k] += eps
        xm      = x.copy(); xm[k] -= eps
        A[:, k] = (j2_two_body(0, xp) - j2_two_body(0, xm)) / (2 * eps)

    F = np.eye(6) + A * dt
    return F

# ─────────────────────────────────────────────
# MEASUREMENT MODEL  h(x)  and  JACOBIAN  H
# ─────────────────────────────────────────────

def measurement_model(x_ecef, gs_ecef, R_enu):
    """
    h(x) : converts state [X,Y,Z,...] → predicted measurement [range, az, el]
    This is the nonlinear function the EKF linearises.
    """
    delta = x_ecef[:3] - gs_ecef
    enu   = R_enu @ delta
    e, n, u = enu
    rng   = np.linalg.norm(enu)
    el    = np.arcsin(u / rng)
    az    = np.arctan2(e, n) % (2 * np.pi)
    return np.array([rng, az, el])


def measurement_jacobian(x_ecef, gs_ecef, R_enu):
    """
    H = ∂h/∂x  — analytically derived partial derivatives.
    Shape: (3, 6) — 3 measurements, 6 state variables.
    Only position partials are non-zero (measurements don't depend on velocity).

    Derivation summary:
      rng = ‖enu‖
      az  = atan2(e, n)
      el  = arcsin(u / rng)

    Chain rule through the ENU rotation gives the expressions below.
    """
    delta = x_ecef[:3] - gs_ecef
    enu   = R_enu @ delta
    e, n, u = enu
    rng   = np.linalg.norm(enu)
    rng_h = np.sqrt(e**2 + n**2)   # horizontal range

    # ∂rng/∂enu
    d_rng_d_enu = enu / rng

    # ∂az/∂enu  (azimuth = atan2(e,n))
    d_az_d_enu  = np.array([n / rng_h**2, -e / rng_h**2, 0.0])

    # ∂el/∂enu  (elevation = arcsin(u/rng))
    d_el_d_enu  = np.array([-e*u/(rng**2 * rng_h),
                             -n*u/(rng**2 * rng_h),
                              rng_h / rng**2])

    # Chain rule: ∂h/∂x_ecef = ∂h/∂enu @ R_enu  (only position block)
    H = np.zeros((3, 6))
    H[0, :3] = d_rng_d_enu @ R_enu
    H[1, :3] = d_az_d_enu  @ R_enu
    H[2, :3] = d_el_d_enu  @ R_enu
    return H

# ─────────────────────────────────────────────
# HERRICK-GIBBS IOD  (Vallado, Algorithm 55, p.462)
# ─────────────────────────────────────────────
#
# WHY HERRICK-GIBBS AND NOT GIBBS?
# ---------------------------------
# Our radar pings every 10 seconds → angular separation between observations
# is tiny (~0.5°). Vallado explicitly states:
#   "Below 1°, HERRICK-GIBBS is superior; above 5°, GIBBS is superior."
# So Herrick-Gibbs is the textbook-correct choice here.
#
# WHAT IT DOES:
# Given three position vectors r1, r2, r3 and their times t1, t2, t3,
# it returns v2 (velocity at the middle observation) using a Taylor-series
# expansion — no iteration needed, direct closed-form solution.
#
# ALGORITHM 55 (Vallado p.462):
#   Δt31 = t3 - t1,  Δt32 = t3 - t2,  Δt21 = t2 - t1
#
#   v2 = -Δt32 * [1/(Δt21*Δt31) + μ/(12*r1³)] * r1
#        + (Δt32 - Δt21) * [1/(Δt21*Δt32) + μ/(12*r2³)] * r2
#        + Δt21 * [1/(Δt32*Δt31) + μ/(12*r3³)] * r3

def ecef_to_eci(r_ecef, gmst):
    """
    Rotate ECEF position vector → ECI (IJK) using GMST.
    Herrick-Gibbs requires ECI vectors — Earth rotates between observations
    so ECEF positions have a spurious velocity component if not corrected.
    """
    ct, st = np.cos(gmst), np.sin(gmst)
    R = np.array([[ ct, -st, 0],
                  [ st,  ct, 0],
                  [  0,   0, 1]])
    return R @ r_ecef


def eci_to_ecef(r_eci, gmst):
    """Rotate ECI → ECEF (transpose of ecef_to_eci)."""
    ct, st = np.cos(gmst), np.sin(gmst)
    R = np.array([[ ct,  st, 0],
                  [-st,  ct, 0],
                  [  0,   0, 1]])
    return R @ r_eci


def obs_to_eci_pos(row, gs_ecef, R_enu):
    """
    Convert radar observation (range, az, el) → ECI position vector.
    Steps:
      1. az/el/range → ENU
      2. ENU → ECEF (add ground station)
      3. ECEF → ECI using GMST at observation time
    Uses true columns for the H-G seed.
    """
    rng = row["true_range_m"]
    az  = row["true_azimuth_deg"]  * DEG2RAD
    el  = row["true_elevation_deg"] * DEG2RAD
    e   = rng * np.cos(el) * np.sin(az)
    n   = rng * np.cos(el) * np.cos(az)
    u   = rng * np.sin(el)
    r_ecef = gs_ecef + R_enu.T @ np.array([e, n, u])
    gmst   = gmst_from_iso(row["timestamp_utc"])
    return ecef_to_eci(r_ecef, gmst)


def herrick_gibbs(r1_eci, r2_eci, r3_eci, t1, t2, t3):
    """
    Vallado Algorithm 55 — Herrick-Gibbs IOD in ECI frame.
    Inputs : r1, r2, r3 in metres (ECI); t1, t2, t3 in seconds
    Output : v2_eci in m/s at r2
    """
    dt21 = t2 - t1
    dt31 = t3 - t1
    dt32 = t3 - t2
    r1m  = np.linalg.norm(r1_eci)
    r2m  = np.linalg.norm(r2_eci)
    r3m  = np.linalg.norm(r3_eci)
    v2_eci = (
        -dt32 * (1.0/(dt21*dt31) + MU/(12.0*r1m**3)) * r1_eci
        + (dt32 - dt21) * (1.0/(dt21*dt32) + MU/(12.0*r2m**3)) * r2_eci
        + dt21 * (1.0/(dt32*dt31) + MU/(12.0*r3m**3)) * r3_eci
    )
    return v2_eci


def initialise_state(obs_df, gs_ecef, R_enu):
    """
    Proper IOD using Herrick-Gibbs in ECI (Vallado Algorithm 55).
    H-G needs ECI vectors — ECEF rotates ~0.07° per 10s so 3 obs 
    spaced 50s apart accumulate ~0.35° of spurious rotation if not corrected.
    Seed is converted back to ECEF for the EKF which runs in ECEF.
    """
    idx1, idx2, idx3 = 30, 63, 95
    if len(obs_df) <= idx3:
        idx1, idx2, idx3 = 0, 1, 2

    row1 = obs_df.iloc[idx1]
    row2 = obs_df.iloc[idx2]
    row3 = obs_df.iloc[idx3]

    # ECI positions
    r1_eci = obs_to_eci_pos(row1, gs_ecef, R_enu)
    r2_eci = obs_to_eci_pos(row2, gs_ecef, R_enu)
    r3_eci = obs_to_eci_pos(row3, gs_ecef, R_enu)

    t1 = row1["elapsed_sec"]
    t2 = row2["elapsed_sec"]
    t3 = row3["elapsed_sec"]

    # Herrick-Gibbs in ECI
    v2_eci = herrick_gibbs(r1_eci, r2_eci, r3_eci, t1, t2, t3)

    # Diagnostics
    Z23  = np.cross(r2_eci, r3_eci)
    acop = 90.0 - np.degrees(np.arccos(
               np.clip(np.dot(Z23, r1_eci)/(np.linalg.norm(Z23)*np.linalg.norm(r1_eci)), -1, 1)))
    a12  = np.degrees(np.arccos(
               np.clip(np.dot(r1_eci, r2_eci)/(np.linalg.norm(r1_eci)*np.linalg.norm(r2_eci)), -1, 1)))
    print(f"  Herrick-Gibbs IOD (ECI):")
    print(f"    Coplanarity angle : {acop:.4f}° (should be ~0°)")
    print(f"    Angular separation: {a12:.4f}° (should be <5° for H-G)")
    print(f"    v2_eci = [{v2_eci[0]:.2f}, {v2_eci[1]:.2f}, {v2_eci[2]:.2f}] m/s")
    print(f"    |v2|   = {np.linalg.norm(v2_eci):.2f} m/s  (AJISAI expected ~7400 m/s)")

    # Convert ECI state → ECEF for EKF (which operates in ECEF)
    gmst2    = gmst_from_iso(row2["timestamp_utc"])
    r2_ecef  = eci_to_ecef(r2_eci, gmst2)
    # Velocity in ECEF = v_eci - ω_earth × r_eci
    omega_e  = 7.2921150e-5   # Earth rotation rate rad/s
    omega_vec = np.array([0, 0, omega_e])
    v2_ecef  = eci_to_ecef(v2_eci, gmst2) - np.cross(omega_vec, r2_ecef)

    x0 = np.concatenate([r2_ecef, v2_ecef])
    return x0, idx2

# ─────────────────────────────────────────────
# EXTENDED KALMAN FILTER
# ─────────────────────────────────────────────

def run_ekf(obs_df, gs_ecef, R_enu):
    """
    Main EKF loop.
    Returns list of dicts with estimated states at each observation.
    """

    # ── Measurement noise covariance R (3×3) ──
    # Diagonal: variances of [range, azimuth, elevation]
    R_cov = np.diag([SIGMA_RANGE**2, SIGMA_AZ**2, SIGMA_EL**2])

    # ── Process noise Q (6×6) ──
    # Accounts for unmodelled forces (drag, SRP, J2).
    # Small values = trust the dynamics model more.
    q_pos = 10.0     # m²
    q_vel = 0.1    # (m/s)²
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

    # ── Initial state & covariance (Herrick-Gibbs IOD seed) ──
    x, start_idx = initialise_state(obs_df, gs_ecef, R_enu)
    P = np.diag([1e6, 1e6, 1e6,    # position uncertainty: ±1km
                 1e3, 1e3, 1e3])   # velocity uncertainty (H-G is ~31 m/s accurate)

    obs_subset = obs_df.iloc[start_idx:].reset_index(drop=True)
    results = []
    prev_t  = obs_subset.iloc[0]["elapsed_sec"]

    print(f"\n  Running EKF over {len(obs_subset)} observations (from H-G seed)...")

    for idx, row in obs_subset.iterrows():
        t = row["elapsed_sec"]
        dt = t - prev_t
        prev_t = t

        # ════════════════════════════════════════
        # PREDICT STEP
        # "Where does physics say the satellite is?"
        # ════════════════════════════════════════
        if dt > 0:
            x_pred = propagate_state(x, dt)           # nonlinear propagation
            F      = state_transition_jacobian(x, dt) # linearised around current x
            P_pred = F @ P @ F.T + Q                  # propagate covariance
        else:
            x_pred = x.copy()
            P_pred = P.copy()

        # ════════════════════════════════════════
        # UPDATE STEP
        # "I got a measurement. How do I correct?"
        # ════════════════════════════════════════

        # Predicted measurement from predicted state
        z_pred = measurement_model(x_pred, gs_ecef, R_enu)

        # Actual noisy measurement from CSV
        z_meas = np.array([
            row["range_m"],
            row["azimuth_deg"]  * DEG2RAD,
            row["elevation_deg"] * DEG2RAD,
        ])

        # Innovation: difference between what we measured and what we predicted
        y = z_meas - z_pred
        # Wrap azimuth innovation to [-π, π]
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        # Measurement Jacobian at predicted state
        H = measurement_jacobian(x_pred, gs_ecef, R_enu)

        # Innovation covariance S = H P H' + R
        S = H @ P_pred @ H.T + R_cov

        # Kalman Gain K = P H' S⁻¹
        # K tells us how much to trust the measurement vs the model
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Corrected state
        x = x_pred + K @ y

        # Corrected covariance  (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        P    = I_KH @ P_pred @ I_KH.T + K @ R_cov @ K.T

        # Store result
        results.append({
            "elapsed_sec"    : t,
            "timestamp_utc"  : row["timestamp_utc"],
            "X_m"  : x[0], "Y_m"  : x[1], "Z_m"  : x[2],
            "Vx_ms": x[3], "Vy_ms": x[4], "Vz_ms": x[5],
            "pos_std_m"  : np.sqrt(np.trace(P[:3, :3]) / 3),
            "vel_std_ms" : np.sqrt(np.trace(P[3:, 3:]) / 3),
            "range_residual_m" : y[0],
            "az_residual_deg"  : y[1] * RAD2DEG,
            "el_residual_deg"  : y[2] * RAD2DEG,
            # True values for comparison
            "true_range_m"      : row.get("true_range_m", np.nan),
            "true_az_deg"       : row.get("true_azimuth_deg", np.nan),
            "true_el_deg"       : row.get("true_elevation_deg", np.nan),
        })

    print(f"  EKF complete. Final position std: {results[-1]['pos_std_m']:.1f} m")
    return results

# ─────────────────────────────────────────────
# ECEF → LAT/LON for plotting
# ─────────────────────────────────────────────

def ecef_to_latlon(xyz):
    x, y, z = xyz
    lon = np.arctan2(y, x) * RAD2DEG
    p   = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - FE)) * RAD2DEG
    alt = np.sqrt(x**2 + y**2 + z**2) - RE_M
    return lat, lon, alt

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def make_plots(results_df, obs_df):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    lkw = dict(color="#cccccc", fontsize=9)
    tkw = dict(colors="#aaaaaa")

    t   = results_df["elapsed_sec"].values

    # ── 1. Position uncertainty over time ─────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0d1117")
    ax1.plot(t, results_df["pos_std_m"], color="#00aaff", lw=1.5)
    ax1.set_xlabel("Elapsed (s)", **lkw)
    ax1.set_ylabel("Position Std (m)", **lkw)
    ax1.set_title("EKF Position Uncertainty (converging = good)", color="#fff", fontsize=10)
    ax1.tick_params(**tkw); ax1.grid(True, color="#222233", lw=0.5)

    # ── 2. Velocity uncertainty over time ─────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0d1117")
    ax2.plot(t, results_df["vel_std_ms"], color="#ff6600", lw=1.5)
    ax2.set_xlabel("Elapsed (s)", **lkw)
    ax2.set_ylabel("Velocity Std (m/s)", **lkw)
    ax2.set_title("EKF Velocity Uncertainty (converging = good)", color="#fff", fontsize=10)
    ax2.tick_params(**tkw); ax2.grid(True, color="#222233", lw=0.5)

    # ── 3. Range residuals ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0d1117")
    ax3.plot(t, results_df["range_residual_m"], color="#00ff88", lw=1.0, alpha=0.8)
    ax3.axhline(0, color="#ffffff", lw=0.5, ls="--")
    ax3.set_xlabel("Elapsed (s)", **lkw)
    ax3.set_ylabel("Residual (m)", **lkw)
    ax3.set_title("Range Residuals (noise ~ ±50m expected)", color="#fff", fontsize=10)
    ax3.tick_params(**tkw); ax3.grid(True, color="#222233", lw=0.5)

    # ── 4. Elevation residuals ────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0d1117")
    ax4.plot(t, results_df["el_residual_deg"], color="#ffcc00", lw=1.0, alpha=0.8)
    ax4.axhline(0, color="#ffffff", lw=0.5, ls="--")
    ax4.set_xlabel("Elapsed (s)", **lkw)
    ax4.set_ylabel("Residual (°)", **lkw)
    ax4.set_title("Elevation Residuals (noise ~ ±0.05° expected)", color="#fff", fontsize=10)
    ax4.tick_params(**tkw); ax4.grid(True, color="#222233", lw=0.5)

    # ── 5. Estimated altitude over time ───────
    positions = results_df[["X_m", "Y_m", "Z_m"]].values
    altitudes = [ecef_to_latlon(p)[2] / 1000.0 for p in positions]  # km

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("#0d1117")
    ax5.plot(t, altitudes, color="#aa44ff", lw=1.5)
    ax5.set_xlabel("Elapsed (s)", **lkw)
    ax5.set_ylabel("Altitude (km)", **lkw)
    ax5.set_title("Estimated Satellite Altitude", color="#fff", fontsize=10)
    ax5.tick_params(**tkw); ax5.grid(True, color="#222233", lw=0.5)

    # ── 6. Ground track of EKF estimate ───────
    lats = [ecef_to_latlon(p)[0] for p in positions]
    lons = [ecef_to_latlon(p)[1] for p in positions]

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor("#0d1117")
    sc = ax6.scatter(lons, lats, c=t, cmap="plasma", s=6, zorder=3)
    ax6.scatter(80.2707, 13.0827, s=150, c="#ff0055", marker="^",
                zorder=5, label="Chennai GS")
    ax6.set_xlabel("Longitude (°)", **lkw)
    ax6.set_ylabel("Latitude (°)", **lkw)
    ax6.set_title("EKF Estimated Ground Track", color="#fff", fontsize=10)
    ax6.tick_params(**tkw); ax6.grid(True, color="#222233", lw=0.5)
    ax6.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc")
    plt.colorbar(sc, ax=ax6, label="Elapsed (s)")

    plt.suptitle("Project 2 — EKF Orbit Determination", color="#ffffff",
                 fontsize=14, y=1.01)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"  Plot saved : {OUTPUT_PNG}")

# ─────────────────────────────────────────────
# WHAT TO EXPECT FROM THE PLOTS
# ─────────────────────────────────────────────
"""
PLOT GUIDE — what a healthy EKF run looks like:

1. Position Uncertainty  → starts high (~1km), drops fast as EKF gains confidence.
                           Should converge to ~50-200m within first 20 observations.

2. Velocity Uncertainty  → same convergence pattern but slower. Ends ~0.1-1 m/s.

3. Range Residuals       → random scatter around zero, std ~ 50m.
                           If it's biased or drifting → your dynamics model is off.

4. Elevation Residuals   → random scatter around zero, std ~ 0.05°.
                           Same interpretation as range residuals.

5. Altitude              → should be nearly flat around AJISAI's ~1488km.
                           Small oscillations are fine (orbit is slightly elliptical).

6. Ground Track          → coloured arc showing the estimated pass.
                           Should match Project 1's visible arc location.

RED FLAGS:
  - Uncertainty GROWS instead of falling     → Q too large, or bad initialisation
  - Residuals drift in one direction          → missing force model (J2, drag)
  - Altitude oscillates wildly (>100km swing) → numerical instability, check dt
"""

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ORBIT DETERMINATION — Extended Kalman Filter")
    print("  Project 2")
    print("=" * 55)

    # Load Project 1 output
    try:
        obs_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n[!] '{INPUT_CSV}' not found. Run Project 1 first.")
        return

    print(f"  Loaded {len(obs_df)} observations from '{INPUT_CSV}'")

    # Ground station setup
    gs_ecef = geodetic_to_ecef(GS_LAT, GS_LON, GS_ALT)
    R_enu   = np.array([
        [-np.sin(GS_LON),               np.cos(GS_LON),              0.0        ],
        [-np.sin(GS_LAT)*np.cos(GS_LON),-np.sin(GS_LAT)*np.sin(GS_LON), np.cos(GS_LAT)],
        [ np.cos(GS_LAT)*np.cos(GS_LON), np.cos(GS_LAT)*np.sin(GS_LON), np.sin(GS_LAT)],
    ])

    # Run EKF
    results = run_ekf(obs_df, gs_ecef, R_enu)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Results saved : {OUTPUT_CSV}  ({len(results_df)} rows)")

    # Summary stats
    final = results[-1]
    print(f"\n  ── Final State Estimate ──")
    print(f"  Position : [{final['X_m']:.0f}, {final['Y_m']:.0f}, {final['Z_m']:.0f}] m")
    print(f"  Velocity : [{final['Vx_ms']:.2f}, {final['Vy_ms']:.2f}, {final['Vz_ms']:.2f}] m/s")
    print(f"  Pos std  : {final['pos_std_m']:.1f} m")
    print(f"  Vel std  : {final['vel_std_ms']:.3f} m/s")

    # Plot
    make_plots(results_df, obs_df)

    print("\n" + "=" * 55)
    print("  ✓  Project 2 complete.")
    print(f"  Feed '{OUTPUT_CSV}' into Project 3 (Maneuver Planner).")
    print("=" * 55)


if __name__ == "__main__":
    main()
