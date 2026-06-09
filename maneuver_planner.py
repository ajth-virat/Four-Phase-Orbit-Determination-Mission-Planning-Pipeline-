"""
PROJECT 3: Station-Keeping & Maneuver Planner
==============================================
INPUT  : ekf_results.csv  (output of Project 2)
OUTPUT : maneuver_plan.csv
         maneuver_plots.png

Libraries: poliastro, spiceypy, astropy, numpy, pandas, matplotlib, scipy

WHAT THIS PROJECT DOES
-----------------------
1. Loads EKF estimated state vector (ECEF)
2. Converts to ECI → Keplerian elements (Vallado Algorithm 9)
3. Propagates orbit with J2 + drag using poliastro
4. Plans Hohmann transfer to correct SMA deviation
5. Plans inclination correction maneuver
6. Outputs full delta-v budget to CSV → input for Project 4

THE MATH
---------
KEPLERIAN ELEMENTS (Vallado Algorithm 9):
  h   = r × v                         angular momentum
  e   = (1/μ)(v×h) - r/|r|            eccentricity vector
  i   = arccos(h_z/|h|)               inclination
  Ω   = arccos(n_x/|n|)               RAAN
  ω   = arccos(n·e/|n||e|)            argument of perigee
  a   = -μ/(2E)  where E = v²/2 - μ/r  semi-major axis

HOHMANN TRANSFER (Vallado Ch. 6):
  a_t  = (a1 + a2) / 2                transfer ellipse SMA
  Δv1  = √(μ/a1) * (√(2a2/(a1+a2)) - 1)   raise orbit
  Δv2  = √(μ/a2) * (1 - √(2a1/(a1+a2)))   circularise

INCLINATION CHANGE:
  Δv = 2 * v_circ * sin(Δi/2)         pure plane change at node
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

# poliastro
try:
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    from poliastro.maneuver import Maneuver
    from poliastro.twobody.propagation import CowellPropagator
    from poliastro.core.perturbations import J2_perturbation, atmospheric_drag_exponential
    POLIASTRO_AVAILABLE = True
    print("[✓] poliastro available")
except ImportError:
    POLIASTRO_AVAILABLE = False
    raise ImportError("poliastro not found: uv pip install poliastro")

# spiceypy
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
    print("[✓] spiceypy available")
except ImportError:
    SPICE_AVAILABLE = False
    print("[!] spiceypy not found")

from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MU      = 3.986004418e14
RE_M    = 6_378_137.0
J2      = 1.08262668e-3
OMEGA_E = 7.2921150e-5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# AJISAI reference orbit
AJISAI_A_REF = RE_M + 1_488_000.0
AJISAI_E_REF = 0.001
AJISAI_I_REF = 50.0 * DEG2RAD

INPUT_CSV  = "ekf_results.csv"
OUTPUT_CSV = "maneuver_plan.csv"
OUTPUT_PNG = "maneuver_plots.png"

# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────

def gmst_from_iso(iso_str):
    t = Time(iso_str[:26], format='isot', scale='utc')
    return t.sidereal_time('mean', 'greenwich').rad


def ecef_to_eci_state(r_ecef, v_ecef, gmst):
    ct, st  = np.cos(gmst), np.sin(gmst)
    R       = np.array([[ ct, -st, 0],
                         [ st,  ct, 0],
                         [  0,   0, 1]])
    r_eci   = R @ r_ecef
    omega   = np.array([0, 0, OMEGA_E])
    v_eci   = R @ v_ecef + np.cross(omega, r_eci)
    return r_eci, v_eci

# ─────────────────────────────────────────────
# KEPLERIAN ELEMENTS (Vallado Algorithm 9)
# ─────────────────────────────────────────────

def state_to_keplerian(r_eci, v_eci):
    r_mag = np.linalg.norm(r_eci)
    v_mag = np.linalg.norm(v_eci)
    h_vec = np.cross(r_eci, v_eci)
    h_mag = np.linalg.norm(h_vec)
    K     = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n_mag = np.linalg.norm(n_vec)
    e_vec = (1/MU) * ((v_mag**2 - MU/r_mag)*r_eci - np.dot(r_eci, v_eci)*v_eci)
    e_mag = np.linalg.norm(e_vec)
    E_mec = v_mag**2/2 - MU/r_mag
    a     = -MU / (2*E_mec)
    i     = np.arccos(np.clip(h_vec[2]/h_mag, -1, 1))
    raan  = np.arccos(np.clip(n_vec[0]/n_mag, -1, 1))
    if n_vec[1] < 0: raan = 2*np.pi - raan
    aop   = np.arccos(np.clip(np.dot(n_vec, e_vec)/(n_mag*e_mag), -1, 1))
    if e_vec[2] < 0: aop = 2*np.pi - aop
    ta    = np.arccos(np.clip(np.dot(e_vec, r_eci)/(e_mag*r_mag), -1, 1))
    if np.dot(r_eci, v_eci) < 0: ta = 2*np.pi - ta
    T     = 2*np.pi * np.sqrt(a**3/MU)
    return dict(a=a, e=e_mag, i_deg=i*RAD2DEG, raan_deg=raan*RAD2DEG,
                aop_deg=aop*RAD2DEG, ta_deg=ta*RAD2DEG, T_s=T,
                alt_km=(a-RE_M)/1000, h_vec=h_vec, e_vec=e_vec, n_vec=n_vec)

# ─────────────────────────────────────────────
# J2 + DRAG PROPAGATOR (numpy fallback)
# ─────────────────────────────────────────────

def j2_drag_propagate(r0, v0, dt_total, dt_step=60.0,
                       Cd=2.2, A=4.0, m=685.0):
    """
    J2 + exponential atmosphere drag propagator.
    Cd: drag coefficient, A: cross-section area (m²), m: mass (kg)
    """
    rho0  = 1.225e-11   # kg/m³ at ~1500km (approximate)
    H     = 200_000.0   # scale height (m)
    B     = Cd * A / m  # ballistic coefficient

    def eom(t, state):
        r_vec   = state[:3]
        v_vec   = state[3:]
        x,y,z   = r_vec
        r       = np.linalg.norm(r_vec)
        v       = np.linalg.norm(v_vec)
        # J2
        a_2b    = -MU/r**3 * r_vec
        fac     = (3/2)*J2*MU*RE_M**2/r**5
        a_j2    = fac*np.array([x*(5*z**2/r**2-1),
                                 y*(5*z**2/r**2-1),
                                 z*(5*z**2/r**2-3)])
        # Drag
        rho     = rho0 * np.exp(-(r - RE_M)/H)
        a_drag  = -0.5 * B * rho * v * v_vec
        return np.concatenate([v_vec, a_2b+a_j2+a_drag])

    sol = solve_ivp(eom, [0, dt_total], np.concatenate([r0, v0]),
                    method='RK45', max_step=dt_step, rtol=1e-9, atol=1e-9)
    return sol.t, sol.y[:3].T, sol.y[3:].T


def poliastro_propagate(r_eci, v_eci, t_iso, dt_total_s):
    """
    High-fidelity propagation using poliastro 0.17 with J2 + drag.
    Uses Cowell propagator with perturbation functions.
    poliastro 0.17 API: propagate() with ad (additional accelerations).
    """
    epoch = Time(t_iso[:26], format='isot', scale='utc')
    r_km  = r_eci / 1000.0
    v_kms = v_eci / 1000.0

    orb = Orbit.from_vectors(
        Earth,
        r_km * u.km,
        v_kms * (u.km / u.s),
        epoch=epoch
    )

    # Perturbation function for Cowell — poliastro 0.17 signature
    R_earth = Earth.R.to(u.km).value

    def perturbations(t0, state, k):
        a_j2 = J2_perturbation(t0, state, k, J2, R_earth)
        a_dr = atmospheric_drag_exponential(
        t0, state, k, R_earth, 2.2, 4e-6/685.0, 200.0, 1.225e-14
    )

    # poliastro 0.17: propagate each timestep individually using Cowell
    tof_vals   = np.linspace(0, dt_total_s, 200)
    positions  = []
    velocities = []

    for tof_s in tof_vals:
        orb_prop = orb.propagate(
            tof_s * u.s,
            method=CowellPropagator(f=perturbations)
        )
        positions.append(orb_prop.r.to(u.m).value)
        velocities.append(orb_prop.v.to(u.m / u.s).value)

    return tof_vals, np.array(positions), np.array(velocities)

# ─────────────────────────────────────────────
# HOHMANN TRANSFER
# ─────────────────────────────────────────────

def hohmann_transfer(a_current, a_target):
    """
    Vallado Ch.6 two-impulse Hohmann transfer.
    a_t  = (a1+a2)/2
    Δv1  = √(μ/a1)*(√(2a2/(a1+a2))-1)
    Δv2  = √(μ/a2)*(1-√(2a1/(a1+a2)))
    """
    a_t      = (a_current + a_target) / 2.0
    v_c1     = np.sqrt(MU / a_current)
    v_c2     = np.sqrt(MU / a_target)
    v_t_peri = np.sqrt(MU * (2/a_current - 1/a_t))
    v_t_apo  = np.sqrt(MU * (2/a_target  - 1/a_t))
    dv1      = v_t_peri - v_c1
    dv2      = v_c2 - v_t_apo
    tof      = np.pi * np.sqrt(a_t**3 / MU)
    return dict(dv1_ms=dv1, dv2_ms=dv2,
                dv_total_ms=abs(dv1)+abs(dv2),
                tof_s=tof, tof_min=tof/60,
                a_transfer_m=a_t)


def inclination_correction(a, i_current_deg, i_target_deg):
    """Δv = 2*v_circ*sin(Δi/2)  — pure plane change at node."""
    delta_i = abs(i_target_deg - i_current_deg) * DEG2RAD
    v_circ  = np.sqrt(MU / a)
    dv      = 2 * v_circ * np.sin(delta_i / 2)
    return dict(delta_i_deg=abs(i_target_deg-i_current_deg),
                dv_ms=dv, v_circ_ms=v_circ)

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def make_plots(times_p, pos_p, elements, hohmann, incl, df):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    lkw = dict(color="#cccccc", fontsize=9)
    tkw = dict(colors="#aaaaaa")

    def ax_(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#0d1117")
        ax.tick_params(**tkw)
        ax.grid(True, color="#222233", lw=0.5)
        return ax

    # 1. Altitude drift
    alts = (np.linalg.norm(pos_p, axis=1) - RE_M)/1000
    ax1  = ax_(gs[0, :2])
    ax1.plot(times_p/60, alts, color="#00aaff", lw=1.5, label="Propagated (J2+drag)")
    ax1.axhline((AJISAI_A_REF-RE_M)/1000, color="#ff0055", lw=1, ls="--",
                label=f"Reference {(AJISAI_A_REF-RE_M)/1000:.0f}km")
    ax1.set_xlabel("Time (min)", **lkw); ax1.set_ylabel("Altitude (km)", **lkw)
    ax1.set_title("Altitude Drift — J2 + Atmospheric Drag", color="#fff", fontsize=10)
    ax1.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc")

    # 2. Element comparison
    ax2  = ax_(gs[0, 2])
    lbls = ["a (km)", "e×10³", "i (°)"]
    cur  = [elements['a']/1000, elements['e']*1000, elements['i_deg']]
    ref  = [AJISAI_A_REF/1000, AJISAI_E_REF*1000, AJISAI_I_REF*RAD2DEG]
    x    = np.arange(3)
    ax2.bar(x-0.2, cur, 0.35, color="#00aaff", label="Estimated", alpha=0.8)
    ax2.bar(x+0.2, ref, 0.35, color="#ff6600", label="Reference", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(lbls, color="#cccccc", fontsize=8)
    ax2.set_title("Orbital Elements vs Reference", color="#fff", fontsize=10)
    ax2.legend(fontsize=7, facecolor="#111122", labelcolor="#cccccc")

    # 3. Hohmann diagram
    ax3  = ax_(gs[1, :2])
    theta = np.linspace(0, 2*np.pi, 300)
    r_c1  = (elements['a'] - RE_M)/1000
    r_c2  = (AJISAI_A_REF - RE_M)/1000
    ax3.plot(r_c1*np.cos(theta), r_c1*np.sin(theta),
             color="#00aaff", lw=1.5, label=f"Current {r_c1:.0f}km")
    ax3.plot(r_c2*np.cos(theta), r_c2*np.sin(theta),
             color="#00ff88", lw=1.5, label=f"Target {r_c2:.0f}km")
    a_tr  = (hohmann['a_transfer_m'] - RE_M)/1000
    theta_h = np.linspace(0, np.pi, 200)
    ax3.plot((a_tr)*np.cos(theta_h) - (a_tr-r_c1),
             (a_tr)*np.sin(theta_h),
             color="#ff6600", lw=1.5, ls="--", label="Transfer arc")
    ax3.scatter([r_c1, -r_c2], [0, 0], s=100, c=["#ff0055","#ffcc00"], zorder=5)
    ax3.annotate(f"Δv₁={hohmann['dv1_ms']:.3f}m/s", (r_c1,0),
                 xytext=(10,10), textcoords="offset points",
                 color="#ff0055", fontsize=8)
    ax3.annotate(f"Δv₂={hohmann['dv2_ms']:.3f}m/s", (-r_c2,0),
                 xytext=(10,-15), textcoords="offset points",
                 color="#ffcc00", fontsize=8)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x (km from centre)", **lkw); ax3.set_ylabel("y (km)", **lkw)
    ax3.set_title("Hohmann Transfer Diagram", color="#fff", fontsize=10)
    ax3.legend(fontsize=8, facecolor="#111122", labelcolor="#cccccc")

    # 4. Delta-v budget
    ax4 = ax_(gs[1, 2])
    mvs  = ["Hohmann Δv₁", "Hohmann Δv₂", "Incl. Corr."]
    dvs  = [abs(hohmann['dv1_ms']), abs(hohmann['dv2_ms']), incl['dv_ms']]
    clrs = ["#00aaff", "#00ff88", "#ff6600"]
    bars = ax4.barh(mvs, dvs, color=clrs, alpha=0.85)
    for bar, val in zip(bars, dvs):
        ax4.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f} m/s", va='center', color="#cccccc", fontsize=8)
    ax4.set_xlabel("Δv (m/s)", **lkw)
    ax4.set_title("Delta-v Budget", color="#fff", fontsize=10)

    # 5. Ground track
    ax5 = ax_(gs[2, :])
    lats = [np.arctan2(p[2], np.sqrt(p[0]**2+p[1]**2))*RAD2DEG for p in pos_p]
    lons = [np.arctan2(p[1], p[0])*RAD2DEG for p in pos_p]
    sc   = ax5.scatter(lons, lats, c=times_p/60, cmap="plasma", s=4)
    ax5.set_xlim(-180,180); ax5.set_ylim(-90,90)
    ax5.set_xlabel("Longitude (°)", **lkw); ax5.set_ylabel("Latitude (°)", **lkw)
    ax5.set_title("Propagated Ground Track (J2+Drag)", color="#fff", fontsize=10)
    plt.colorbar(sc, ax=ax5, label="Time (min)")

    plt.suptitle("Project 3 — Maneuver Planner", color="#ffffff", fontsize=14, y=1.01)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"  Plot saved : {OUTPUT_PNG}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*55)
    print("  MANEUVER PLANNER — Project 3")
    print("="*55)

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[!] '{INPUT_CSV}' not found. Run Project 2 first.")
        return

    print(f"  Loaded {len(df)} state estimates from '{INPUT_CSV}'")

    final   = df.iloc[-1]
    r_ecef  = np.array([final["X_m"], final["Y_m"], final["Z_m"]])
    v_ecef  = np.array([final["Vx_ms"], final["Vy_ms"], final["Vz_ms"]])
    t_iso   = final["timestamp_utc"]

    gmst        = gmst_from_iso(t_iso)
    r_eci, v_eci = ecef_to_eci_state(r_ecef, v_ecef, gmst)

    print(f"\n  |r| = {np.linalg.norm(r_eci)/1000:.2f} km")
    print(f"  |v| = {np.linalg.norm(v_eci):.2f} m/s")

    print(f"\n[1/4] Keplerian elements (Vallado Algorithm 9)...")
    el = state_to_keplerian(r_eci, v_eci)
    print(f"  a   = {el['a']/1000:.3f} km  (ref: {AJISAI_A_REF/1000:.3f} km)")
    print(f"  e   = {el['e']:.6f}")
    print(f"  i   = {el['i_deg']:.4f}°  (ref: {AJISAI_I_REF*RAD2DEG:.4f}°)")
    print(f"  T   = {el['T_s']/60:.2f} min")
    print(f"  Alt = {el['alt_km']:.2f} km")

    print(f"\n[2/4] Propagating 1 orbital period with J2 + drag...")
    times_p, pos_p, vel_p = poliastro_propagate(r_eci, v_eci, t_iso, el['T_s'])
    print(f"  Propagated {len(times_p)} steps over {el['T_s']/60:.1f} min")

    print(f"\n[3/4] Hohmann transfer planning...")
    hohmann = hohmann_transfer(el['a'], AJISAI_A_REF)
    print(f"  Δv₁ = {hohmann['dv1_ms']:.4f} m/s")
    print(f"  Δv₂ = {hohmann['dv2_ms']:.4f} m/s")
    print(f"  TOF = {hohmann['tof_min']:.2f} min")

    print(f"\n[4/4] Inclination correction...")
    incl = inclination_correction(el['a'], el['i_deg'], AJISAI_I_REF*RAD2DEG)
    print(f"  Δi  = {incl['delta_i_deg']:.4f}°")
    print(f"  Δv  = {incl['dv_ms']:.4f} m/s")

    plan = pd.DataFrame([
        dict(maneuver="Hohmann_burn_1", dv_ms=hohmann['dv1_ms'],
             location="periapsis", tof_min=0,
             a_before_m=el['a'], a_after_m=hohmann['a_transfer_m']),
        dict(maneuver="Hohmann_burn_2", dv_ms=hohmann['dv2_ms'],
             location="apoapsis", tof_min=hohmann['tof_min'],
             a_before_m=hohmann['a_transfer_m'], a_after_m=AJISAI_A_REF),
        dict(maneuver="Inclination_correction", dv_ms=incl['dv_ms'],
             location="ascending_node",
             tof_min=hohmann['tof_min']+el['T_s']/60/2,
             a_before_m=AJISAI_A_REF, a_after_m=AJISAI_A_REF),
    ])
    plan.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Plan saved: {OUTPUT_CSV}")
    print(f"  Total Δv  : {hohmann['dv_total_ms']+incl['dv_ms']:.4f} m/s")

    make_plots(times_p, pos_p, el, hohmann, incl, df)

    print("\n"+"="*55)
    print("  ✓ Project 3 complete.")
    print(f"  Feed 'maneuver_plan.csv' into Project 4.")
    print("="*55)

if __name__ == "__main__":
    main()
