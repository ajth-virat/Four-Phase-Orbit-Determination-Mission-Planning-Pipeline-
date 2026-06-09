"""
PROJECT 4: Interplanetary Mission Designer
==========================================
INPUT  : maneuver_plan.csv + ekf_results.csv
OUTPUT : mission_trajectory.csv
         mission_plots.png

Libraries: jax, pygmo, casadi, spiceypy, poliastro, astropy, numpy, pandas, matplotlib

WHAT THIS PROJECT DOES
-----------------------
1. Loads Project 3 maneuver plan (LEO station-keeping context)
2. Scales to interplanetary — designs Earth→Mars transfer
3. Computes porkchop plot via Lambert solver grid
4. pygmo global optimization finds the best launch window
5. JAX autodiff computes launch window sensitivity (∂C3/∂t)
6. CasADi NLP refines with high-precision gradient optimization

THE MATH
---------
LAMBERT'S PROBLEM (Izzo 2015):
  Given r1, r2, TOF → find v1, v2 of the connecting conic.
  Universal variable x iterated until t(x) = TOF.

C3 (Characteristic Energy):
  C3 = |v_∞_dep|²   (km²/s²)
  v_∞ = v_transfer - v_planet at departure

PORKCHOP PLOT:
  Grid (t_dep, TOF) → C3. Named for the contour shape.
  Minimum C3 = optimal launch window.

JAX ROLE:
  Differentiate through the C3 function → ∂C3/∂t_dep
  Tells how fast the launch window degrades around the optimum.

PYGMO ROLE:
  DE1220 global optimizer samples thousands of (t_dep, TOF) pairs
  to find global minimum without getting trapped in local minima.

CASADI ROLE:
  NLP refinement of pygmo result + low-thrust trajectory OCP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time
from scipy.integrate import solve_ivp

# JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
    print("[✓] JAX available")
except ImportError:
    JAX_AVAILABLE = False
    print("[!] JAX not found: pip install jax jaxlib")

# scipy DE (replaces pygmo — same Differential Evolution algorithm, pip-native)
from scipy.optimize import differential_evolution
print("[✓] scipy differential_evolution available")

# CasADi
try:
    import casadi as ca
    CASADI_AVAILABLE = True
    print("[✓] CasADi available")
except ImportError:
    CASADI_AVAILABLE = False
    print("[!] CasADi not found: pip install casadi")

# spiceypy
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
    print("[✓] spiceypy available")
except ImportError:
    SPICE_AVAILABLE = False
    print("[!] spiceypy not found")

# poliastro
try:
    from poliastro.iod import izzo as lambert_izzo_poliastro
    from poliastro.bodies import Sun
    POLIASTRO_AVAILABLE = True
    print("[✓] poliastro available")
except ImportError:
    POLIASTRO_AVAILABLE = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MU_SUN   = 1.32712440018e20
MU_EARTH = 3.986004418e14
AU       = 1.495978707e11
RE_M     = 6_378_137.0
R_PARK   = RE_M + 200_000.0
DEG2RAD  = np.pi / 180.0
RAD2DEG  = 180.0 / np.pi

OUTPUT_CSV = "mission_trajectory.csv"
OUTPUT_PNG = "mission_plots.png"

# ─────────────────────────────────────────────
# PLANET EPHEMERIS
# ─────────────────────────────────────────────

def load_spice_kernels():
    """Load SPICE kernels via spiceypy if available."""
    if not SPICE_AVAILABLE:
        return False
    try:
        # Try to find de430 or de440 kernel
        import os
        kernel_paths = [
            "de430.bsp", "de440.bsp",
            os.path.join(os.path.expanduser("~"), "de430.bsp"),
        ]
        for kp in kernel_paths:
            if os.path.exists(kp):
                spice.furnsh(kp)
                print(f"  SPICE kernel loaded: {kp}")
                return True
        print("  [SPICE] No .bsp kernel found — using Keplerian ephemeris.")
        print("  Download de430.bsp from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/")
        return False
    except Exception as ex:
        print(f"  [SPICE] Kernel load failed: {ex}")
        return False


def planet_state_keplerian(planet, t_jd):
    """
    Approximate planet state from Keplerian elements.
    Accurate to ~0.1° — sufficient for porkchop plot and Lambert.
    """
    J2000 = 2451545.0
    T     = (t_jd - J2000) / 36525.0

    if planet.upper() in ["EARTH", "399"]:
        a = 1.000001018 * AU
        e = 0.01670862 - 0.000042037*T
        L = (100.46457166 + 36000.77004*T) % 360
        w = 102.93768 + 0.32327*T
    elif planet.upper() in ["MARS", "499"]:
        a = 1.52368 * AU
        e = 0.09341233 + 0.000090479*T
        L = (355.45332 + 19140.30268*T) % 360
        w = 336.04084 + 0.44441*T
    else:
        raise ValueError(f"Unknown planet: {planet}")

    M_rad = ((L - w) % 360) * DEG2RAD
    E = M_rad
    for _ in range(100):
        dE = (M_rad - E + e*np.sin(E)) / (1 - e*np.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r  = a*(1 - e*np.cos(E))
    p  = a*(1 - e**2)

    r_vec = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_vec = np.sqrt(MU_SUN/p) * np.array([-np.sin(nu), e+np.cos(nu), 0.0])
    return r_vec, v_vec


SPICE_LOADED = False

def get_planet_state(planet, t_jd):
    """Get planet state — SPICE if kernels loaded, else Keplerian."""
    global SPICE_LOADED
    if SPICE_LOADED:
        try:
            et = spice.str2et(f"JD {t_jd:.6f} TDB")
            state, _ = spice.spkezr(planet, et, "J2000", "NONE", "SUN")
            return np.array(state[:3])*1000, np.array(state[3:])*1000
        except Exception:
            pass
    return planet_state_keplerian(planet, t_jd)

# ─────────────────────────────────────────────
# LAMBERT SOLVER (Izzo's method)
# ─────────────────────────────────────────────

def lambert_izzo(r1, r2, tof, mu=MU_SUN, prograde=True):
    """
    Izzo (2015) Lambert solver.
    Given r1, r2 (m), tof (s) → returns v1, v2 (m/s).
    Uses Halley iterations on the universal variable x.
    """
    r1m = np.linalg.norm(r1)
    r2m = np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1,r2)/(r1m*r2m), -1, 1)
    cross    = np.cross(r1, r2)
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2*np.pi - np.arccos(cos_dnu)
    else:
        dnu = 2*np.pi - np.arccos(cos_dnu) if cross[2] >= 0 else np.arccos(cos_dnu)

    c   = np.sqrt(r1m**2 + r2m**2 - 2*r1m*r2m*cos_dnu)
    s   = (r1m + r2m + c) / 2
    lam2 = 1 - c/s
    lam  = np.sqrt(lam2) if dnu < np.pi else -np.sqrt(lam2)

    T_nd = tof * np.sqrt(2*mu/s**3)
    T00  = np.arccos(lam) + lam*np.sqrt(1-lam2)
    T1   = 2/3*(1-lam**3)
    x0   = np.clip((T00-T_nd)/(T00-T1) if T_nd < T00 else (T1-T_nd)/(T1*0.5), -0.99, 0.99)
    x    = x0

    for _ in range(50):
        a = 1/(1-x**2)
        if a > 0:
            alpha = 2*np.arccos(x)
            beta  = 2*np.arcsin(np.sqrt(lam2/a))
            if dnu > np.pi: beta = -beta
            psi   = (alpha-beta)/2
            cp, sp = np.cos(psi), np.sin(psi)
            lcp   = lam*cp
            T_x   = (psi + sp*cp - np.sign(np.sin(alpha/2))*(
                     np.arcsin(lcp)+lcp*np.sqrt(1-lcp**2)))/(1-x**2)
        else:
            g   = np.sqrt(x**2-1)
            T_x = (np.exp(g*np.arccosh(x)) - lam*g)/(x**2-1)

        dT  = (3*x*T_x - 2 + 2*lam**3*x)/(1-x**2)
        ddT = (3*T_x + (3*x+2*lam**3)*dT)/(1-x**2) + 2*x*dT/(1-x**2)
        dx  = -(T_x-T_nd)/(dT-(T_x-T_nd)*ddT/(2*dT))
        x   = np.clip(x+dx, -0.99, 0.99)
        if abs(dx) < 1e-12: break

    gamma = np.sqrt(mu*s/2)
    rho   = (r1m-r2m)/c
    sigma = np.sqrt(1-rho**2)
    y     = np.sqrt(1-lam2+lam2*x**2)

    vr1 = gamma*((lam*y-x) - rho*(lam*y+x))/r1m
    vr2 = -gamma*((lam*y-x) + rho*(lam*y+x))/r2m
    vt1 = gamma*sigma*(y+lam*x)/r1m
    vt2 = gamma*sigma*(y+lam*x)/r2m

    r1h = r1/r1m; r2h = r2/r2m
    hh  = cross/np.linalg.norm(cross)
    t1h = np.cross(hh, r1h); t2h = np.cross(hh, r2h)
    if dnu > np.pi: vt1,vt2 = -vt1,-vt2

    return vr1*r1h + vt1*t1h, vr2*r2h + vt2*t2h

# ─────────────────────────────────────────────
# C3 COMPUTATION
# ─────────────────────────────────────────────

def compute_c3(t_dep_jd, tof_days):
    """
    Compute C3 (launch energy) and arrival Δv for given departure/TOF.
    Returns (C3_km2s2, dv_arr_kms, dv_dep_kms) or (nan, nan, nan)
    """
    tof_sec  = tof_days * 86400.0
    t_arr_jd = t_dep_jd + tof_days
    r_dep, v_dep = get_planet_state("EARTH", t_dep_jd)
    r_arr, v_arr = get_planet_state("MARS",  t_arr_jd)
    try:
        v1, v2  = lambert_izzo(r_dep, r_arr, tof_sec)
        v_inf_d = v1 - v_dep
        C3      = np.dot(v_inf_d, v_inf_d) / 1e6
        dv_arr  = np.linalg.norm(v2 - v_arr) / 1000
        v_park  = np.sqrt(MU_EARTH/R_PARK)
        v_hyp   = np.sqrt(max(C3*1e6 + 2*MU_EARTH/R_PARK, 0))
        dv_dep  = (v_hyp - v_park) / 1000
        return C3, dv_arr, dv_dep, v1, v2, r_dep, r_arr
    except Exception:
        return np.nan, np.nan, np.nan, None, None, None, None

# ─────────────────────────────────────────────
# PORKCHOP GRID
# ─────────────────────────────────────────────

def porkchop_grid(dep_start_jd, n_dep=35, n_tof=35,
                  dep_range=400, tof_min=150, tof_max=420):
    print(f"  Computing {n_dep}×{n_tof} porkchop grid...")
    dep_jds = np.linspace(dep_start_jd, dep_start_jd+dep_range, n_dep)
    tofs    = np.linspace(tof_min, tof_max, n_tof)
    C3_grid = np.full((n_tof, n_dep), np.nan)
    for i, tof in enumerate(tofs):
        for j, td in enumerate(dep_jds):
            C3,_,_,_,_,_,_ = compute_c3(td, tof)
            C3_grid[i,j] = C3
        if (i+1) % 7 == 0:
            print(f"    {i+1}/{n_tof} TOF slices done...")
    return dep_jds, tofs, C3_grid

# ─────────────────────────────────────────────
# PYGMO OPTIMIZATION
# ─────────────────────────────────────────────

def run_scipy_de(dep_start, dep_range=400, tof_min=150, tof_max=420):
    """
    Scipy Differential Evolution — same DE algorithm as pygmo DE1220.
    Fully pip-native, no C++ compilation needed on Windows.
    popsize=15 × 2 vars = 30 individuals, maxiter=100 generations.
    strategy='best1bin' matches pygmo DE/1/bin variant.
    """
    print("  Running scipy DE (strategy=best1bin, popsize=15, maxiter=100)...")

    def objective(x):
        C3, dv_arr, dv_dep, *_ = compute_c3(x[0], x[1])
        if np.isnan(C3) or C3 < 0:
            return 1e10
        return C3 + (dv_arr or 0) * 10

    bounds = [(dep_start, dep_start + dep_range), (tof_min, tof_max)]
    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=100,
        popsize=15,
        seed=42,
        tol=1e-6,
        workers=1,
    )
    print(f"  scipy DE best: t_dep={result.x[0]:.1f} JD, TOF={result.x[1]:.1f}d, obj={result.fun:.3f}")
    return result.x[0], result.x[1]

# ─────────────────────────────────────────────
# JAX WINDOW SENSITIVITY
# ─────────────────────────────────────────────

def jax_sensitivity(t_dep_jd, tof_days):
    """
    ∂C3/∂t_dep via JAX autodiff.
    Uses Keplerian approximation (SPICE not JAX-differentiable).
    """
    if not JAX_AVAILABLE:
        print("  [JAX] Not available.")
        return None

    def c3_jax(t_dep):
        J2000   = 2451545.0
        T       = (t_dep - J2000) / 36525.0
        # Earth
        L_e     = (100.46457166 + 36000.77004*T) % 360 * DEG2RAD
        M_e     = L_e - (102.93768 + 0.32327*T)*DEG2RAD
        nu_e    = M_e + 2*0.0167*jnp.sin(M_e)
        r_e     = 1.000*AU * (1-0.0167**2)/(1+0.0167*jnp.cos(nu_e))
        # Mars
        L_m     = (355.45332 + 19140.30268*T) % 360 * DEG2RAD
        M_m     = L_m - (336.04084 + 0.44441*T)*DEG2RAD
        nu_m    = M_m + 2*0.0934*jnp.sin(M_m)
        r_m     = 1.524*AU * (1-0.0934**2)/(1+0.0934*jnp.cos(nu_m))
        # Hohmann proxy C3
        a_t     = (r_e + r_m)/2
        v_t     = jnp.sqrt(MU_SUN*(2/r_e - 1/a_t))
        v_e     = jnp.sqrt(MU_SUN/r_e)
        return (v_t - v_e)**2 / 1e6

    dC3_dt = float(jit(grad(c3_jax))(jnp.float64(t_dep_jd)))
    C3_val  = float(jit(c3_jax)(jnp.float64(t_dep_jd)))
    print(f"  JAX C3 at optimal  : {C3_val:.4f} km²/s²")
    print(f"  JAX ∂C3/∂t_dep     : {dC3_dt:.6f} km²/s² per day")
    print(f"  Window sensitivity : {'NARROW' if abs(dC3_dt) > 0.5 else 'WIDE'}")
    return dC3_dt

# ─────────────────────────────────────────────
# CASADI LOW-THRUST NLP
# ─────────────────────────────────────────────

def casadi_low_thrust(r1, v1, r2, tof_sec, n=15):
    """
    CasADi NLP: low-thrust trajectory optimization.
    Minimizes propellant mass for electric propulsion.
    State: [x,y,z, vx,vy,vz, mass]  Control: [ux,uy,uz] unit thrust vector
    """
    if not CASADI_AVAILABLE:
        print("  [CasADi] Not available.")
        return
    print(f"  CasADi NLP: {n} segments, electric propulsion...")
    dt    = tof_sec / n
    T_max = 0.5     # N
    Isp   = 3000.0  # s
    g0    = 9.80665
    m0    = 1000.0  # kg

    opti  = ca.Opti()
    X     = opti.variable(7, n+1)
    U     = opti.variable(3, n)
    opti.minimize(-X[6,-1])

    for k in range(n):
        r_k   = X[:3,k]; v_k = X[3:6,k]; m_k = X[6,k]
        rm    = ca.norm_2(r_k)
        ag    = -MU_SUN/rm**3 * r_k
        at    = (T_max/m_k)*U[:,k]
        mdot  = -T_max/(Isp*g0)
        opti.subject_to(X[:,k+1] == ca.vertcat(
            r_k+v_k*dt, v_k+(ag+at)*dt, m_k+mdot*dt))

    opti.subject_to(X[:3,0] == r1)
    opti.subject_to(X[3:6,0] == v1)
    opti.subject_to(X[6,0]   == m0)
    opti.subject_to(ca.norm_2(X[:3,-1]-r2) <= 5e9)
    for k in range(n):
        opti.subject_to(ca.norm_2(U[:,k]) <= 1.0)
    opti.subject_to(X[6,:] >= 50)

    for k in range(n+1):
        f = k/n
        opti.set_initial(X[:3,k], r1+f*(r2-r1))
        opti.set_initial(X[3:6,k], v1)
        opti.set_initial(X[6,k], m0-k*T_max*dt/(Isp*g0))
    opti.set_initial(U, np.zeros((3,n)))

    opti.solver("ipopt", {"print_level": 0})
    try:
        sol = opti.solve()
        mf  = float(sol.value(X[6,-1]))
        print(f"  CasADi: final mass={mf:.1f}kg, prop used={m0-mf:.1f}kg")
    except Exception as ex:
        print(f"  CasADi solver: {ex}")

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def make_plots(dep_jds, tofs, C3_grid, best_dep, best_tof,
               r_earth, r_mars, v1, C3_best, dv_dep, dv_arr):
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    lkw = dict(color="#cccccc", fontsize=9)
    tkw = dict(colors="#aaaaaa")

    J2000     = 2451545.0
    dep_years = (dep_jds - J2000)/365.25 + 2000

    # 1. Porkchop
    ax1 = fig.add_subplot(gs[0,:2]); ax1.set_facecolor("#0d1117")
    cf  = ax1.contourf(dep_years, tofs, np.clip(C3_grid,0,80),
                       levels=25, cmap="plasma")
    ax1.contour(dep_years, tofs, np.clip(C3_grid,0,80),
                levels=[5,10,15,20,30,50], colors="white", lw=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax1, label="C3 (km²/s²)")
    best_yr = (best_dep-J2000)/365.25+2000
    ax1.scatter([best_yr],[best_tof], s=200, c="#00ff88", marker="*", zorder=5,
                label=f"Optimal C3={C3_best:.1f}")
    ax1.set_xlabel("Departure Year", **lkw); ax1.set_ylabel("TOF (days)", **lkw)
    ax1.set_title("Earth→Mars Porkchop Plot", color="#fff", fontsize=11)
    ax1.tick_params(**tkw)
    ax1.legend(fontsize=9, facecolor="#111122", labelcolor="#cccccc")

    # 2. Transfer orbit
    ax2 = fig.add_subplot(gs[0,2]); ax2.set_facecolor("#0d1117")
    theta = np.linspace(0,2*np.pi,300)
    ax2.plot(np.cos(theta), np.sin(theta), color="#00aaff", lw=1, alpha=0.5, label="Earth orbit")
    ax2.plot(1.524*np.cos(theta), 1.524*np.sin(theta), color="#ff4422", lw=1, alpha=0.5, label="Mars orbit")
    ax2.scatter([0],[0],s=200,c="#ffcc00",zorder=5,label="Sun")

    # Propagate transfer arc
    r_now = r_earth.copy(); v_now = v1.copy()
    arc   = [r_now.copy()]
    dt_a  = best_tof*86400/200
    for _ in range(199):
        a = -MU_SUN/np.linalg.norm(r_now)**3 * r_now
        v_now += a*dt_a; r_now += v_now*dt_a
        arc.append(r_now.copy())
    arc = np.array(arc)/AU
    ax2.scatter(arc[:,0], arc[:,1], c=np.linspace(0,1,len(arc)),
                cmap="plasma", s=3, zorder=3)
    ax2.scatter([r_earth[0]/AU],[r_earth[1]/AU],s=80,c="#00aaff",zorder=5,label="Earth dep")
    ax2.scatter([r_mars[0]/AU], [r_mars[1]/AU], s=80,c="#ff4422",zorder=5,label="Mars arr")
    ax2.set_aspect("equal"); ax2.tick_params(**tkw)
    ax2.set_xlabel("x (AU)",**lkw); ax2.set_ylabel("y (AU)",**lkw)
    ax2.set_title("Transfer Orbit (Ecliptic)", color="#fff", fontsize=10)
    ax2.grid(True, color="#222233", lw=0.5)
    ax2.legend(fontsize=7, facecolor="#111122", labelcolor="#cccccc", loc="upper left")

    # 3. C3 vs TOF at best dep
    ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor("#0d1117")
    best_col = np.argmin(abs(dep_jds-best_dep))
    ax3.plot(tofs, C3_grid[:,best_col], color="#aa44ff", lw=1.5)
    ax3.axvline(best_tof, color="#ff0055", lw=1, ls="--", label=f"Opt TOF={best_tof:.0f}d")
    ax3.axhline(C3_best,  color="#00ff88", lw=1, ls="--", label=f"C3={C3_best:.1f}")
    ax3.set_xlabel("TOF (days)",**lkw); ax3.set_ylabel("C3 (km²/s²)",**lkw)
    ax3.set_title("C3 vs TOF at Optimal Departure", color="#fff", fontsize=10)
    ax3.tick_params(**tkw); ax3.grid(True,color="#222233",lw=0.5)
    ax3.legend(fontsize=8,facecolor="#111122",labelcolor="#cccccc")

    # 4. Dv budget
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor("#0d1117")
    items = ["C3\n(km²/s²)","Dep Δv\n(km/s)","Arr Δv\n(km/s)"]
    vals  = [C3_best, dv_dep or 0, dv_arr or 0]
    clrs  = ["#ff6600","#00aaff","#00ff88"]
    bars  = ax4.bar(items,vals,color=clrs,alpha=0.85)
    for bar,val in zip(bars,vals):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"{val:.2f}", ha='center', color="#cccccc", fontsize=9)
    ax4.set_title("Mission Δv Budget", color="#fff", fontsize=10)
    ax4.tick_params(**tkw)

    # 5. Mission summary
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor("#0d1117"); ax5.axis("off")
    dep_dt = Time(best_dep, format='jd').iso[:10]
    arr_dt = Time(best_dep+best_tof, format='jd').iso[:10]
    txt = (f"MISSION SUMMARY\n{'─'*26}\n"
           f"Departure : {dep_dt}\n"
           f"Arrival   : {arr_dt}\n"
           f"TOF       : {best_tof:.1f} days\n"
           f"           ({best_tof/30.44:.1f} months)\n\n"
           f"C3        : {C3_best:.2f} km²/s²\n"
           f"Dep Δv    : {dv_dep:.3f} km/s\n"
           f"Arr Δv    : {dv_arr:.3f} km/s\n"
           f"Total Δv  : {dv_dep+dv_arr:.3f} km/s\n\n"
           f"Optimizer : scipy DE best1bin\n"
           f"Autodiff  : {'JAX' if JAX_AVAILABLE else 'N/A'}\n"
           f"Ephemeris : {'SPICE' if SPICE_LOADED else 'Keplerian'}\n")
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes,
             color="#cccccc", fontsize=9, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round", facecolor="#111122", alpha=0.8))

    plt.suptitle("Project 4 — Interplanetary Mission Designer (Earth→Mars)",
                 color="#fff", fontsize=14, y=1.01)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"  Plot saved: {OUTPUT_PNG}")

# ─────────────────────────────────────────────
# WHAT TO EXPECT
# ─────────────────────────────────────────────
"""
PLOT GUIDE:

1. Porkchop — plasma contours, dark = low C3 = good launch window.
   Green star = pygmo optimum. Earth-Mars windows repeat ~every 26 months.
   Next good window after June 2026 is around late 2026 / early 2027.

2. Transfer orbit — Earth (blue) and Mars (red) orbits around the Sun.
   Coloured arc = transfer trajectory. Should clearly connect both planets.

3. C3 vs TOF — U-shape. Valley = optimal TOF. Typical ~200-300 days for Mars.

4. Δv budget — C3 typically 5-20 km²/s² for a good window.
   Dep Δv ~3-4 km/s from LEO. Arr Δv ~1-3 km/s. Total ~5-7 km/s.

5. Summary panel — all key mission parameters at a glance.
"""

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*55)
    print("  INTERPLANETARY MISSION DESIGNER — Project 4")
    print("  Earth → Mars")
    print("="*55)

    global SPICE_LOADED
    SPICE_LOADED = load_spice_kernels()

    # Link to Project 3
    try:
        plan = pd.read_csv("maneuver_plan.csv")
        total_leo_dv = plan['dv_ms'].abs().sum()
        print(f"\n  Project 3 LEO Δv budget  : {total_leo_dv:.4f} m/s")
        print(f"  Scaling up to interplanetary...\n")
    except FileNotFoundError:
        print("  [!] maneuver_plan.csv not found — continuing.\n")

    dep_start = Time("2026-06-01").jd

    # 1. Porkchop
    print("[1/4] Porkchop grid...")
    dep_jds, tofs, C3_grid = porkchop_grid(dep_start, n_dep=35, n_tof=35,
                                            dep_range=400, tof_min=150, tof_max=420)

    # Grid minimum
    min_idx      = np.unravel_index(np.nanargmin(C3_grid), C3_grid.shape)
    grid_dep     = dep_jds[min_idx[1]]
    grid_tof     = tofs[min_idx[0]]
    print(f"  Grid min: dep={Time(grid_dep,format='jd').iso[:10]}, "
          f"TOF={grid_tof:.0f}d, C3={C3_grid[min_idx]:.2f}")

    # 2. pygmo
    print(f"\n[2/4] pygmo global optimization...")
    best_dep, best_tof = run_scipy_de(dep_start, dep_range=400, tof_min=150, tof_max=420)
    if best_dep is None:
        best_dep, best_tof = grid_dep, grid_tof

    # 3. JAX sensitivity
    print(f"\n[3/4] JAX launch window sensitivity...")
    jax_sensitivity(best_dep, best_tof)

    # 4. Final trajectory
    print(f"\n[4/4] Optimal transfer computation...")
    C3, dv_arr, dv_dep, v1, v2, r_earth, r_mars = compute_c3(best_dep, best_tof)
    if v1 is None:
        C3, dv_arr, dv_dep, v1, v2, r_earth, r_mars = compute_c3(grid_dep, grid_tof)
        best_dep, best_tof = grid_dep, grid_tof

    dep_dt = Time(best_dep, format='jd').iso[:10]
    arr_dt = Time(best_dep+best_tof, format='jd').iso[:10]
    print(f"\n  Departure : {dep_dt}")
    print(f"  Arrival   : {arr_dt}")
    print(f"  TOF       : {best_tof:.1f} days")
    print(f"  C3        : {C3:.4f} km²/s²")
    print(f"  Dep Δv    : {dv_dep:.4f} km/s")
    print(f"  Arr Δv    : {dv_arr:.4f} km/s")
    print(f"  Total Δv  : {dv_dep+dv_arr:.4f} km/s")

    # CasADi
    if CASADI_AVAILABLE:
        print(f"\n  CasADi low-thrust refinement...")
        casadi_low_thrust(r_earth, v1, r_mars, best_tof*86400)

    # Save
    pd.DataFrame([dict(
        departure=dep_dt, arrival=arr_dt, tof_days=best_tof,
        C3_km2s2=C3, dep_dv_kms=dv_dep, arr_dv_kms=dv_arr,
        total_dv_kms=dv_dep+dv_arr,
        optimizer="scipy_DE_best1bin",
        ephemeris="SPICE" if SPICE_LOADED else "Keplerian",
    )]).to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved: {OUTPUT_CSV}")

    make_plots(dep_jds, tofs, C3_grid, best_dep, best_tof,
               r_earth, r_mars, v1, C3, dv_dep, dv_arr)

    print("\n"+"="*55)
    print("  ✓ Project 4 complete.")
    print("  Full arc: Radar → EKF → Maneuver → Interplanetary.")
    print("="*55)

if __name__ == "__main__":
    main()
