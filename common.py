"""
common.py - shared constants, frame transforms and the single GMST used everywhere
===================================================================================
Why this file exists
--------------------
In the old project every script re-defined its own constants and frame helpers, and
the EKF used a *different* GMST (astropy) from the simulator (a hand formula).  Any
mismatch between "the frame the data was made in" and "the frame the filter assumes"
shows up as a bias in the estimate.  Here there is ONE definition of each thing.

Frame convention (important - an interviewer WILL ask):
  ECI  = TEME, the frame SGP4 outputs.  It is inertial for our purposes.
  ECEF = TEME rotated about Z by GMST.  (Polar motion and nutation are ignored;
         that is a ~10 m effect, far below the 50 m range noise.)
  ENU  = local East-North-Up at the ground station.

Units: metres, seconds, radians internally.  Degrees only in CSV files.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
MU_EARTH = 3.986004418e14          # m^3/s^2
RE_M = 6_378_137.0                 # WGS84 equatorial radius, m
FE = 1.0 / 298.257223563           # WGS84 flattening
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5             # Earth rotation rate, rad/s
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Ground station (Chennai) - change here, every phase picks it up
GS_LAT_DEG = 13.0827
GS_LON_DEG = 80.2707
GS_ALT_M = 6.0

# Radar noise (1-sigma).  Phase 1 uses these to CREATE noise, Phase 2 to BUILD R.
SIGMA_RANGE_M = 50.0
SIGMA_AZ_DEG = 0.05
SIGMA_EL_DEG = 0.05

ELEVATION_MASK_DEG = 5.0


# ---------------------------------------------------------------- time / GMST
def gmst_rad(jd_ut1):
    """Greenwich Mean Sidereal Time (rad) from a Julian Date (UT1 ~ UTC here).

    Vallado eq. 3-47.  Accepts scalars or numpy arrays.
    """
    jd_ut1 = np.asarray(jd_ut1, dtype=float)
    T = (jd_ut1 - 2_451_545.0) / 36_525.0
    gmst_sec = (67310.54841
                + (876_600.0 * 3600.0 + 8_640_184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    return np.mod(gmst_sec, 86_400.0) / 86_400.0 * 2.0 * np.pi


def rot_eci_to_ecef(gmst):
    """3x3 matrix R such that r_ecef = R @ r_eci."""
    c, s = np.cos(gmst), np.sin(gmst)
    return np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def eci_to_ecef_state(r_eci, v_eci, gmst):
    """Position AND velocity ECI -> ECEF.  Velocity needs the omega x r term."""
    R = rot_eci_to_ecef(gmst)
    r_ecef = R @ r_eci
    v_ecef = R @ v_eci - np.cross([0.0, 0.0, OMEGA_E], r_ecef)
    return r_ecef, v_ecef


def ecef_to_eci_state(r_ecef, v_ecef, gmst):
    """Inverse of eci_to_ecef_state."""
    R = rot_eci_to_ecef(gmst).T
    r_eci = R @ r_ecef
    v_eci = R @ (v_ecef + np.cross([0.0, 0.0, OMEGA_E], r_ecef))
    return r_eci, v_eci


# ---------------------------------------------------------------- geometry
def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = lat_deg * DEG2RAD, lon_deg * DEG2RAD
    N = RE_M / np.sqrt(1.0 - FE * (2.0 - FE) * np.sin(lat) ** 2)
    return np.array([(N + alt_m) * np.cos(lat) * np.cos(lon),
                     (N + alt_m) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - FE) ** 2 + alt_m) * np.sin(lat)])


def enu_matrix(lat_deg, lon_deg):
    """Rows are East, North, Up unit vectors expressed in ECEF: enu = R @ delta_ecef."""
    lat, lon = lat_deg * DEG2RAD, lon_deg * DEG2RAD
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])


def enu_to_rae(enu):
    """ENU -> (range m, azimuth rad in [0,2pi), elevation rad)."""
    e, n, u = enu
    rng = np.linalg.norm(enu)
    return rng, np.arctan2(e, n) % (2 * np.pi), np.arcsin(u / rng)


def rae_to_enu(rng, az, el):
    return np.array([rng * np.cos(el) * np.sin(az),
                     rng * np.cos(el) * np.cos(az),
                     rng * np.sin(el)])


def ecef_to_geodetic_approx(r_ecef):
    """Quick (lat_deg, lon_deg, alt_m) - spherical-ish, good enough for plots."""
    x, y, z = r_ecef
    lon = np.arctan2(y, x) * RAD2DEG
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - FE)) * RAD2DEG
    alt = np.linalg.norm(r_ecef) - RE_M
    return lat, lon, alt


class Station:
    """Ground station helper: caches ECEF position and ENU rotation."""

    def __init__(self, lat_deg=GS_LAT_DEG, lon_deg=GS_LON_DEG, alt_m=GS_ALT_M):
        self.lat_deg, self.lon_deg, self.alt_m = lat_deg, lon_deg, alt_m
        self.ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        self.R_enu = enu_matrix(lat_deg, lon_deg)

    def rae_from_eci(self, r_eci, jd):
        """True range/az/el of an ECI position at Julian date jd."""
        R = rot_eci_to_ecef(gmst_rad(jd))
        return enu_to_rae(self.R_enu @ (R @ r_eci - self.ecef))

    def eci_from_rae(self, rng, az, el, jd):
        """Radar measurement -> ECI position."""
        r_ecef = self.ecef + self.R_enu.T @ rae_to_enu(rng, az, el)
        return rot_eci_to_ecef(gmst_rad(jd)).T @ r_ecef


# ---------------------------------------------------------------- keplerian
def state_to_keplerian(r, v, mu=MU_EARTH):
    """Vallado Algorithm 9 (RV2COE).  Returns dict, angles in degrees.

    Handles the circular / equatorial singularities by falling back to
    longitude of periapsis / true longitude style conventions only where needed;
    for our near-circular orbits we report u (argument of latitude) as well.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rm, vm = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hm = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    nm = np.linalg.norm(n)
    evec = ((vm ** 2 - mu / rm) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(evec)
    energy = vm ** 2 / 2.0 - mu / rm
    a = -mu / (2.0 * energy)
    inc = np.arccos(np.clip(h[2] / hm, -1, 1))

    raan = 0.0
    if nm > 1e-9:
        raan = np.arccos(np.clip(n[0] / nm, -1, 1))
        if n[1] < 0:
            raan = 2 * np.pi - raan

    aop = 0.0
    if nm > 1e-9 and e > 1e-9:
        aop = np.arccos(np.clip(np.dot(n, evec) / (nm * e), -1, 1))
        if evec[2] < 0:
            aop = 2 * np.pi - aop

    nu = 0.0
    if e > 1e-9:
        nu = np.arccos(np.clip(np.dot(evec, r) / (e * rm), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu

    # argument of latitude (well defined even when e ~ 0)
    u = 0.0
    if nm > 1e-9:
        u = np.arccos(np.clip(np.dot(n, r) / (nm * rm), -1, 1))
        if r[2] < 0:
            u = 2 * np.pi - u

    return dict(a=a, e=e, i_deg=inc * RAD2DEG, raan_deg=raan * RAD2DEG,
                aop_deg=aop * RAD2DEG, nu_deg=nu * RAD2DEG, u_deg=u * RAD2DEG,
                period_s=2 * np.pi * np.sqrt(abs(a) ** 3 / mu),
                alt_km=(a - RE_M) / 1000.0, rp=a * (1 - e), ra=a * (1 + e))


def classify_orbit(period_min: float, ecc: float, inc_deg: float) -> str:
    """Rough orbit class from period.  Used to build a mixed test set."""
    if period_min < 128.0:
        return "LEO"
    if 1380.0 < period_min < 1500.0 and ecc < 0.05 and inc_deg < 20.0:
        return "GEO"
    if period_min < 1380.0 and ecc < 0.3:
        return "MEO"
    return "OTHER"
