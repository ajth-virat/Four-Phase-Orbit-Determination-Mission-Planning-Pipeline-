"""
ephemeris.py - heliocentric Earth / Mars states in the ecliptic J2000 frame
===========================================================================
WHY THIS REPLACES THE OLD EPHEMERIS
    The old Keplerian approximation placed the orbit with the wrong angle: it used the
    true anomaly directly as the heliocentric longitude and forgot the longitude of
    perihelion, so Earth was ~103 degrees and Mars ~24 degrees away from where they
    really are (up to 1.55 AU of position error).  Every C3 downstream was meaningless.

NOW
    * Positions come from astropy's built-in solar-system ephemeris (ERFA planetary
      theory, works fully offline).  Accuracy is good to the level a porkchop needs.
    * If you have a JPL kernel (de440s.bsp) and spiceypy, pass kernel="path.bsp" and it is
      used instead (higher accuracy).  Not required.
    * States are tabulated on a regular grid and interpolated with a cubic spline, so the
      optimiser can call it thousands of times cheaply.  Velocity is the spline
      derivative; tests/ checks it against astropy's own velocity.

Frame: heliocentric, mean ecliptic and equinox J2000, so "prograde" == counter-clockwise
about +Z and the planets move in an almost-flat plane.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

AU = 1.495978707e11
MU_SUN = 1.32712440018e20
EPS_J2000 = np.radians(23.439291111)           # mean obliquity at J2000
# rotate ICRS (equatorial) -> ecliptic J2000
R_EQ2ECL = np.array([[1, 0, 0],
                     [0, np.cos(EPS_J2000), np.sin(EPS_J2000)],
                     [0, -np.sin(EPS_J2000), np.cos(EPS_J2000)]])


def _astropy_states(body, jd_utc):
    """Heliocentric ICRS -> ecliptic position (m) and velocity (m/s) from astropy."""
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
    solar_system_ephemeris.set("builtin")
    t = Time(np.atleast_1d(jd_utc), format="jd", scale="utc")
    pb, vb = get_body_barycentric_posvel(body, t)
    ps, vs = get_body_barycentric_posvel("sun", t)
    r = (pb - ps).xyz.to("m").value.T
    v = (vb - vs).xyz.to("m/s").value.T
    return r @ R_EQ2ECL.T, v @ R_EQ2ECL.T


def _spice_states(body, jd_utc, kernel):
    import spiceypy as spice
    spice.furnsh(kernel)
    out_r, out_v = [], []
    for jd in np.atleast_1d(jd_utc):
        et = spice.str2et(f"JD {jd:.9f} UTC")
        s, _ = spice.spkezr(body.upper(), et, "ECLIPJ2000", "NONE", "SUN")
        out_r.append(np.array(s[:3]) * 1e3)
        out_v.append(np.array(s[3:]) * 1e3)
    return np.array(out_r), np.array(out_v)


class Ephemeris:
    """Interpolated heliocentric states for a set of bodies over [jd_start, jd_end]."""

    def __init__(self, jd_start, jd_end, bodies=("earth", "mars"), step_days=1.0, kernel=None):
        self.jd0, self.jd1 = float(jd_start), float(jd_end)
        self.kind = "SPICE kernel" if kernel else "astropy built-in (ERFA)"
        grid = np.arange(self.jd0 - 2 * step_days, self.jd1 + 3 * step_days, step_days)
        self._spl = {}
        for b in bodies:
            r, _ = (_spice_states(b, grid, kernel) if kernel else _astropy_states(b, grid))
            self._spl[b] = CubicSpline(grid, r, axis=0)

    def state(self, body, jd):
        """(r, v) in metres and m/s, heliocentric ecliptic J2000."""
        spl = self._spl[body]
        return spl(jd), spl(jd, 1) / 86400.0       # derivative is per day -> per second

    def positions(self, body, jds):
        return self._spl[body](np.asarray(jds))
