"""
lambert.py - Lambert's problem by universal variables (Bate-Mueller-White / Curtis Alg. 5.2)
============================================================================================
Given two position vectors and a time of flight, find the conic that connects them.

WHY THIS REPLACES THE OLD SOLVER
    The old "Izzo" routine was hand-rolled and wrong: it missed Curtis Example 5.2 by
    0.1 km/s and its trajectories missed Mars by ~272 million km.  This version is the
    textbook algorithm, root-found robustly (bracketed Brent, no Newton blow-ups), and is
    verified in tests/ against Curtis Example 5.2 AND by propagating the answer.

Scope: single revolution, prograde or retrograde.  Multi-revolution solutions are not
implemented (Earth-Mars transfers of 100-450 days do not need them).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_Z_GRID = np.concatenate([np.linspace(-60.0, -1e-3, 300),
                          np.linspace(0.0, 4 * np.pi ** 2 - 1e-3, 900)])


def _stumpff(z):
    """Stumpff functions C(z), S(z), vectorised and safe near z = 0."""
    z = np.asarray(z, dtype=float)
    C = np.empty_like(z)
    S = np.empty_like(z)
    pos, neg = z > 1e-8, z < -1e-8
    mid = ~(pos | neg)
    sp = np.sqrt(z[pos])
    C[pos] = (1 - np.cos(sp)) / z[pos]
    S[pos] = (sp - np.sin(sp)) / sp ** 3
    sn = np.sqrt(-z[neg])
    C[neg] = (np.cosh(sn) - 1) / (-z[neg])
    S[neg] = (np.sinh(sn) - sn) / sn ** 3
    C[mid] = 0.5 - z[mid] / 24.0
    S[mid] = 1.0 / 6.0 - z[mid] / 120.0
    return C, S


def lambert_universal(r1, r2, tof, mu, prograde=True):
    """Return (v1, v2) in the same length/time units as the inputs.

    Raises ValueError if no single-revolution solution exists for this tof.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1m, r2m = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (r1m * r2m), -1.0, 1.0)
    cz = np.cross(r1, r2)[2]
    dnu = np.arccos(cos_dnu)
    if prograde and cz < 0:
        dnu = 2 * np.pi - dnu
    elif (not prograde) and cz >= 0:
        dnu = 2 * np.pi - dnu

    denom = 1.0 - np.cos(dnu)
    if denom < 1e-12:
        raise ValueError("degenerate geometry (transfer angle 0)")
    A = np.sin(dnu) * np.sqrt(r1m * r2m / denom)
    if abs(A) < 1e-9 * (r1m + r2m):
        raise ValueError("transfer angle of 180 deg: orbit plane undefined")

    sqmu = np.sqrt(mu)

    def y_of(z, C, S):
        return r1m + r2m + A * (z * S - 1.0) / np.sqrt(C)

    C, S = _stumpff(_Z_GRID)
    y = y_of(_Z_GRID, C, S)
    ok = y > 0
    F = np.full_like(_Z_GRID, np.nan)
    F[ok] = (y[ok] / C[ok]) ** 1.5 * S[ok] + A * np.sqrt(y[ok]) - sqmu * tof

    def F_scalar(z):
        c, s = _stumpff(np.array([z]))
        yy = y_of(z, c[0], s[0])
        return (yy / c[0]) ** 1.5 * s[0] + A * np.sqrt(yy) - sqmu * tof

    z_sol = None
    for k in range(len(_Z_GRID) - 1):
        if ok[k] and ok[k + 1] and F[k] * F[k + 1] < 0:
            z_sol = brentq(F_scalar, _Z_GRID[k], _Z_GRID[k + 1], xtol=1e-14, rtol=1e-13, maxiter=200)
            break
    if z_sol is None:
        raise ValueError("no single-revolution Lambert solution for this time of flight")

    c, s = _stumpff(np.array([z_sol]))
    yy = y_of(z_sol, c[0], s[0])
    f = 1.0 - yy / r1m
    g = A * np.sqrt(yy / mu)
    gdot = 1.0 - yy / r2m
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2
