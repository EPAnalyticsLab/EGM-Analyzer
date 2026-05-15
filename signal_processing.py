"""
EGM Analyzer — core signal-processing routines.

This module exposes the algorithmic primitives used by the Dash GUI
(``main.py``) as pure, dependency-light functions so they can be imported
in isolation by unit tests and validation scripts.

All functions take and return NumPy arrays (or plain Python scalars) and
have no side effects.
"""

from __future__ import annotations

import math
import re

import numpy as np


# ---------------------------------------------------------------------------
# Scalar metrics on a single 1-D signal
# ---------------------------------------------------------------------------

def compute_lat(signal: np.ndarray) -> float:
    """Local activation time (LAT) of a unipolar signal.

    Implements the standard ``-dV/dt_max`` criterion: LAT is the sample
    index at which the time derivative is most negative.

    Parameters
    ----------
    signal : array_like
        1-D unipolar electrogram. May contain NaNs.

    Returns
    -------
    float
        Sample index (NOT milliseconds) of the maximum negative
        derivative on the finite portion of the signal. Returns ``0.0``
        if fewer than three finite samples are available.
    """
    s = np.asarray(signal, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 3:
        return 0.0
    s_valid = s[valid]
    grad = np.gradient(s_valid)
    return float(np.where(valid)[0][np.argmin(grad)])


def compute_vpp(signal: np.ndarray) -> float:
    """Peak-to-peak voltage ``max(s) - min(s)`` on the finite samples."""
    s = np.asarray(signal, dtype=float)
    valid = s[np.isfinite(s)]
    if len(valid) == 0:
        return 0.0
    return float(valid.max() - valid.min())


# ---------------------------------------------------------------------------
# Omnipolar electrogram from a bipolar pair (clique)
# ---------------------------------------------------------------------------

def compute_omnipolar(bip_x: np.ndarray, bip_y: np.ndarray):
    """Omnipolar projection of a 2-D bipolar electric-field trace.

    Given two orthogonal bipolar signals (e.g. ``b_h(t)``, ``b_v(t)``)
    forming a 2-D electric-field trajectory in the local clique plane,
    this routine:

    1. Identifies the direction of maximum excursion in that plane.
    2. Aligns the trajectory with that direction by a 2-D rotation.
    3. Returns the rotated coordinates:
       * ``omni`` — the component along the maximum-excursion direction
         (the omnipolar electrogram ``o(t)``).
       * ``residue`` — the SIGNED component orthogonal to ``omni`` in
         the clique plane, i.e. the projection of ``b(t)`` onto the
         in-plane vector ``w^perp``.

    This formulation matches the implementation expected by the GUI:
    ``residue`` is signed (so that ``Vpp(residue) = max - min`` is
    meaningful) and not the Euclidean norm of the residual vector.

    Parameters
    ----------
    bip_x, bip_y : array_like
        1-D bipolar signals defining a 2-D trajectory ``b(t)``.

    Returns
    -------
    omni : ndarray
        Omnipolar signal ``o(t)``.
    residue : ndarray
        In-plane orthogonal component ``r(t)`` (signed).
    angle_deg : float
        Angle of the omnipolar direction in degrees.
    vector : ndarray, shape (2,)
        Maximum-excursion vector ``w`` scaled by its magnitude.
    """
    bx = np.asarray(bip_x, dtype=float)
    by = np.asarray(bip_y, dtype=float)
    pts = np.column_stack([bx, by])
    dists = np.linalg.norm(pts, axis=1)
    idx = int(np.argmax(dists))
    direction = pts[idx].copy()
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return bx, by, 0.0, np.array([0.0, 0.0])
    direction /= norm
    c, s = direction[0], direction[1]
    R = np.array([[c, s], [-s, c]])           # rotation aligning w with x-axis
    rotated = (R @ pts.T).T
    angle = math.degrees(math.atan2(s, c))
    return rotated[:, 0], rotated[:, 1], angle, direction * dists[idx]


def compute_ror(omni: np.ndarray, residue: np.ndarray) -> float:
    """Residue-to-Omnipole Ratio (ROR) = ``Vpp(residue) / Vpp(omni)``.

    Returns ``nan`` if ``Vpp(omni)`` is numerically zero.
    """
    vpp_omni = compute_vpp(omni)
    vpp_res = compute_vpp(residue)
    if vpp_omni < 1e-9:
        return float("nan")
    return float(vpp_res / vpp_omni)


# ---------------------------------------------------------------------------
# Grid helpers (4x4 multi-electrode catheter)
# ---------------------------------------------------------------------------

_COORDS_RE = re.compile(r"([A-D])([1-4])")


def label_to_grid(label: str):
    """Map an electrode label like ``A1``/``D4`` to a ``(row, col)`` index.

    Rows are indexed by the letter (``A``→0 … ``D``→3) and columns by the
    digit (``1``→0 … ``4``→3). Returns ``(None, None)`` if the label
    does not match the expected pattern.
    """
    m = _COORDS_RE.search(str(label))
    if not m:
        return None, None
    return ord(m.group(1)) - ord("A"), int(m.group(2)) - 1
