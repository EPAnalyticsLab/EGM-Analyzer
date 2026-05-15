"""Unit tests for the omnipolar electrogram and Residue-to-Omnipole Ratio."""

import math

import numpy as np
import pytest

from signal_processing import (
    compute_omnipolar,
    compute_ror,
    compute_vpp,
)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

def test_omnipolar_recovers_amplitude_for_axis_aligned_signal():
    """If b(t) lies entirely on the x-axis, the omnipolar trace equals b_x."""
    bx = np.sin(np.linspace(0, 4 * np.pi, 1000))
    by = np.zeros_like(bx)
    omni, residue, angle, vec = compute_omnipolar(bx, by)
    assert compute_vpp(omni) == pytest.approx(compute_vpp(bx), rel=1e-9)
    # Residue along an axis-aligned trace must be ~zero up to numerical noise.
    assert compute_vpp(residue) == pytest.approx(0.0, abs=1e-9)


def test_omnipolar_recovers_amplitude_for_diagonal_propagation():
    """If b(t) = A·sin(t)·(1/√2, 1/√2), Vpp(omni) = 2A exactly."""
    n = 2000
    t = np.linspace(0, 2 * np.pi, n)
    A = 1.7
    bx = (A / math.sqrt(2)) * np.sin(t)
    by = (A / math.sqrt(2)) * np.sin(t)
    omni, residue, angle, _ = compute_omnipolar(bx, by)
    assert compute_vpp(omni) == pytest.approx(2 * A, rel=1e-4)
    assert compute_vpp(residue) == pytest.approx(0.0, abs=1e-8)
    # The recovered direction must be along the 45° diagonal.
    assert abs(angle) == pytest.approx(45.0, abs=1.0) or \
           abs(abs(angle) - 135.0) < 1.0


def test_omnipolar_orientation_independence():
    """Two 2-D traces that differ only by a global rotation must yield
    identical omnipolar Vpp values."""
    n = 1500
    t = np.linspace(0, 2 * np.pi, n)
    bx0 = np.sin(t)
    by0 = 0.4 * np.cos(t)

    # Reference Vpp
    omni0, _, _, _ = compute_omnipolar(bx0, by0)
    v0 = compute_vpp(omni0)

    # Rotate the trace by several angles and re-evaluate.
    for theta_deg in [0, 30, 45, 90, 137, 200]:
        theta = math.radians(theta_deg)
        c, s = math.cos(theta), math.sin(theta)
        bxr = c * bx0 - s * by0
        byr = s * bx0 + c * by0
        omni_r, _, _, _ = compute_omnipolar(bxr, byr)
        assert compute_vpp(omni_r) == pytest.approx(v0, rel=1e-6)


def test_omnipolar_residue_orthogonal_to_omni_direction():
    """omni and residue components should be (essentially) decorrelated for a
    well-defined linear propagation."""
    n = 3000
    t = np.linspace(0, 2 * np.pi, n)
    bx = np.sin(t)
    by = 0.3 * np.cos(t)
    omni, residue, _, _ = compute_omnipolar(bx, by)
    omni = omni - omni.mean()
    residue = residue - residue.mean()
    inner = float(np.abs(np.dot(omni, residue) / n))
    # The rotation explicitly aligns omni with the major axis of the
    # data scatter, so the residue carries the minor axis only.
    assert inner < 1e-3 * compute_vpp(omni) ** 2


def test_omnipolar_zero_input_returns_zero_outputs():
    bx = np.zeros(500)
    by = np.zeros(500)
    omni, residue, angle, vec = compute_omnipolar(bx, by)
    assert np.allclose(omni, 0.0)
    assert np.allclose(residue, 0.0)
    assert angle == 0.0


# ---------------------------------------------------------------------------
# Residue-to-Omnipole Ratio
# ---------------------------------------------------------------------------

def test_ror_is_zero_for_pure_linear_propagation():
    """A purely linear b(t) ⇒ residue = 0 ⇒ ROR = 0."""
    t = np.linspace(0, 2 * np.pi, 1500)
    bx = np.sin(t)
    by = np.zeros_like(bx)
    omni, residue, _, _ = compute_omnipolar(bx, by)
    assert compute_ror(omni, residue) == pytest.approx(0.0, abs=1e-9)


def test_ror_is_close_to_one_for_circular_propagation():
    """For an isotropic circular trace b(t) = (cos t, sin t), the major and
    minor axes have equal Vpp ⇒ ROR ≈ 1."""
    t = np.linspace(0, 2 * np.pi, 4000)
    bx = np.cos(t)
    by = np.sin(t)
    omni, residue, _, _ = compute_omnipolar(bx, by)
    assert compute_ror(omni, residue) == pytest.approx(1.0, rel=5e-3)


def test_ror_returns_nan_for_silent_clique():
    omni = np.zeros(100)
    residue = np.zeros(100)
    assert math.isnan(compute_ror(omni, residue))
