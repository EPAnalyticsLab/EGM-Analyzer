"""Unit tests for the peak-to-peak voltage estimator."""

import numpy as np
import pytest

from signal_processing import compute_vpp


def test_vpp_constant_signal_is_zero():
    assert compute_vpp(np.zeros(100)) == 0.0
    assert compute_vpp(np.full(50, 3.7)) == 0.0


def test_vpp_all_nans_returns_zero():
    assert compute_vpp(np.full(100, np.nan)) == 0.0


def test_vpp_matches_analytical_value_for_sine():
    """For ``A·sin(·)``, Vpp = 2·A exactly when the period is sampled densely."""
    t = np.linspace(0, 2 * np.pi, 10_000)
    for amplitude in [0.5, 1.0, 2.5, 10.0]:
        signal = amplitude * np.sin(t)
        assert compute_vpp(signal) == pytest.approx(2.0 * amplitude, rel=1e-4)


def test_vpp_offset_invariant():
    """``Vpp`` must be invariant to a constant DC offset."""
    base = np.sin(np.linspace(0, 4 * np.pi, 1000))
    v0 = compute_vpp(base)
    for offset in [-3.0, 0.0, 5.5]:
        assert compute_vpp(base + offset) == pytest.approx(v0, rel=1e-9)


def test_vpp_scales_linearly():
    """``Vpp(k·s) = |k|·Vpp(s)`` for any scalar ``k``."""
    rng = np.random.default_rng(42)
    s = rng.standard_normal(2000)
    v0 = compute_vpp(s)
    for k in [0.1, 1.0, 3.7, 10.0]:
        assert compute_vpp(k * s) == pytest.approx(k * v0, rel=1e-9)


def test_vpp_ignores_nans():
    s = np.array([1.0, np.nan, -1.0, np.nan, 0.5])
    assert compute_vpp(s) == pytest.approx(2.0)


def test_vpp_known_min_max():
    s = np.array([-2.0, 0.0, 5.0, 1.5, -0.3])
    assert compute_vpp(s) == pytest.approx(7.0)
