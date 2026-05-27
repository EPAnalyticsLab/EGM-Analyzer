"""Unit tests for the local activation time (LAT) estimator."""

import numpy as np
import pytest

from signal_processing import compute_lat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_unipolar(t_peak: int, n: int = 500, sharpness: float = 6.0) -> np.ndarray:
    """Synthetic unipolar activation centred at ``t_peak``.

    Models the canonical biphasic unipolar morphology: a smooth fall to a
    negative peak followed by a rebound. The maximum negative derivative
    is, by construction, located at ``t_peak``.
    """
    x = np.arange(n)
    z = (x - t_peak) / sharpness
    # Negative Gaussian: minimum derivative coincides with the inflexion
    # point just before the minimum value (i.e., t_peak).
    return -np.exp(-(z ** 2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lat_on_constant_signal_returns_zero():
    s = np.zeros(100)
    assert compute_lat(s) == 0.0


def test_lat_on_all_nans_returns_zero():
    s = np.full(100, np.nan)
    assert compute_lat(s) == 0.0


def test_lat_short_signal_returns_zero():
    s = np.array([1.0, 2.0])
    assert compute_lat(s) == 0.0


def test_lat_is_finite_on_synthetic_unipolar():
    s = synthetic_unipolar(t_peak=200)
    lat = compute_lat(s)
    assert np.isfinite(lat)
    assert 0 <= lat < len(s)


def test_lat_locates_maximum_negative_derivative():
    """Built-in: argmin of np.gradient(s) MUST equal compute_lat for finite signals."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal(500)
    expected = float(np.argmin(np.gradient(s)))
    assert compute_lat(s) == expected


def test_lat_handles_partial_nans():
    """LAT must be returned in the original (padded) index space."""
    s = synthetic_unipolar(t_peak=200)
    s_padded = np.concatenate([np.full(50, np.nan), s, np.full(20, np.nan)])
    lat_clean = compute_lat(s)
    lat_padded = compute_lat(s_padded)
    # The padded LAT should be shifted by exactly the NaN prefix length.
    assert lat_padded == pytest.approx(lat_clean + 50, abs=1.0)


def test_lat_responds_to_temporal_shift():
    """Shifting the activation in time must shift the LAT by the same amount."""
    s1 = synthetic_unipolar(t_peak=150)
    s2 = synthetic_unipolar(t_peak=300)
    assert compute_lat(s2) - compute_lat(s1) == pytest.approx(150, abs=2.0)
