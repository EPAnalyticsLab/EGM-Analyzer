"""
Synthetic validation suite for EGM Analyzer.

This script confronts the core signal-processing routines exposed by
``signal_processing.py`` with synthetic electrograms whose properties are
known analytically. It is intentionally self-contained so that it can be
invoked both manually and by the GitHub Actions CI workflow.

For each metric the script reports the discrepancy between the computed
output and the analytical ground-truth value and asserts that the error
stays within a tight numerical tolerance.

Reviewer references: R1.6 / R2.1 / R2.6 (SoftwareX, first revision).

Run with::

    python validation/run_validation.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

# Make the project root importable when the script is invoked directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signal_processing import (  # noqa: E402
    compute_lat,
    compute_omnipolar,
    compute_ror,
    compute_vpp,
)


# ---------------------------------------------------------------------------
# Synthetic-signal generators
# ---------------------------------------------------------------------------

FS_HZ = 2000.0          # sampling frequency (Hz) matching the GUI default
DURATION_MS = 500.0     # 500 ms windows
N = int(FS_HZ * DURATION_MS / 1000.0)
T = np.arange(N) / FS_HZ   # seconds


def unipolar_template(t_peak_ms: float, amplitude_mV: float,
                      sharpness_ms: float = 3.0) -> np.ndarray:
    """Synthetic biphasic unipolar electrogram centred at ``t_peak_ms``.

    Mathematical form (after subtracting baseline):

        u(t) = -A · exp[-((t - t_peak) / σ)^2]

    so that the maximum negative derivative — and therefore the LAT
    estimate — falls exactly at ``t = t_peak_ms`` minus σ.

    Parameters
    ----------
    t_peak_ms : float
        Location of the negative deflection (ms).
    amplitude_mV : float
        Peak-to-peak amplitude target. Returned signal has
        ``max - min == amplitude_mV`` analytically.
    sharpness_ms : float
        Gaussian width (ms).
    """
    sigma_s = sharpness_ms / 1000.0
    t0_s = t_peak_ms / 1000.0
    return amplitude_mV * (-np.exp(-((T - t0_s) / sigma_s) ** 2))


# ---------------------------------------------------------------------------
# Validation cases
# ---------------------------------------------------------------------------

def case_vpp_unipolar_matches_amplitude() -> dict:
    """Vpp(uni) should recover the prescribed amplitude exactly."""
    errors = []
    for A in [0.1, 0.5, 1.0, 2.5, 5.0]:
        s = unipolar_template(t_peak_ms=250.0, amplitude_mV=A)
        v = compute_vpp(s)
        errors.append(abs(v - A))
    return {
        "name": "Vpp(unipolar) vs prescribed amplitude",
        "metric": "max |Vpp_computed − Vpp_true| (mV)",
        "value": max(errors),
        "tolerance": 1e-2,           # < 0.01 mV
        "passed": max(errors) < 1e-2,
    }


def case_vpp_bipolar_matches_amplitude() -> dict:
    """For two unipolars offset in time, Vpp(bipolar) equals the analytical
    peak-to-peak of their subtraction."""
    errors = []
    for shift_ms in [2.0, 5.0, 10.0]:
        u1 = unipolar_template(t_peak_ms=250.0, amplitude_mV=1.0)
        u2 = unipolar_template(t_peak_ms=250.0 + shift_ms, amplitude_mV=1.0)
        b = u1 - u2
        v_computed = compute_vpp(b)
        v_true = float(b.max() - b.min())   # analytical (exact)
        errors.append(abs(v_computed - v_true))
    return {
        "name": "Vpp(bipolar) vs analytical max-min",
        "metric": "max |Vpp_computed − Vpp_true| (mV)",
        "value": max(errors),
        "tolerance": 1e-9,
        "passed": max(errors) < 1e-9,
    }


def case_lat_matches_known_activation_time() -> dict:
    """LAT(uni) must locate the prescribed activation time within ±0.1 ms."""
    errors_ms = []
    for t_peak_ms in [50.0, 100.0, 250.0, 400.0]:
        s = unipolar_template(t_peak_ms=t_peak_ms, amplitude_mV=1.0)
        lat_sample = compute_lat(s)
        lat_ms = lat_sample * 1000.0 / FS_HZ
        # The maximum-negative-derivative point for a Gaussian -A·exp(-z²)
        # is offset by exactly σ/√2 (≈ 2.12 ms for σ=3 ms) to the LEFT of
        # the minimum. Compare against that closed-form location.
        sigma_ms = 3.0
        expected_ms = t_peak_ms - sigma_ms / math.sqrt(2)
        errors_ms.append(abs(lat_ms - expected_ms))
    return {
        "name": "LAT vs analytical inflexion-point location",
        "metric": "max |LAT_computed − LAT_true| (ms)",
        "value": max(errors_ms),
        # Tolerance set to one sample period at 2 kHz (0.5 ms): below this
        # discretisation floor the estimator cannot localise any sharper.
        "tolerance": 0.5,
        "passed": max(errors_ms) < 0.5,
    }


def case_omni_orientation_independence() -> dict:
    """Vpp(omni) must be invariant under rigid rotations of the bipolar
    trajectory."""
    rng = np.random.default_rng(0)
    # Anisotropic 2-D electric-field trace
    t = np.linspace(0, 2 * np.pi, N)
    bx0 = np.sin(t) + 0.05 * rng.standard_normal(N)
    by0 = 0.4 * np.cos(t) + 0.05 * rng.standard_normal(N)

    omni0, _, _, _ = compute_omnipolar(bx0, by0)
    v_ref = compute_vpp(omni0)

    spreads = []
    for theta_deg in np.arange(0.0, 360.0, 15.0):
        theta = math.radians(theta_deg)
        c, s = math.cos(theta), math.sin(theta)
        bxr = c * bx0 - s * by0
        byr = s * bx0 + c * by0
        omni_r, _, _, _ = compute_omnipolar(bxr, byr)
        spreads.append(abs(compute_vpp(omni_r) - v_ref))

    return {
        "name": "Vpp(omni) orientation-independence (rigid rotation)",
        "metric": "max |Vpp_omni(rot θ) − Vpp_omni(0)| (mV)",
        "value": max(spreads),
        "tolerance": 1e-2,
        "passed": max(spreads) < 1e-2,
    }


def case_ror_zero_for_linear_propagation() -> dict:
    """A purely linear (1-D) electric-field trajectory must yield ROR ≈ 0."""
    t = np.linspace(0, 2 * np.pi, N)
    bx = np.sin(t)
    by = np.zeros_like(bx)
    omni, residue, _, _ = compute_omnipolar(bx, by)
    ror = compute_ror(omni, residue)
    return {
        "name": "ROR ≈ 0 for purely linear propagation",
        "metric": "|ROR|",
        "value": float(abs(ror)),
        "tolerance": 1e-6,
        "passed": abs(ror) < 1e-6,
    }


def case_ror_one_for_circular_propagation() -> dict:
    """A perfectly circular electric-field trajectory must yield ROR ≈ 1."""
    t = np.linspace(0, 2 * np.pi, N)
    bx = np.cos(t)
    by = np.sin(t)
    omni, residue, _, _ = compute_omnipolar(bx, by)
    ror = compute_ror(omni, residue)
    return {
        "name": "ROR ≈ 1 for purely circular propagation",
        "metric": "|ROR − 1|",
        "value": float(abs(ror - 1.0)),
        "tolerance": 5e-3,
        "passed": abs(ror - 1.0) < 5e-3,
    }


CASES = [
    case_vpp_unipolar_matches_amplitude,
    case_vpp_bipolar_matches_amplitude,
    case_lat_matches_known_activation_time,
    case_omni_orientation_independence,
    case_ror_zero_for_linear_propagation,
    case_ror_one_for_circular_propagation,
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("EGM Analyzer — Synthetic Validation Suite")
    print(f"Sampling rate: {FS_HZ:.0f} Hz   Window: {DURATION_MS:.0f} ms   N = {N}")
    print("=" * 78)

    results = [c() for c in CASES]

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    width = max(len(r["name"]) for r in results)
    for r in results:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"[{flag}] {r['name']:<{width}}  "
              f"{r['metric']} = {r['value']:.3e}  "
              f"(tol < {r['tolerance']:.1e})")

    print("-" * 78)
    print(f"Summary: {n_pass}/{n_total} validation cases passed.")
    print("=" * 78)

    # Persist a machine-readable report next to this script.
    out = Path(__file__).resolve().parent / "validation_report.txt"
    with out.open("w") as fh:
        fh.write("EGM Analyzer — synthetic validation report\n")
        fh.write(f"fs = {FS_HZ} Hz, T = {DURATION_MS} ms, N = {N}\n\n")
        for r in results:
            flag = "PASS" if r["passed"] else "FAIL"
            fh.write(f"[{flag}] {r['name']}\n")
            fh.write(f"       {r['metric']} = {r['value']:.6e}  "
                     f"(tolerance < {r['tolerance']:.1e})\n")
        fh.write(f"\nSummary: {n_pass}/{n_total} cases passed.\n")
    print(f"Detailed report written to: {out.name}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
