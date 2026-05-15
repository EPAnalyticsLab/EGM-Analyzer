# Synthetic validation

This folder contains the synthetic validation suite that confronts the core
signal-processing routines of EGM Analyzer (LAT, Vpp, omnipolar, ROR) with
electrograms whose properties are known analytically.

## Running

```bash
python validation/run_validation.py
```

The script prints a per-case PASS/FAIL summary to stdout and writes a
detailed report to `validation/validation_report.txt`. Exit status is 0
when every case passes, 1 otherwise. The suite is part of the GitHub
Actions CI workflow (`.github/workflows/tests.yml`) and runs on every
push and pull request.

## Cases

| Metric | Synthetic input | Ground truth | Tolerance |
|---|---|---|---|
| `Vpp(uni)` | Negative-Gaussian unipolar of prescribed amplitude `A` | `Vpp = A` | < 0.01 mV |
| `Vpp(bip)` | Subtraction of two time-shifted unipolars | `max(b) − min(b)` (exact) | < 1e-9 mV |
| `LAT(uni)` | Negative-Gaussian unipolar centred at `t_peak` | Analytical inflexion-point location | < 0.5 ms (one sample period at 2 kHz) |
| `Vpp(omni)` | 2-D bipolar trajectory rotated by θ ∈ [0°, 360°) | Vpp invariant under rigid rotation | < 0.01 mV |
| `ROR` (linear) | `b(t) = (sin t, 0)` | `ROR = 0` | < 1e-6 |
| `ROR` (circular) | `b(t) = (cos t, sin t)` | `ROR = 1` | < 5e-3 |
