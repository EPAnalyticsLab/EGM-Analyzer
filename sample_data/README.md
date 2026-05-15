# Sample dataset

This folder contains a minimal **anonymised, fully synthetic** dataset
that reproduces the visualisations shown in the manuscript figures. No
patient or animal data is included; all signals are generated
analytically by `generate_sample_data.py`.

## Contents

| File | Description |
|---|---|
| `generate_sample_data.py` | Self-contained generator. Re-run to regenerate every file in this folder. |
| `sample_geometry.xml` | Minimal DIF-style mesh (4 vertices, 2 triangles) bounding the 4×4 catheter footprint. Loadable through *Import geometry*. |
| `sample_signals.csv` | Tidy table with 16 rows (one per electrode) and one column per time sample. Loadable through *Import signals*. |
| `sample_session.pkl` | Pickled session reproducing the GUI's internal state. **Fastest reproduction path**: load with the `💾 Load Session (.pkl)` button. |

## Dataset specification

* Catheter geometry: 4 × 4 grid (labels `A1`…`D4`), 3 mm spacing.
* Sampling rate: 2 kHz, recording length 1 s (2 000 samples).
* Synthetic activations: planar wavefront propagating along `(1, 0.5)` at 0.8 mm/ms.
* Each electrode carries a negative-Gaussian unipolar waveform (`σ = 4 ms`, amplitude 1.5 mV) whose timing follows the projection of its position onto the propagation direction.

## How to reproduce the manuscript figures

1. Launch the application (`python main.py`) and open the browser at
   `http://localhost:8050`.
2. Click `💾 Load Session (.pkl)` and select `sample_session.pkl`.
3. Pick freeze group `374` from the dropdown.
4. Click `LAT`, `Vpp Uni`, `Vpp Bip`, `Vpp Omni`, and `ROR` in turn to
   render the parameter maps shown in **Figure 2**.
5. Click `Select interval`, drag-select around the activation window
   (≈ 380–430 ms) and `Apply` to obtain the bipolar loops and
   omnipolar electrograms shown in **Figure 3**.
6. Use `📤 Export` to download the computed metrics as CSV and the
   currently displayed figures as PNG, all bundled into a single ZIP.

## Regenerating the dataset

```bash
python sample_data/generate_sample_data.py
```

The generator is deterministic (the only stochastic element is the
optional noise term in the validation suite, which is seeded), so the
resulting files are byte-stable across runs on the same Python/NumPy
version.
