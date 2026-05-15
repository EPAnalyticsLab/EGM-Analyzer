# EGM Analyzer

A modular software platform for the visualisation and analysis of cardiac electrograms.

[![tests](https://github.com/EPAnalyticsLab/EGM-Analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/EPAnalyticsLab/EGM-Analyzer/actions/workflows/tests.yml)
[![Python ≥ 3.9](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **3D geometry visualisation**: interactive rendering of cardiac chamber geometry with colour-mapped parameter overlays.
- **Signal processing**: unipolar, bipolar, and omnipolar analysis with band-pass and notch filtering.
- **Voltage and timing maps**: LAT (local activation time), Vpp (unipolar / bipolar / omnipolar), and ROR (residue-to-omnipole ratio).
- **Omnipolar electrograms**: triangular (L-shape) and cross clique configurations.
- **Temporal interval selection**: focus analysis on a user-defined activation window.
- **Data export**: parameter maps as **CSV** and current figures as **PNG**, bundled into a single ZIP archive.
- **Accessibility**: perceptually uniform, colourblind-safe colormap (`viridis`) for all parameter maps by default.
- **Cross-platform**: fully browser-based Dash interface; identical behaviour on Windows, macOS, and Linux. No OS-native windowing toolkit is required.

## Installation

### Requirements

- Python ≥ 3.9
- Dependencies listed in `requirements.txt`
- Tested on Windows 10 / 11, macOS 12 (Monterey), and Ubuntu 22.04 LTS

### Setup

```bash
git clone https://github.com/EPAnalyticsLab/EGM-Analyzer.git
cd EGM-Analyzer
pip install -r requirements.txt
```

For development (tests and validation suite):

```bash
pip install -r requirements-dev.txt
```

## Usage

### 1. Start the application

```bash
python main.py
```

Then open `http://localhost:8050` in your browser.

### 2. Load data

**Option A — Sample dataset (recommended for first use):**
1. Click `💾 Load Session (.pkl)` and pick `sample_data/sample_session.pkl`.
2. A `4 × 4` synthetic freeze group named `374` is now available in the dropdown.

**Option B — EnSite Precision:**
1. Select `EnSite Precision` from the system dropdown.
2. Click `Load Geometry` → upload your `.html` or `.xml` geometry file.
3. Click `Load Signals` → upload your `DxL_*.csv` signal files.

**Option C — EnSite X:**
1. Select `EnSite X` from the system dropdown.
2. Enter the full file paths for `Wave_rov.csv` and `Map_LAT_uni.csv`.
3. Click `Load`.

> Optional: enable `Estimate missing electrodes` to interpolate signals for incomplete grids.

### 3. Analyse signals

- Select a freeze group from the dropdown.
- Click any map button to render a parameter map on the 3D mesh:
  - `LAT` — Local activation time.
  - `Vpp Uni` — Unipolar peak-to-peak voltage.
  - `Vpp Bip` — Bipolar voltage (horizontal/vertical).
  - `Vpp Omni` — Omnipolar voltage (triangular or cross clique).
  - `ROR` — Residue-to-Omnipole ratio.
- Adjust the colour-bar range (manual or auto) and toggle spatial smoothing (σ).

> **Note on LAT.** The minimum-derivative LAT estimator may be unreliable for fractionated or multi-component electrograms. In such cases, use the temporal interval selection (below) to restrict the computation to a user-defined activation window.

### 4. Signal visualisation panels

- **Bottom-left**: unipolar signals with optional filtering (band-pass 2–100 Hz, notch 50 Hz).
- **Top-right**: bipolar signals (horizontal/vertical) and custom bipolar pairs.
- **Bottom-right**: omnipolar signals, residue, and bipolar loops with propagation vectors.

### 5. Temporal interval selection

- Click `Select interval`.
- Drag-select on the preview graph, or enter `t₀` and `t₁` manually (in ms).
- Click `Apply` to update every omnipolar plot.

### 6. Export

Click `📤 Export` and tick the items you want:

| Option | Format | Content |
|---|---|---|
| `Vpp Unipolar / Bipolar / Omnipolar + ROR` | CSV | One row per electrode/clique with metric value |
| `Signals Unipolar / Bipolar / Omnipolar` | CSV | One row per electrode/clique with full time-series |
| `3D mesh screenshot` | PNG | Currently rendered 3D mesh |
| `Signal panels` | PNG | Unipolar, bipolar, omnipolar, and loop panels |

All selected outputs are packaged into a single ZIP archive (`egm_analyzer_export.zip`). PNG export requires the `kaleido` package (installed automatically through `requirements.txt`).

## Repository layout

```
EGM-Analyzer/
├── main.py                       # Dash application
├── signal_processing.py          # Core algorithmic primitives (LAT, Vpp, omnipolar, ROR)
├── import_data/                  # EnSite Precision / EnSite X importers
├── utils/filters.py              # Band-pass and notch filters
├── tests/                        # pytest unit tests (run on every push/PR)
├── validation/                   # Synthetic validation suite (analytical ground truth)
├── sample_data/                  # Anonymised sample dataset reproducing manuscript figures
├── .github/workflows/tests.yml   # GitHub Actions CI (Linux + macOS + Windows × Python 3.9–3.12)
├── requirements.txt              # Runtime dependencies
└── requirements-dev.txt          # Test / CI dependencies
```

## Testing and continuous integration

Unit tests live under `tests/` and cover the core signal-processing
functions (LAT, Vpp for unipolar / bipolar / omnipolar, omnipolar
orientation-independence, and ROR). The synthetic validation suite
under `validation/` confronts every metric with a closed-form ground
truth (see `validation/README.md` for tolerances).

```bash
# Run the unit tests
pytest tests/

# Run the synthetic validation suite
python validation/run_validation.py
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) executes both
on every push and pull request across `ubuntu-latest`, `macos-latest`,
and `windows-latest` for Python 3.9 – 3.12.

## File formats

### Input

- **Geometry**: `.html` (Plotly mesh) or `.xml` (DIF format).
- **Signals (EnSite Precision)**: `DxL_*.csv` files exported from the system.
- **Signals (EnSite X)**: `Wave_rov.csv` + `Map_LAT_uni.csv`.
- **Session resumption**: `.pkl` file produced by a previous run.

### Output

- **CSV**: one file per requested parameter map or signal block.
- **PNG**: one file per requested figure panel.
- All bundled into `egm_analyzer_export.zip`.

## Performance

Tested on a standard laptop (Intel Core i7, 16 GB RAM):

| Dataset size | Load + render time | RAM usage |
|---|---|---|
| ≈ 3 500 sites, 1 s @ 2 kHz | 4 – 8 s | ≈ 600 MB |
| > 10 000 sites | Interactive responsiveness begins to degrade | — |

## Citation

```
[to be added upon publication]
```

## License

MIT License. See `LICENSE`.

## Acknowledgements

This platform builds on well-established electrophysiology methods —
peak-to-peak amplitude, omnipolar electrogram computation, and LAT
estimation — that have been extensively validated in the cardiac
electrophysiology literature. EGM Analyzer integrates these methods
into a single interactive Python-based platform.

Several MATLAB-based toolboxes for EGM analysis (Narayan, Vigmond, and
collaborating groups) offer powerful analysis capabilities and inspired
parts of this work. EGM Analyzer, being fully Python-based and
open-source under the MIT License, removes the dependency on a
proprietary MATLAB licence and lowers the barrier to access for the
broader research community.

## Contact

`qepcontact@gmail.com`
