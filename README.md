# EGM Analyzer
A Modular Software Platform for Visualization and Analysis of Cardiac Electrograms

## Features
- **3D Geometry Visualization**: Interactive rendering of cardiac chamber geometry with color-mapped metrics
- **Signal Processing**: Unipolar, bipolar, and omnipolar signal analysis with customizable filtering
- **Voltage Mapping**: Local activation time (LAT), voltage peak-to-peak (Vpp), and rate-of-rise (ROR) computation
- **Omnipolar Technology**: Triangular and cross configurations for direction-independent voltage measurement
- **Temporal Interval Selection**: Focused analysis on specific signal segments
- **Data Export**: CSV export of all computed metrics; PNG export of signal visualizations and 3D mesh screenshots
- **Missing Electrode Estimation**: IDW-based interpolation for incomplete electrode arrays
- **Accessibility**: Colorblind-safe colormaps (viridis) and keyboard shortcuts for common operations (zoom, pan, rotate)

## Installation

### Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`
- Tested on: Windows 10/11, macOS 12 (Monterey), and Ubuntu 22.04 LTS

> **Note (macOS):** Minor differences in Tkinter rendering may occur (e.g., menu bar placement).

### Setup
```bash
# Clone the repository
git clone https://github.com/EPAnalyticsLab/EGM-Analyzer.git
cd EGM-Analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Start the application
```bash
python main.py
```
Then open your browser at: `http://localhost:8050`

### 2. Load data

**For Ensite Precision:**
1. Select `Ensite Precision` from the System dropdown
2. Click `Load Geometry` → upload your `.html` or `.xml` geometry file
3. Click `Load Signals` → upload your `DxL_*.csv` signal files

**For Ensite X:**
1. Select `Ensite X` from the System dropdown
2. Enter the full file paths:
   - `Wave_rov.csv` (roving signals)
   - `Map_LAT_uni.csv` (coordinates and LAT data)
3. Click `Load`

Optional: Enable `Estimate missing electrodes` to interpolate signals for incomplete grids.

**Sample dataset:** A minimal anonymized dataset (geometry in XML format and electrogram recordings in CSV format) is available under `/sample_data/`. It is sufficient to reproduce all visualizations shown in the manuscript figures and serves as a tutorial starting point.

### 3. Analyze signals
- Select a `freeze group` from the dropdown
- Click map buttons to visualize:
  - `LAT`: Local activation time
  - `Vpp Uni`: Unipolar voltage peak-to-peak
  - `Vpp Bip`: Bipolar voltage (horizontal/vertical)
  - `Vpp Omni`: Omnipolar voltage (triangular or cross)
  - `ROR`: Rate-of-rise (residue/omnipolar ratio)
- Adjust `colorbar range` (manual or auto)
- Toggle `spatial interpolation` and adjust σ for smoothing

> **Note on LAT estimation:** The minimum-derivative LAT estimator may be unreliable for fractionated or multi-component electrograms. In such cases, use the temporal interval selection (see below) to manually restrict computation to a user-specified activation window.

### 4. Signal visualization panels
- `Bottom-left`: Unipolar signals with filtering (bandpass 2–100 Hz, notch 50 Hz)
- `Top-right`: Bipolar signals (horizontal/vertical) + custom bipolar pairs
- `Bottom-right`: Omnipolar signals, residue, and bipolar loops with propagation vectors

### 5. Temporal interval selection
- Click `Select interval` to focus omnipolar analysis on a specific time window
- Drag-select on the preview graph or enter `t₀` and `t₁` manually
- Click `Apply` to update all omnipolar plots

### 6. Export data
- Click `Export`
- Select metrics to export:
  - Vpp Unipolar / Bipolar / Omnipolar
  - Signals Unipolar / Bipolar / Omnipolar
- Click `Export selected` → downloads metrics as **CSV** files; signal visualizations and 3D screenshots as **PNG** files

## File Formats

### Input
- `Geometry`: `.html` (Plotly mesh) or `.xml` (DIF format)
- `Signals (Precision)`: `DxL_*.csv` files exported from Ensite Precision
- `Signals (Ensite X)`: `Wave_rov.csv` + `Map_LAT_uni.csv`

### Output
- `CSV`: Computed electrophysiological parameter maps (LAT, Vpp, ROR, etc.)
- `PNG`: Signal visualizations and 3D mesh screenshots

## Performance

Tested on a standard laptop (Intel Core i7, 16 GB RAM):

| Dataset size | Load + render time | RAM usage |
|---|---|---|
| ~3,500 sites, 1 s @ 2 kHz | 4–8 seconds | ~600 MB |
| >10,000 sites | Interactive responsiveness begins to degrade | — |

## Testing & CI

Unit tests covering the core signal processing functions are located under `/tests/`. A GitHub Actions CI workflow (`.github/workflows/ci.yml`) runs these tests automatically on every push and pull request.

### What is tested

| Module | Functions covered |
|---|---|
| LAT estimation | `compute_lat` — dip detection, NaN handling, edge cases |
| Voltage (unipolar/bipolar) | `compute_vpp` — amplitude, NaN masking, signed signals |
| Omnipolar | `compute_omnipolar` — triangular and cross configurations, rotation, energy preservation |
| Rate-of-Rise | `compute_ror` — ratio computation, zero-residue handling |
| Bandpass filter | `bandpass_filter` — passband preservation (50 Hz), stopband rejection (500 Hz), DC removal |
| Notch filter | `notch_filter` — mains rejection (50 Hz), passband preservation (30 Hz, 80 Hz) |

### Run tests locally

Install the test dependencies (lighter than the full app stack) and run pytest from the repository root:

```bash
pip install pytest numpy scipy
pytest tests/ -v
```

### CI pipeline

The workflow is defined in `.github/workflows/ci.yml` and triggers on every push and pull request to any branch. It runs on `ubuntu-latest` with Python 3.11 and installs only `pytest`, `numpy`, and `scipy` — no Dash server is required.

## Citation
If you use this software in your research, please cite:
```
[to be added upon publication]
```

## License
MIT License

## Contact
For questions or support: [qepcontact@gmail.com]

## Acknowledgments
Developed for high-density cardiac electrophysiology mapping analysis.

This platform builds on well-established electrophysiology methods (peak-to-peak amplitude, omnipolar computation, LAT estimation) that have been extensively validated in the cardiac electrophysiology literature. EGM Analyzer integrates these methods into a single interactive Python-based platform.

Several MATLAB-based toolboxes for EGM analysis exist in the literature (Narayan, Vigmond, and collaborating groups) and offer powerful analysis capabilities. EGM Analyzer, being fully Python-based and open-source, does not require a proprietary license and is designed to lower the barrier to access for the broader research community.

## Repository Metadata
- `Version`: v1.0
- `Repository`: https://github.com/EPAnalyticsLab/EGM-Analyzer
- `License`: MIT License
- `Language`: Python
- `Versioning`: Git (GitHub)
- `Dependencies`: See `requirements.txt`
- `Documentation`: This README
- `Sample Data`: `/sample_data/`
- `Tests`: `/tests/`
- `CI`: GitHub Actions (`.github/workflows/ci.yml`)
