# EGM Analyzer

A Modular Software Platform for Visualization and Analysis of Cardiac Electrograms

## Features

- **3D Geometry Visualization**: Interactive rendering of cardiac chamber geometry with color-mapped metrics
- **Signal Processing**: Unipolar, bipolar, and omnipolar signal analysis with customizable filtering
- **Voltage Mapping**: Local activation time (LAT), voltage peak-to-peak (Vpp), and rate-of-rise (ROR) computation
- **Omnipolar Technology**: Triangular and cross configurations for direction-independent voltage measurement
- **Temporal Interval Selection**: Focused analysis on specific signal segments
- **Data Export**: Excel export of all computed metrics and signals
- **Missing Electrode Estimation**: IDW-based interpolation for incomplete electrode arrays

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/seciuk/PS2-cardiac-electrophysiology.git
cd PS2-cardiac-electrophysiology

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Start the application
```bash
python main.py
```

Then open your browser at: 'http://localhost:8050'

### 2. Load data

For Ensite Precision:
1. Select 'Ensite Precision' from the System dropdown
2. Click 'Load Geometry' → upload your `.html` or `.xml` geometry file
3. Click 'Load Signals' → upload your `DxL_*.csv` signal files

For Ensite X:
1. Select 'Ensite X' from the System dropdown
2. Enter the full file paths:
   - `Wave_rov.csv` (roving signals)
   - `Map_LAT_uni.csv` (coordinates and LAT data)
3. Click 'Load'

Optional: Enable 'Estimate missing electrodes' to interpolate signals for incomplete grids.

### 3. Analyze signals

- Select a 'freeze group' from the dropdown
- Click map buttons to visualize:
  - 'LAT': Local activation time
  - 'Vpp Uni': Unipolar voltage peak-to-peak
  - 'Vpp Bip': Bipolar voltage (horizontal/vertical)
  - 'Vpp Omni': Omnipolar voltage (triangular or cross)
  - 'ROR': Rate-of-rise (residue/omnipolar ratio)
- Adjust 'colorbar range' (manual or auto)
- Toggle 'spatial interpolation' and adjust σ for smoothing

### 4. Signal visualization panels

- 'Bottom-left': Unipolar signals with filtering (bandpass 2-100 Hz, notch 50 Hz)
- 'Top-right': Bipolar signals (horizontal/vertical) + custom bipolar pairs
- 'Bottom-right': Omnipolar signals, residue, and bipolar loops with propagation vectors

### 5. Temporal interval selection

- Click 'Select interval' to focus omnipolar analysis on a specific time window
- Drag-select on the preview graph or enter `t₀` and `t₁` manually
- Click 'Apply' to update all omnipolar plots

### 6. Export data

- Click 'Export'
- Select metrics to export:
  - Vpp Unipolar / Bipolar / Omnipolar
  - Signals Unipolar / Bipolar / Omnipolar
- Click 'Export selected' → downloads `ep_export.xlsx`

## File Formats

### Input
- 'Geometry': `.html` (Plotly mesh) or `.xml` (DIF format)
- 'Signals (Precision)': `DxL_*.csv` files exported from Ensite Precision
- 'Signals (Ensite X)': `Wave_rov.csv` + `Map_LAT_uni.csv`

### Output
- 'Excel': Multi-sheet workbook with all selected metrics and signals

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

## Repository Metadata

- 'Version': v1
- 'Repository': https://github.com/EPAnalyticsLab/EGM-Analyzer
- 'License': MIT License
- 'Language': Python
- 'Dependencies': See `requirements.txt`
- 'Documentation': This README
