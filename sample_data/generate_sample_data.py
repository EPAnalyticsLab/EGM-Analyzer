"""
Generate a minimal anonymised sample dataset for EGM Analyzer.

This script writes a synthetic but morphologically realistic dataset
mimicking a single freeze group acquired with a 4 x 4 multi-electrode
catheter on a small atrial-like surface. The dataset is provided in two
complementary formats:

* ``sample_data/sample_geometry.xml``   — a DIF-style XML mesh (vertices
  + triangles) accepted by the ``Import geometry`` route.
* ``sample_data/sample_signals.csv``    — a tidy CSV holding the 16
  unipolar electrograms together with their 3-D coordinates.
* ``sample_data/sample_session.pkl``    — a pickled session reproducing
  the GUI's internal state, directly loadable through the
  ``Load Session (.pkl)`` button. This is the fastest reproduction
  path and is the one referenced from the manuscript figures.

All signals are synthetic. No patient data is included.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE

FS_HZ = 2000.0
DURATION_MS = 1000.0
N = int(FS_HZ * DURATION_MS / 1000.0)
GRID_ROWS = 4
GRID_COLS = 4
ELECTRODE_SPACING_MM = 3.0


def unipolar(t_peak_ms: float, amplitude_mV: float = 1.0,
             sharpness_ms: float = 4.0) -> np.ndarray:
    """Synthetic negative-Gaussian unipolar electrogram."""
    t = np.arange(N) / FS_HZ
    z = (t - t_peak_ms / 1000.0) / (sharpness_ms / 1000.0)
    return amplitude_mV * (-np.exp(-(z ** 2)))


def make_grid_coordinates():
    """Build coordinates for the 4x4 catheter centred on a small mesh."""
    xs, ys, zs, labels = [], [], [], []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            xs.append(c * ELECTRODE_SPACING_MM)
            ys.append(r * ELECTRODE_SPACING_MM)
            zs.append(0.0)
            labels.append(f"{chr(ord('A') + r)}{c + 1}")
    return (np.array(xs, dtype=float),
            np.array(ys, dtype=float),
            np.array(zs, dtype=float),
            np.array(labels))


def synthesise_signals(xs, ys, propagation_dir=np.array([1.0, 0.5]),
                       conduction_velocity_mm_per_ms: float = 0.8):
    """Generate 16 unipolar electrograms propagating along ``propagation_dir``.

    Activation arrives sequentially at each electrode according to its
    projection onto the wavefront propagation direction.
    """
    direction = propagation_dir / np.linalg.norm(propagation_dir)
    projections = xs * direction[0] + ys * direction[1]
    # Centre the activation around 400 ms
    t0_arrival = (projections - projections.min()) / conduction_velocity_mm_per_ms
    t_peaks = 400.0 + t0_arrival
    signals = np.stack([unipolar(t, 1.5) for t in t_peaks], axis=0)
    return signals, t_peaks


def write_geometry_xml(path: Path, xs, ys, zs):
    """Write a minimal DIF-style XML geometry containing two triangles
    bounding the electrode array, sufficient for the GUI to render."""
    mesh_pad = 6.0
    x_min, x_max = xs.min() - mesh_pad, xs.max() + mesh_pad
    y_min, y_max = ys.min() - mesh_pad, ys.max() + mesh_pad

    vertices = [
        (x_min, y_min, 0.0),
        (x_max, y_min, 0.0),
        (x_max, y_max, 0.0),
        (x_min, y_max, 0.0),
    ]
    triangles = [(0, 1, 2), (0, 2, 3)]

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<DIF version="1.0">',
             '  <Map name="sample_anonymised">',
             '    <Volume>',
             '      <Vertices>']
    for vx, vy, vz in vertices:
        lines.append(f'        <V x="{vx:.4f}" y="{vy:.4f}" z="{vz:.4f}"/>')
    lines += ['      </Vertices>', '      <Polygons>']
    for a, b, c in triangles:
        lines.append(f'        <P v1="{a}" v2="{b}" v3="{c}"/>')
    lines += ['      </Polygons>', '    </Volume>', '  </Map>', '</DIF>']
    path.write_text("\n".join(lines), encoding="utf-8")


def write_signals_csv(path: Path, xs, ys, zs, labels, signals):
    """Write the signals as a tidy CSV (one row per electrode)."""
    cols = {"label": labels, "x": xs, "y": ys, "z": zs}
    for j in range(signals.shape[1]):
        cols[f"t{j}"] = signals[:, j]
    pd.DataFrame(cols).to_csv(path, index=False)


def write_session_pkl(path: Path, xs, ys, zs, labels, signals):
    """Persist a pickled session matching the GUI's internal layout."""
    n = len(labels)
    idx = pd.RangeIndex(n)
    data_table = pd.DataFrame({
        "pt number":   ["374"] * n,
        "roving x":    xs,
        "roving y":    ys,
        "roving z":    zs,
        "Sample rate": [FS_HZ] * n,
        "rov LAT":     [0.0] * n,
        "peak2peak":   [0.0] * n,
    }, index=idx)

    sample_cols = [f"s{i}" for i in range(signals.shape[1])]
    rov_df = pd.DataFrame(signals, columns=sample_cols, index=idx)
    rov_df.insert(0, "label", labels)
    rov_df.insert(1, "x", xs)
    rov_df.insert(2, "y", ys)

    geometry = {
        "vertices": {"x": np.array([-6, 15, 15, -6], dtype=float),
                     "y": np.array([-6, -6, 15, 15], dtype=float),
                     "z": np.array([0.0, 0.0, 0.0, 0.0])},
        "faces":    {"v1": np.array([0, 0]),
                     "v2": np.array([1, 2]),
                     "v3": np.array([2, 3])},
    }

    signals_dict = {
        "data_table": data_table,
        "signals":    {"rov trace": rov_df},
    }
    payload = {"signals": signals_dict, "geometry": geometry}

    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def main() -> int:
    xs, ys, zs, labels = make_grid_coordinates()
    signals, t_peaks = synthesise_signals(xs, ys)

    write_geometry_xml(OUT / "sample_geometry.xml", xs, ys, zs)
    write_signals_csv(OUT / "sample_signals.csv", xs, ys, zs, labels, signals)
    write_session_pkl(OUT / "sample_session.pkl", xs, ys, zs, labels, signals)

    print("Sample dataset generated.")
    print(f"  geometry  : {(OUT / 'sample_geometry.xml').name}")
    print(f"  signals   : {(OUT / 'sample_signals.csv').name}")
    print(f"  session   : {(OUT / 'sample_session.pkl').name}")
    print(f"  grid      : {GRID_ROWS}x{GRID_COLS}, spacing {ELECTRODE_SPACING_MM} mm")
    print(f"  fs        : {FS_HZ:.0f} Hz, duration {DURATION_MS:.0f} ms, N = {N}")
    print(f"  activations span: {t_peaks.min():.1f}–{t_peaks.max():.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
