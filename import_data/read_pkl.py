"""
read_pkl.py  –  cross-platform session loader
Uses a simple path input instead of tkinter (which requires a display on Linux/macOS servers).
"""
import pickle


def read_pkl(path: str):
    """Load a session pickle file from a given filesystem path."""
    if not path:
        raise ValueError("No path provided")
    path = path.strip().strip('"').strip("'")
    print(f"Importing session from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Import complete")
    return data


def read_pkl_bytes(content_bytes: bytes):
    """Load a session pickle from raw bytes (e.g. from a Dash dcc.Upload)."""
    import io
    return pickle.load(io.BytesIO(content_bytes))
