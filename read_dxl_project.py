import re

import numpy as np

COORDS_RE = re.compile(r"([A-D])([1-4])")


def extract_coords(x):
    if x is None:
        return np.nan, np.nan

    if not isinstance(x, str):
        return np.nan, np.nan

    coords = COORDS_RE.findall(x)
    if len(coords) != 1:
        return np.nan, np.nan

    return ord(coords[0][0]) - ord("A"), int(coords[0][1]) - 1
