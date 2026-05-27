"""Unit tests for the electrode-label parser."""

import pytest

from signal_processing import label_to_grid


@pytest.mark.parametrize("label,expected", [
    ("A1", (0, 0)),
    ("A4", (0, 3)),
    ("B2", (1, 1)),
    ("C3", (2, 2)),
    ("D4", (3, 3)),
])
def test_label_to_grid_canonical(label, expected):
    assert label_to_grid(label) == expected


def test_label_to_grid_accepts_decorated_strings():
    # Real catheter labels often include catheter prefixes; the parser
    # should still extract the (letter, digit) token.
    assert label_to_grid("Cath-A1") == (0, 0)
    assert label_to_grid("Spline_B3_ROV") == (1, 2)


def test_label_to_grid_unrecognised_returns_none():
    assert label_to_grid("Z9") == (None, None)
    assert label_to_grid("") == (None, None)
    assert label_to_grid("---") == (None, None)
