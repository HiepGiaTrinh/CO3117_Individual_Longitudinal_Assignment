import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "from_scratch"))
from decision_tree_split import best_threshold, entropy, gini, information_gain  # noqa: E402


def test_entropy_known_values():
    assert entropy([0, 0, 1, 1]) == 1.0
    assert entropy([1, 1, 1]) == 0.0
    assert entropy([]) == 0.0
    assert abs(entropy([1] * 8 + [0] * 2) - 0.7219) < 1e-4


def test_gini_known_values():
    assert gini([0, 0, 1, 1]) == 0.5
    assert gini([1, 1, 1]) == 0.0


def test_worked_example_gain():
    # Parent 8 pos / 2 neg; children (5 pos, 0 neg) and (3 pos, 2 neg).
    y = np.array([1] * 8 + [0] * 2)
    left = np.array([1] * 5)
    right = np.array([1] * 3 + [0] * 2)
    assert abs(information_gain(y, left, right) - 0.2364) < 1e-4


def test_best_threshold_perfect_split():
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    t, g = best_threshold(x, y)
    assert t == 6.5
    assert abs(g - 1.0) < 1e-12


def test_best_threshold_constant_feature():
    assert best_threshold([5, 5, 5], [0, 1, 0]) == (None, 0.0)
