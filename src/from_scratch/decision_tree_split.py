"""Impurity and split routines for decision trees (CO3117 Ch.2, release catch-up).

Only the split-selection core is implemented here, not a full tree.
"""
import numpy as np


def entropy(y):
    """H(S) = -sum_i p_i * log2(p_i). Returns 0.0 for an empty set."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts = True)
    
    p = counts / y.size
    return float(-np.sum(p * np.log2(p)))


def gini(y):
    """Gini = 1 - sum_i p_i^2. Returns 0.0 for an empty set."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts = True)
    
    p = counts / y.size
    
    return float(1.0 - np.sum(p ** 2))


def information_gain(y, y_left, y_right, impurity = entropy):
    """Impurity(parent) - weighted average impurity of the two children."""
    n = len(y)
    weighted = (len(y_left) / n) * impurity(y_left) + (len(y_right) / n) * impurity(y_right)
    
    return impurity(y) - weighted


def best_threshold(x, y, impurity = entropy):
    """Best binary split `x <= t` for one continuous attribute.

    Candidate thresholds are midpoints between consecutive distinct sorted values.
    Returns (threshold, gain); (None, 0.0) if x has a single distinct value.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.unique(x)
    
    if values.size < 2:
        return None, 0.0
    
    best_t, best_gain = None, -np.inf
    
    for t in (values[:-1] + values[1:]) / 2.0:
        mask = x <= t
        gain = information_gain(y, y[mask], y[~mask], impurity)
        if gain > best_gain:
            best_t, best_gain = float(t), gain
            
    return best_t, float(best_gain)
