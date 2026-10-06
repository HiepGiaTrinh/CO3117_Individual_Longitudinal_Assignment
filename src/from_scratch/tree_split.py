"""W03 (release catch-up). One impurity/split routine for decision trees.

Interface (Appendix A):
    entropy(y) -> float                 Shannon entropy in bits of the label array y.
    gini(y) -> float                    Gini impurity of y.
    information_gain(y, mask) -> float  Entropy of y minus the weighted entropy of y[mask], y[~mask].
    best_threshold(x, y) -> (t, gain)   Best split x <= t on one continuous feature by information
                                        gain; t is a midpoint between two consecutive distinct values.
                                        Return (None, 0.0) when x has a single distinct value.
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


def information_gain(y, mask, impurity = entropy):
    """Impurity(parent) - weighted average impurity of the two children."""
    y_left, y_right = y[mask], y[~mask] 
    n = len(y)
    weighted = (len(y_left) / n) * impurity(y_left) + (len(y_right) / n) * impurity(y_right)
    
    return impurity(y) - weighted
    


def best_threshold(x, y):
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
        gain = information_gain(y, mask)
        if gain > best_gain:
            best_t, best_gain = float(t), gain
            
    return best_t, float(best_gain)

