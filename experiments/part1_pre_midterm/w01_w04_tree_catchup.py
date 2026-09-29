"""Release catch-up experiments (W01-W04): Gini vs entropy split choice, and depth vs generalization.

Uses datasets bundled with scikit-learn (no download). Seed fixed for reproducibility.
Run from repo root: python experiments/part1_pre_midterm/w01_w04_tree_catchup.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "from_scratch"))
from decision_tree_split import best_threshold, entropy, gini  # noqa: E402

SEED = 42

# Part 1: does Gini or entropy pick a different root split? (my code vs sklearn) ---
iris = load_iris()
X, y = iris.data, iris.target

print("Part 1: root split on Iris (150 samples, 4 attributes)")

for name, imp in [("entropy", entropy), ("gini", gini)]:
    results = [(iris.feature_names[j],) + best_threshold(X[:, j], y, imp) for j in range(X.shape[1])]
    best = max(results, key=lambda r: r[2])
    print(f"  own {name:7s}:", ", ".join(f"{n}: t={t:.2f} gain={g:.4f}" for n, t, g in results))
    print(f"  own {name:7s} -> root = {best[0]} <= {best[1]:.2f}")
    sk = DecisionTreeClassifier(criterion=name, max_depth=1, random_state=SEED).fit(X, y)
    print(f"  sklearn {name:7s} -> root = {iris.feature_names[sk.tree_.feature[0]]} <= {sk.tree_.threshold[0]:.2f}")

# Part 2: tree depth vs train/test accuracy (overfitting) ---
bc = load_breast_cancer()

Xtr, Xte, ytr, yte = train_test_split(bc.data, bc.target, test_size=0.3, stratify=bc.target, random_state=SEED)
print("\nPart 2: max_depth vs accuracy on Breast Cancer (70/30 stratified split, entropy)")
print("  depth  train_acc  test_acc  n_leaves")
for depth in [1, 2, 3, 4, 5, 8, None]:
    t = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=SEED).fit(Xtr, ytr)
    print(f"  {str(depth):>5}  {t.score(Xtr, ytr):9.4f}  {t.score(Xte, yte):8.4f}  {t.get_n_leaves():8d}")
