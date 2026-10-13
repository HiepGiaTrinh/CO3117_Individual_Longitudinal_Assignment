"""W07: GA feature selection on UCI HAR, three independent seeds.

Train split only; validation is GroupKFold by subject inside train (see ga.py).
The test split is never read. Outputs a per-seed table, results/w07_ga_seeds.csv
and results/figures/w07_ga_convergence.png.
"""
import csv
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.from_scratch.ga import fitness, load_har_train, make_pool, run_ga  # noqa: E402

SEEDS = [42, 43, 44]
POOL_SIZE, POOL_SEED, LAM = 40, 0, 0.05

X, y, groups = load_har_train(os.path.join(ROOT, "data", "UCI HAR Dataset"))
pool = make_pool(X.shape[1], POOL_SIZE, POOL_SEED)

# Reference: all 40 pool features selected.
all_fit = fitness(np.ones(POOL_SIZE, dtype=bool), X, y, groups, pool, LAM)
print(f"reference, all {POOL_SIZE} pool features: fitness = {all_fit:.4f}")

rows, runs = [], {}
for s in SEEDS:
    t0 = time.time()
    r = run_ga(X, y, groups, seed=s, pool_size=POOL_SIZE, pool_seed=POOL_SEED, lam=LAM)
    h = r["history"]
    k = int(r["best_chrom"].sum())
    runs[s] = r
    rows.append({
        "seed": s,
        "best_fitness": round(r["best_fitness"], 4),
        "n_features": k,
        "macro_f1_cv": round(r["best_fitness"] + LAM * k / POOL_SIZE, 4),
        "generation_of_best": int(np.argmax(h["best_fitness"])) + 1,
        "diversity_first": round(h["diversity"][0], 2),
        "diversity_last": round(h["diversity"][-1], 2),
        "seconds": round(time.time() - t0),
    })
    print(rows[-1], flush=True)

best = np.array([r["best_fitness"] for r in rows])
print(f"\nbest fitness over seeds: mean = {best.mean():.4f}, std = {best.std(ddof=1):.4f}, "
      f"min = {best.min():.4f}, max = {best.max():.4f}")

sets = {s: set(runs[s]["best_features"].tolist()) for s in SEEDS}
for i, a in enumerate(SEEDS):
    for b in SEEDS[i + 1:]:
        inter, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
        print(f"selected columns, seed {a} vs {b}: {inter} in common, Jaccard = {inter / union:.2f}")

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
with open(os.path.join(ROOT, "results", "w07_ga_seeds.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for s in SEEDS:
    ax[0].plot(runs[s]["history"]["best_fitness"], label=f"seed {s}")
    ax[1].plot(runs[s]["history"]["diversity"], label=f"seed {s}")
ax[0].axhline(all_fit, color="gray", ls="--", label="all 40 features")
ax[0].set(xlabel="generation", ylabel="best fitness in generation")
ax[1].set(xlabel="generation", ylabel="mean pairwise Hamming distance")
ax[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "figures", "w07_ga_convergence.png"), dpi=150)
print("saved results/w07_ga_seeds.csv and results/figures/w07_ga_convergence.png")
