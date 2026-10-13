"""Genetic algorithm for feature selection on UCI HAR (train split only)."""
import os

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB


def load_har_train(data_dir):
    """Load X, y, subject from <data_dir>/train. The test split is never read."""
    d = os.path.join(data_dir, "train")
    X = np.loadtxt(os.path.join(d, "X_train.txt"))
    y = np.loadtxt(os.path.join(d, "y_train.txt"), dtype=int)
    subj = np.loadtxt(os.path.join(d, "subject_train.txt"), dtype=int)
    return X, y, subj


# 1. Representation: binary chromosome of length pool_size.
# Pool = pool_size columns drawn at random (no labels used) with pool_seed,
# independent of the GA seed, so every GA run uses the same pool.
def make_pool(n_columns, pool_size=40, pool_seed=0):
    rng = np.random.default_rng(pool_seed)
    return np.sort(rng.choice(n_columns, size=pool_size, replace=False))


# 2. Decode: chromosome -> indices of selected columns of X.
def decode(chrom, pool):
    return pool[np.asarray(chrom, dtype=bool)]


# 3. Fitness: mean Macro-F1 over GroupKFold(subject) minus lam * k / pool_size.
# All-zero chromosome gets -1.0, below any valid fitness (which is > -lam).
def fitness(chrom, X, y, groups, pool, lam=0.05):
    k = int(np.sum(chrom))
    if k == 0:
        return -1.0
    f1 = cross_val_score(GaussianNB(), X[:, decode(chrom, pool)], y, groups=groups,
                         cv=GroupKFold(n_splits=5), scoring="f1_macro").mean()
    return float(f1) - lam * k / len(pool)


# 4. Selection: tournament.
def tournament_select(fit, rng, n, k=3):
    cand = rng.integers(0, len(fit), size=(n, k))
    return cand[np.arange(n), np.argmax(fit[cand], axis=1)]


# 5. Crossover (uniform) and mutation (bit-flip per bit).
def uniform_crossover(p1, p2, rng, p_c=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < p_c:
        mask = rng.random(p1.shape) < 0.5
        c1[mask], c2[mask] = p2[mask], p1[mask]
    return c1, c2


def bitflip_mutation(chrom, rng, p_m):
    return chrom ^ (rng.random(chrom.shape) < p_m)


# 6. GA loop. Stopping rule: fixed number of generations.
def hamming_diversity(pop):
    """Mean pairwise Hamming distance of the population."""
    n = len(pop)
    c = pop.sum(0)
    return float((c * (n - c)).sum() / (n * (n - 1) / 2))


def run_ga(X, y, groups, seed, pop_size=30, n_generations=40, pool_size=40,
           pool_seed=0, lam=0.05, p_c=0.9, p_m=None, tournament_k=3):
    rng = np.random.default_rng(seed)
    pool = make_pool(X.shape[1], pool_size, pool_seed)
    p_m = 1.0 / pool_size if p_m is None else p_m
    pop = rng.random((pop_size, pool_size)) < 0.5
    best_fit, best_chrom = -np.inf, None
    history = {"best_fitness": [], "diversity": []}

    for _ in range(n_generations):
        fit = np.array([fitness(c, X, y, groups, pool, lam) for c in pop])
        b = int(np.argmax(fit))
        history["best_fitness"].append(float(fit[b]))
        history["diversity"].append(hamming_diversity(pop))
        if fit[b] > best_fit:
            best_fit, best_chrom = float(fit[b]), pop[b].copy()

        children = []
        while len(children) < pop_size:
            i, j = tournament_select(fit, rng, 2, tournament_k)
            c1, c2 = uniform_crossover(pop[i], pop[j], rng, p_c)
            children += [bitflip_mutation(c1, rng, p_m), bitflip_mutation(c2, rng, p_m)]
        pop = np.array(children[:pop_size])

    return {"best_chrom": best_chrom, "best_features": decode(best_chrom, pool),
            "best_fitness": best_fit, "history": history}
