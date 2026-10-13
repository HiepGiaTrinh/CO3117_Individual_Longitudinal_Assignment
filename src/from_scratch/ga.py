"""Genetic algorithm for feature selection on UCI HAR (numpy/pandas only).

Only the train split is read. Validation is GroupKFold by subject inside the
train set. The test split is never loaded.
"""
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- data
def load_har_train(data_dir):
    """Load X_train, y_train, subject_train from the 'UCI HAR Dataset' folder."""
    d = os.path.join(data_dir, "train")
    X = pd.read_csv(os.path.join(d, "X_train.txt"), sep=r"\s+", header=None).to_numpy(float)
    y = pd.read_csv(os.path.join(d, "y_train.txt"), header=None).to_numpy().ravel()
    subj = pd.read_csv(os.path.join(d, "subject_train.txt"), header=None).to_numpy().ravel()
    return X, y, subj


# ------------------------------------------------- 1. representation
def make_pool(n_columns, pool_size=40, pool_seed=0):
    """Candidate pool = random subset of the original columns.

    The pool is drawn WITHOUT using labels, so no information from the
    validation folds leaks into it. It is controlled by `pool_seed`, which is
    independent of the GA seed, so all GA runs are compared on the same pool.
    A chromosome is a boolean vector of length pool_size: gene j = 1 means
    pool[j] is selected.
    """
    rng = np.random.default_rng(pool_seed)
    return np.sort(rng.choice(n_columns, size=pool_size, replace=False))


# ------------------------------------------------------- 2. decode
def decode(chrom, pool):
    """Chromosome -> indices of the selected columns of the original X."""
    return pool[np.asarray(chrom, dtype=bool)]


# ------------------------------------------------------- 3. fitness
def group_kfold_splits(groups, n_splits=5):
    """GroupKFold: whole subjects go to one fold; folds balanced by size."""
    ids, counts = np.unique(groups, return_counts=True)
    fold = np.empty(len(groups), dtype=int)
    load = np.zeros(n_splits)
    for k in np.argsort(-counts, kind="stable"):
        f = int(np.argmin(load))
        load[f] += counts[k]
        fold[groups == ids[k]] = f
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(n_splits)]


def gnb_predict(Xtr, ytr, Xva, eps=1e-6):
    """Gaussian Naive Bayes (fast, closed form)."""
    cls = np.unique(ytr)
    mu = np.stack([Xtr[ytr == c].mean(0) for c in cls])
    var = np.stack([Xtr[ytr == c].var(0) for c in cls]) + eps
    log_prior = np.log(np.array([(ytr == c).mean() for c in cls]))
    ll = -0.5 * (np.log(2 * np.pi * var).sum(1)
                 + ((Xva[:, None, :] - mu) ** 2 / var).sum(2))
    return cls[np.argmax(ll + log_prior, axis=1)]


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        den = 2 * tp + fp + fn
        f1s.append(2 * tp / den if den else 0.0)
    return float(np.mean(f1s))


def fitness(chrom, X, y, pool, folds, classes, lam=0.05):
    """Mean Macro-F1 over GroupKFold folds minus lam * k / pool_size.

    All-zero chromosome (k = 0): return -1.0. Valid fitness is >= -lam > -1
    (for lam < 1), so an empty chromosome always ranks last.
    """
    k = int(np.sum(chrom))
    if k == 0:
        return -1.0
    Xs = X[:, decode(chrom, pool)]
    scores = [macro_f1(y[va], gnb_predict(Xs[tr], y[tr], Xs[va]), classes)
              for tr, va in folds]
    return float(np.mean(scores)) - lam * k / len(pool)


# --------------------------------------------------- 4. selection
def tournament_select(fit, rng, n, k=3):
    """Return n parent indices; each is the best of k random contestants."""
    cand = rng.integers(0, len(fit), size=(n, k))
    return cand[np.arange(n), np.argmax(fit[cand], axis=1)]


# ------------------------------------- 5. crossover and mutation
def uniform_crossover(p1, p2, rng, p_c=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < p_c:
        mask = rng.random(p1.shape) < 0.5
        c1[mask], c2[mask] = p2[mask], p1[mask]
    return c1, c2


def bitflip_mutation(chrom, rng, p_m):
    """Flip each bit independently with probability p_m."""
    return chrom ^ (rng.random(chrom.shape) < p_m)


# --------------------------------------------------- 6. GA loop
def hamming_diversity(pop):
    """Mean pairwise Hamming distance of the population (in bits)."""
    n = len(pop)
    if n < 2:
        return 0.0
    c = pop.sum(0)
    return float((c * (n - c)).sum() / (n * (n - 1) / 2))


def run_ga(X, y, groups, seed, pop_size=30, n_generations=40, patience=10,
           pool_size=40, pool_seed=0, lam=0.05, n_splits=5,
           p_c=0.9, p_m=None, tournament_k=3, elitism=1):
    """Run the GA.

    Stopping rule: stop at n_generations, or earlier if the best fitness has
    not improved for `patience` generations.
    p_m defaults to 1 / pool_size per bit.
    Returns a dict with best chromosome / columns / fitness and `history`
    (per generation: best_fitness, mean_fitness, diversity, best_k).
    """
    rng = np.random.default_rng(seed)
    pool = make_pool(X.shape[1], pool_size, pool_seed)
    L = len(pool)
    p_m = 1.0 / L if p_m is None else p_m
    folds = group_kfold_splits(groups, n_splits)
    classes = np.unique(y)

    cache = {}

    def evaluate(pop):
        out = np.empty(len(pop))
        for i, c in enumerate(pop):
            key = c.tobytes()
            if key not in cache:
                cache[key] = fitness(c, X, y, pool, folds, classes, lam)
            out[i] = cache[key]
        return out

    pop = rng.random((pop_size, L)) < 0.5
    empty = ~pop.any(1)
    pop[empty, rng.integers(0, L, empty.sum())] = True

    hist = {"best_fitness": [], "mean_fitness": [], "diversity": [], "best_k": []}
    best_fit, best_chrom, stale = -np.inf, None, 0

    for gen in range(n_generations):
        fit = evaluate(pop)
        b = int(np.argmax(fit))
        hist["best_fitness"].append(float(fit[b]))
        hist["mean_fitness"].append(float(fit.mean()))
        hist["diversity"].append(hamming_diversity(pop))
        hist["best_k"].append(int(pop[b].sum()))

        if fit[b] > best_fit + 1e-9:
            best_fit, best_chrom, stale = float(fit[b]), pop[b].copy(), 0
        else:
            stale += 1
        if stale >= patience or gen == n_generations - 1:
            break

        # next generation: elites + children from tournament parents
        nxt = [pop[i].copy() for i in np.argsort(-fit)[:elitism]]
        while len(nxt) < pop_size:
            i, j = tournament_select(fit, rng, 2, tournament_k)
            c1, c2 = uniform_crossover(pop[i], pop[j], rng, p_c)
            nxt += [bitflip_mutation(c1, rng, p_m), bitflip_mutation(c2, rng, p_m)]
        pop = np.array(nxt[:pop_size])

    return {
        "best_chrom": best_chrom,
        "best_features": decode(best_chrom, pool),
        "best_fitness": best_fit,
        "pool": pool,
        "history": {k: np.array(v) for k, v in hist.items()},
    }
