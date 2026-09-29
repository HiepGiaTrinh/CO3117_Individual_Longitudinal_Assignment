# PRE-RELEASE Catch-up: Foundations (Ch.1) + Decision Trees (Ch.2)

Covers W01–W04. Written after release (drafted 2026-09-27, revised 2026-09-29). No back-dated Git history is claimed.

## A. Concept capsule

**Foundations.** Supervised learning learns a function from labeled examples so that it works on new data. The workflow is: split the data into train, validation and test; fit only on train; tune on validation; report once on test. Preprocessing (scaling, encoding) is also fitted on train only, so no test information leaks into training. Useful metrics are accuracy, precision, recall, F1 and the confusion matrix, which shows which classes get confused.

Underfitting means the model is too simple: it does badly on both train and test (high bias). Overfitting means the model memorizes noise: it does well on train but badly on test (high variance). The bias–variance tradeoff says that a more complex model lowers bias but raises variance.

**Decision tree.** A tree splits the data step by step. At each node it picks the split that reduces impurity the most. Entropy is H = −Σ p·log₂p, information gain is H(parent) − weighted H(children), and Gini is 1 − Σ p². For a continuous attribute, sort the values, try the midpoints as thresholds, and keep the best one. Missing values are filled in (mean for numbers, most common value for categories). Pruning, or limiting the depth, stops the tree from overfitting.

## B. Worked example

A node has 8 positive and 2 negative examples, so H = 0.7219 bits. A split gives child A (5 positive, 0 negative, H = 0) and child B (3 positive, 2 negative, H = 0.9710). The weighted child entropy is 0.5 × 0 + 0.5 × 0.9710 = 0.4855. The information gain is 0.7219 − 0.4855 = **0.2364 bits**. This is checked in `tests/test_decision_tree_split.py`.

## C. Code-to-theory trace

File `src/from_scratch/decision_tree_split.py`:
- `entropy(y)` is H = −Σ p·log₂p, and `gini(y)` is 1 − Σ p².
- `information_gain(y, y_left, y_right)` is the parent impurity minus the weighted child impurity.
- `best_threshold(x, y)` tries the midpoints between sorted distinct values and keeps the one with the highest gain.

The 5 unit tests pass.

## D. Experiments

Script: `experiments/part1_pre_midterm/w01_w04_tree_catchup.py` (seed 42).

**D1. Iris root split.** With both entropy and Gini, the code above and scikit-learn choose `petal length <= 2.45`. Petal width ties with it (gain 0.9183 for entropy, 0.3333 for Gini) because both separate one class perfectly, so the choice depends on tie-breaking.

**D2. Depth vs accuracy** (Breast Cancer, 70/30 split, 171 test samples):

| max_depth | train acc | test acc |
|---|---|---|
| 1 | 0.9271 | 0.9123 |
| 3 | 0.9799 | 0.9474 |
| unlimited | 1.0000 | 0.9474 |

Depth 3 already matches the unlimited tree on test accuracy, while train accuracy keeps rising. The train–test gap grows from 0.015 to 0.033 to 0.053.

## E. Failure / misconception

"A deeper tree is always better" is wrong. Beyond depth 3 the extra splits only fit the training set. Also, with 171 test samples, one sample changes accuracy by about 0.006, so tiny differences are not evidence.

## F. Written-exam capsule

A decision tree asks a series of questions about the features. At each node it picks the question that makes the labels purest, meaning the highest information gain or the lowest Gini. Trees are easy to read, but a tree grown until every leaf is pure overfits. Limiting the depth or pruning adds a little bias but removes a lot of variance. This is the bias–variance tradeoff in a concrete case.

## G. Reflection

In the handwritten baseline I described entropy only as "uncertainty", with no formula, and for continuous attributes I wrote only "mean" and "most common value". Now I can compute information gain by hand (the 0.2364-bit example) and I know a continuous attribute is split by testing midpoint thresholds, while mean or most common value is for filling missing values. Still unclear: how post-pruning decides which branch to cut.

## H. Inquiry trail

**Question I asked.** Why is entropy more than "uncertainty", and how is a continuous attribute split, as opposed to how missing values are filled? My initial belief (from the handwritten baseline) was that entropy just means uncertainty and that continuous attributes are handled with the mean.

**Pre-AI attempt.** `exercises/release-baseline-w01-w02.pdf`, done by hand with no AI and no notes on 2026-09-29 (22:51). An earlier draft of this post dated 2026-09-27 existed before the baseline; the current version was revised after it.

**AI hint (summary, no solution).** A Socratic tool (Claude) asked me back: several questions about entropy and about how thresholds are chosen for continuous attributes (approximate summary; exact wording not retained). AI was also used for scaffolding and to generate unit tests; both are logged in `AI_USE.md` (Entries 1 and 2).

**Verified against.** `Decision_Tree.pdf` (CO3117 lecture slides): the entropy and information-gain formulas, and midpoint thresholds for continuous attributes.

**What I can reproduce without AI.** The entropy and information-gain calculation by hand (the 0.2364-bit example in section B). Not yet: `best_threshold` and the depth-vs-accuracy explanation; I will do a closed-book retrieval within 24–72 hours and log it.
