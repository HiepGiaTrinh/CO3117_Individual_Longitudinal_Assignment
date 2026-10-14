# PROGRESS

One row per Course Week. Post and Drill are relative links; First evidence and Revision commit are
commit hashes (at least 7 characters). `python tools/check.py` verifies every row up to the current week.

| Period | Topic | Post | Drill | First evidence | Revision commit | Tag | Status |
|---|---|---|---|---|---|---|---|
| W01-W04 | PRE-RELEASE catch-up (Foundations, Decision Tree) | [catch-up](docs/pre-release/PRE_RELEASE_CATCHUP.md) | [release baseline](exercises/release-baseline-w01-w04.pdf) |  |  | release-baseline | PRE-RELEASE |
| W05 | Perceptron, Delta Rule & Neural Networks (ANN) + onboarding | [post](docs/weekly/w05-perceptron-ann.md) | [drill](exercises/w05-first-attempt.pdf) | 62ff803 | fd3ca44 | w05 | ACTIVE |
| W06 | Bayesian learning / Naive Bayes | [post](docs/weekly/w06-bayesian-naive-bayes.md) | [drill](exercises/w06-first-attempt.pdf) | 01f52a7 | e39ac95 | w06 | ACTIVE |
| W07 | Genetic Algorithms & Bayesian Networks (TAN), Part I closeout | [post](docs/weekly/w07-genetic-algorithms-bayesian-networks.md) | [drill](exercises/w07-first-attempt.pdf) | 2252515 | 7412379 | w07, part1-final | ACTIVE |
| W08 | MIDTERM (16 Oct) |  |  | n/a | n/a | w08-midterm | MIDTERM |
| W09 | HMM / sequence modelling |  |  |  |  | w09 |  |
| W10 | SVM: maximum and soft margin |  |  |  |  | w10 |  |
| W11 | Kernel SVM, cross-model comparison |  |  |  |  | w11 |  |
| W12 | PCA, curse of dimensionality |  |  |  |  | w12 |  |
| W13 | LDA, feature engineering |  |  |  |  | w13 |  |
| W14 | Bagging, boosting, AdaBoost |  |  |  |  | w14 |  |
| W15 | Generative vs discriminative, Logistic/MaxEnt, CRF, synthesis |  |  |  |  | w15, part2-final |  |

Status values: PRE-RELEASE (W01-W04), ACTIVE (current week), DONE (tag pushed), MIDTERM (W08).
W08: link the compact entry and the timed rehearsal if you have them (for example `docs/weekly/w08-midterm.md`,
`exercises/w08-rehearsal.pdf`); the two-state commit rule does not apply.

Example of a completed row:

`| W05 | Perceptron, Delta Rule & ANN | [post](docs/weekly/w05-perceptron-ann.md) | [drill](exercises/w05-first-attempt.pdf) | 3f9c2ab | 8d41e07 | w05 | DONE |`
