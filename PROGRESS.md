# PROGRESS

Instructor dashboard — one row per Course Week. Verifiable in under 2 minutes.

| Period | Topic | Post | Drill | First evidence | Revision commit | Tag | Status |
|---|---|---|---|---|---|---|---|
| W01–W04 | PRE-RELEASE catch-up (Ch.1 Foundations, Ch.2 Decision Tree) | `docs/pre-release/PRE_RELEASE_CATCHUP.md` | release-day baseline diagnostic | `exercises/release-baseline-w01-w02.pdf` | — | `release-baseline` | PRE-RELEASE |
| W05 | Perceptron/Delta + repo onboarding | `docs/weekly/w05-perceptron-delta.md` | `exercises/w05-first-attempt.pdf` | pre-ref commit | post-ref commit | `w05` | ACTIVE |
| W05 | ANN / Backpropagation | `docs/weekly/w05-ann-backprop.md` | `exercises/w05-ann-first-attempt.pdf` | pre-ref commit | post-ref commit | `w05` | ACTIVE |
| W06 | Bayesian learning / Naive Bayes | | | | | | PENDING |
| W07 | Genetic Algorithm | | | | | | PENDING |
| W07 | Bayesian Network / TAN — Part I closeout | | | | | | PENDING |
| W08 | MIDTERM (protected week, 16 Oct 2026) | `exam/midterm-reflection.md` | timed rehearsal | | | `w08-midterm` | PENDING |
| W09 | HMM (Forward / Viterbi) | | | | | | PENDING |
| W10 | SVM: max-margin / soft margin | | | | | | PENDING |
| W11 | Kernel SVM + cross-model comparison | | | | | | PENDING |
| W12 | PCA | | | | | | PENDING |
| W13 | LDA / feature engineering | | | | | | PENDING |
| W14 | Bagging / Boosting / AdaBoost | | | | | | PENDING |
| W15 | Generative vs discriminative; Logistic/MaxEnt/CRF; synthesis | | | | | | PENDING |

## Checkpoints

| Checkpoint | Due | Status |
|---|---|---|
| R0 — setup gate | before W06 class + 1 day | IN PROGRESS |
| Part I (40/100) | 14 Oct 2026 | PENDING |
| Part II (60/100) | 2 calendar days before final exam | PENDING |

## Dataset / use case

- **Canonical**: UCI Human Activity Recognition Using Smartphones — predict physical activity from smartphone inertial sensors.
- **Status**: confirmed at R0 (2026-09-30).

## Frozen protocol (fill at R0 close)

- Split policy: predefined subject-wise split shipped with the dataset (21 subjects train, 9 subjects test); validation by `GroupKFold` on subject id inside train only; test set sealed until final comparison
- Primary metric: Macro-F1 (secondary: accuracy, confusion matrix)
- Random seed: 42
