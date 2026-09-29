# CO3117 Machine Learning — Individual Longitudinal Assignment

One dataset, one use case, many models. Each week the model changes, the problem does not.

- **Use case**: predict a person's current physical activity from smartphone inertial sensors.
- **Dataset**: UCI *Human Activity Recognition Using Smartphones* (30 subjects, 10,299 windows, 561 features, 6 activities).
- **Progress dashboard**: [PROGRESS.md](PROGRESS.md) · **Model records**: [MODEL_LOG.md](MODEL_LOG.md) · **Sources**: [REFERENCES.md](REFERENCES.md) · **AI use log**: [AI_USE.md](AI_USE.md)

## Common protocol (proposed at R0; confirm before freezing)

| Item | Choice |
|---|---|
| Split | Predefined subject-wise split shipped with the dataset: 21 subjects train (7,352 windows), 9 subjects test (2,947). No subject appears in both. |
| Validation | `GroupKFold` by subject id, inside the training set only. |
| Test set | Sealed until final model comparison. |
| Primary metric | Macro-F1. Also report accuracy and the confusion matrix. |
| Preprocessing | Scalers, PCA/LDA, feature selection fitted on train only. |
| Baseline | Majority-class predictor, kept all semester. |
| Random seed | 42 |

## Data

Download the dataset from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones and unzip into `data/`. Raw data is not committed (see `.gitignore`).

## Setup and run

```bash
pip install -r requirements.txt
python -m pytest tests -q
python experiments/part1_pre_midterm/w01_w04_tree_catchup.py
```

## Layout

| Path | Contents |
|---|---|
| `src/from_scratch/` | Own implementations (e.g. `decision_tree_split.py`) |
| `src/data.py`, `src/metrics.py` | Shared data loading and metrics |
| `experiments/` | Scripts per part (`part1_pre_midterm/`, `part2_post_midterm/`) |
| `tests/` | Unit tests for own code |
| `docs/pre-release/` | W01–W04 catch-up |
| `docs/weekly/` | One post per week |
| `exercises/`, `exam/` | Handwritten drills and exam-preparation material |
| `results/` | `metrics.csv` and figures |
