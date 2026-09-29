# Data

**Dataset**: UCI *Human Activity Recognition Using Smartphones* — 30 subjects, 10,299 windows, 561 features, 6 activities (walking, walking upstairs, walking downstairs, sitting, standing, laying).

**Source**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

**How to get it**: download the zip from the link above and unzip it into this folder, so that the files are under `data/UCI HAR Dataset/` (`train/`, `test/`, `features.txt`, `activity_labels.txt`).

**Not committed**: the raw data and the zip are excluded by `.gitignore`. Only this README is versioned.

**Split used**: the predefined subject-wise split shipped with the dataset — 21 subjects for training (7,352 windows) and 9 for testing (2,947). See the frozen protocol in `PROGRESS.md`.
