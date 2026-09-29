# AI_USE.md

Log of every AI-assisted learning episode, per the course IBL protocol (Section 12.1). One entry per use. Never paste a full AI-generated solution here — summarize the hint/question only.

## Entry template

| Field | Entry |
|---|---|
| Week/date | W__ / YYYY-MM-DD |
| Learning question | |
| Pre-AI evidence | commit hash or handwritten first-attempt artifact |
| AI tool | |
| Prompt purpose | Socratic hint / counterexample / debugging question / quiz |
| Hint/question received | concise summary, no pasted solution |
| Verification source | CO3117 note / textbook section / NPTEL lecture / repository reference |
| What changed | misconception, derivation, test, or code decision corrected |
| Closed-book reproduction | Yes / Not yet — link delayed-retrieval artifact |

---

## Log

*(entries below, newest first)*

### Entry 4 — Outline and formula hints for `decision_tree_split.py`

| Field | Entry |
|---|---|
| Week/date | W01–W04 (pre-release) / 2026-09-29 |
| Learning question | Which functions does a decision-tree split need, and what are the entropy / Gini / information-gain formulas? |
| Pre-AI evidence | `exercises/release-baseline-w01-w02.pdf` (W01–W04 release-day baseline). No separate committed first attempt of the code. |
| AI tool | Claude (Anthropic) |
| Prompt purpose | Hint: rough outline of the parts to implement and formula hints |
| Hint/question received | A short list of the functions needed (entropy, Gini, information gain, best threshold) and the formula for each; no code given |
| Verification source | `Decision_Tree.pdf`; hand calculation of the 0.2364-bit example |
| What changed | I wrote `src/from_scratch/decision_tree_split.py` myself from the outline and formulas. |
| Closed-book reproduction | Yes for entropy and information gain. Not yet for `best_threshold`. |

### Entry 3 — W01–W04 catch-up, Decision Tree concepts

| Field | Entry |
|---|---|
| Week/date | W01–W04 (pre-release) / 2026-09-29 |
| Learning question | Why is entropy more than "uncertainty", and how is a continuous attribute split (vs. filling missing values)? |
| Pre-AI evidence | `exercises/release-baseline-w01-w02.pdf` (handwritten, no AI, no notes; file timestamp 2026-09-29 22:51; the catch-up post had an earlier draft dated 2026-09-27) |
| AI tool | Claude (Anthropic) |
| Prompt purpose | Socratic hint |
| Hint/question received | AI asked me back several questions about entropy and about how thresholds are chosen for continuous attributes (approximate summary; exact wording not retained). |
| Verification source | `Decision_Tree.pdf` (CO3117 lecture slides) |
| What changed | Baseline defined entropy only as "uncertainty" with no formula, and used mean/most-common for continuous splits. Corrected to H = −Σ p·log₂p, information gain, and midpoint thresholds; mean/most-common is for missing values only. |
| Closed-book reproduction | Yes for entropy and information gain (hand calculation, 0.2364 bits). Not yet for `best_threshold` and the depth-vs-accuracy result. |

### Entry 2 — Unit tests for decision-tree split

| Field | Entry |
|---|---|
| Week/date | W01–W04 (pre-release) / 2026-09-29 |
| Learning question | Do `entropy`, `gini`, `information_gain` and `best_threshold` behave correctly on small cases? |
| Pre-AI evidence | `exercises/release-baseline-w01-w02.pdf` (W01–W04 release-day baseline) |
| AI tool | Claude (Anthropic) |
| Prompt purpose | Test generation |
| Hint/question received | Test cases for the split functions; the 8-positive/2-negative example (information gain 0.2364 bits) |
| Verification source | `Decision_Tree.pdf`; hand calculation of the 0.2364-bit example |
| What changed | Tests in `tests/test_decision_tree_split.py`; the worked example in the catch-up post is checked by hand and by the test. |
| Closed-book reproduction | Yes for the entropy / information-gain calculation. Not yet for `best_threshold`. |

### Entry 1 — Repository scaffolding

| Field | Entry |
|---|---|
| Week/date | Pre-release / 2026-09-27 (skeleton), README revised 2026-09-29 |
| Learning question | What must the repo skeleton, PROGRESS.md and README contain for the R0 gate? |
| Pre-AI evidence | Not applicable (setup, not a learning artifact) |
| AI tool | Claude (Anthropic) |
| Prompt purpose | Scaffolding: `README.md`, `PROGRESS.md`, the `AI_USE.md` template, the common-protocol table |
| Hint/question received | Layout and tables following Sections 5, 5.1, 9 and 10 of the assignment specification |
| Verification source | Assignment specification (Sections 5, 5.1, 10) |
| What changed | Protocol values (split, metric, seed) are proposals and are confirmed by me at R0. |
| Closed-book reproduction | Not applicable |

