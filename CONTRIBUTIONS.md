# Individual contributions — HIT140 Assessment 3, Darwin Group 26

Roles were chosen by the team on 3 October 2026, each member claiming a numbered
role in the HIT140 Darwin Group 26 channel on Microsoft Teams.

| Role | Member | Student ID | Primary deliverable |
|---|---|---|---|
| 1 — Data and features | Abel Qin | S393924 | Both modelling datasets and the data dictionary |
| 2 — Linear Regression 2.1 | Peng Song | S394967 | Goal-difference model (35% of this assessment) |
| 3 — Linear Regression 2.2 | Luke Ren | S394678 | Goals-scored model (40% of this assessment) |
| 4 — Report and submission | Sihao Cui | S399370 | Report, verification, repository, submission |

## What each member produced

**Abel Qin — Role 1**
- `src/build_datasets.py` — builds both datasets from the raw inputs; asserts row counts,
  exactly eight explanatory variables per model, absence of missing values, and
  concept-level overlap of at most four
- `data/lr21_match_level_104.csv` (104 rows), `data/lr22_team_match_208.csv` (208 rows)
- `docs/DATA_DICTIONARY.md` — variable definitions, sources, the leakage rule, the
  shrinkage treatment of opening-round form, and open caveats
- Collected the external pre-tournament inputs (FIFA ranking, squad values, confederation)
- Delivered 7 October via Microsoft Teams

**Peng Song — Role 2**
- `notebooks/lr21_goal_difference_peng.ipynb` — 84 cells covering exploratory analysis,
  the relabelling test, repeated cross-validation, walk-forward validation and twenty
  competing models
- `figures/lr21/` — 14 figures
- `docs/lr21_report_section_peng.docx` — the LR2.1 report section, revised 12 October
- Delivered 9 October to the channel folder; revision 12 October

**Luke Ren — Role 3**
- `notebooks/lr22_goals_scored_luke.ipynb` — exhaustive 255-subset search, VIF, HC3 robust
  standard errors, residual diagnostics
- `figures/lr22/` — 5 figures
- `docs/lr22_report_section_luke.docx` — the LR2.2 report section, revised 11 October to
  tabulate all eight explanatory variables with their selection status and to state both
  the full and reduced regression equations
- Delivered 9 October; revision 11 October

**Sihao Cui — Role 4**
- `report/HIT140_A3_Group26_report.docx` — consolidated the three contributions into the
  four-page group report; wrote the introduction, the data and feature-engineering section,
  and the comparison and conclusions
- `src/evaluate_week9.py` — the Week 9 hold-out protocol (80/20 split, five metrics,
  training-set-mean baseline) for both models, which neither notebook contained
- `src/compare_models_lr22.py` — exploratory correlations and the competing-algorithm
  comparison for LR2.2
- Verified the feature pipeline for leakage and resolved the squad-value dating question
- Repository, issue board, `repo_link.txt`, Turnitin submission, AI declaration, packaging

## Timeline

| Date | Event |
|---|---|
| 3 Oct | Roles claimed on Teams; internal deadlines agreed |
| 7 Oct | Abel delivers the datasets, builder and data dictionary |
| 9 Oct | Peng and Luke deliver their models, figures and report sections |
| 9 Oct | Pipeline independently checked for leakage — no issues found |
| 11–12 Oct | Both report sections revised after review (see `docs/REVIEW_LOG.md`) |
| 13 Oct | Week 9 hold-out evaluation and competing-model comparison added; squad-value caveat resolved; first full report draft |
| 14 Oct | Student IDs added; layout relaxed within the four-page limit; Turnitin run (0% similarity); issue board opened |
| 15 Oct | AI Usage Declaration signed by all four members; submission packaged |
