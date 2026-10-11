# HIT140 Assessment 3 — Group Project Report (Objective 2)

**Charles Darwin University · HIT140 Foundations of Data Science · Semester 2, 2026**
**Darwin Group 26** — Abel Qin · Luke Ren · Peng Song · Sihao Cui

Two multiple linear regression models on FIFA World Cup 2026 data. Every explanatory
variable is information available **before kick-off**; nothing from the match itself is used.

| Model | Rows | Target | Explanatory variables |
|---|---|---|---|
| **LR2.1** | 104 (one per match) | `goal_diff` = team1 goals − team2 goals | 8, all expressed as team1 − team2 |
| **LR2.2** | 208 (one per team-match) | `goals_scored` by that team | 8 |

Shared between the two models, counted at concept level: FIFA rank, squad value,
host-at-home, stage = **4** (the brief allows a maximum of 4). No column name is identical.

Goals are counted **after extra time, excluding penalty shoot-outs**.

## Who did what

| Member | Role | Contribution |
|---|---|---|
| **Abel Qin** | 1 — Data & Features | `src/build_datasets.py`, `docs/DATA_DICTIONARY.md`, both datasets |
| **Peng Song** | 2 — LR2.1 | `notebooks/lr21_goal_difference_peng.ipynb`, `figures/lr21/`, LR2.1 report section |
| **Luke Ren** | 3 — LR2.2 | `notebooks/lr22_goals_scored_luke.ipynb`, `figures/lr22/`, LR2.2 report section |
| **Sihao Cui** | 4 — Report & Submission | Report consolidation, Turnitin, AI declaration, repository, submission |

## Layout

```
src/build_datasets.py     Builds both datasets from the raw inputs. One script, one command.
data/                     The two modelling datasets plus match metadata.
notebooks/                One notebook per model.
figures/lr21, figures/lr22   Every figure produced by the notebooks.
docs/                     Data dictionary and the two report sections.
report/                   Final 4-page group report.
```

## Reproducing the datasets

```bash
pip install -r requirements.txt
python src/build_datasets.py
```

This regenerates `data/lr21_match_level_104.csv` and `data/lr22_team_match_208.csv`,
plus `candidate_features_match_level.csv` and `candidate_features_team_match.csv`
(every engineered column, so variables can be swapped).

The script asserts its own output: row counts, exactly eight explanatory variables per
model, no missing values, concept-level overlap ≤ 4, and that the goal totals match the
source results.

## Data sources

| Input | Source |
|---|---|
| Match results (all 104) | OpenFootball `worldcup.json` (CC0) — github.com/openfootball/worldcup.json |
| FIFA ranking | Release of 11 June 2026, before the tournament |
| Squad market value | Transfermarkt, pre-tournament valuation |
| Confederation | Standard FIFA membership |

Ranking and squad value are outside the three sites listed in the project brief. The
unit coordinator confirmed on the Objective 2 Padlet that additional sources may be used
provided they are cited.

## How leakage is avoided

`build_datasets.py` walks the fixtures grouped by date. Each team's running record is
updated only **after every match on that date has been read**, so a match's explanatory
variables can only draw on:

1. pre-tournament inputs (rank, squad value, confederation, host status), or
2. results of matches played on **strictly earlier dates**.

Same-day matches never feed each other, and nothing from the match being predicted is used.

In-tournament form is **shrunk** toward a prior — `(sum + 2 × prior) / (games + 2)`, with
priors of 1.3 goals scored/conceded and 1.0 points per game — so every team starts its
first match at the prior instead of a missing value. This is why all 104 and 208 rows are
complete without any imputation or dropped rows.

## Still to add

- [ ] `data/raw/worldcup.json` and `data/raw/team_inputs.csv` (Abel)
- [ ] `candidate_features_match_level.csv`, `candidate_features_team_match.csv` (Abel)
- [ ] Final 4-page report in `report/` (Sihao)
