# Work review log — HIT140 Assessment 3, Darwin Group 26

Each contribution was reviewed against the project brief and the marking rubric before it
entered the report. This log records what was checked, what was found, and how it was
resolved. Issue numbers refer to the repository issue board.

## 9 October — data pipeline (reviewer: Sihao Cui)

Checked `src/build_datasets.py` against the brief's requirement that every explanatory
variable be available before kick-off.

- **Finding.** No leakage. The builder groups fixtures by date and updates each team's
  running record only after every match on that date has been read, so variables draw only
  on pre-tournament inputs or on matches played on strictly earlier dates.
- **Finding.** Row counts, variable counts and concept-level overlap are asserted by the
  script itself, and the opening round is handled by shrinkage towards a prior rather than
  by imputation or dropped rows.
- **Outcome.** Accepted without change (issue #2).

## 11 October — LR2.2 section (reviewer: Sihao Cui, author: Luke Ren)

- **Finding.** The report section tabulated only the six variables retained by the subset
  search. The brief requires exactly eight explanatory variables in the dataset, and a
  marker reading only the coefficient table could conclude otherwise.
- **Resolution.** Section revised to tabulate all eight with their selection status, and to
  state both the full eight-variable and the reduced six-variable regression equations
  (issue #8).
- **Finding.** The notebook supplied with the section did not contain the 80/20 hold-out
  evaluation whose figures the text quotes.
- **Resolution.** `src/evaluate_week9.py` was written independently and reproduces those
  figures exactly (MAE 0.918, RMSE 1.280, R² 0.184), so every number in the report now has
  a source in this repository. The original working notebook is still outstanding
  (issue #12).

## 12 October — LR2.1 section (reviewer: Sihao Cui, author: Peng Song)

- **Finding.** The final model is reported as `goal_diff = 0.90 × strength + 0.82 × host`,
  where `strength` is a composite of two of the eight variables. As written it could be
  read as a ninth variable.
- **Resolution.** Section and notebook revised to state explicitly that `strength` combines
  `log_rank_diff` and `log_value_diff` with equal weights and that the model therefore uses
  three of the eight original variables.

## 13 October — rubric coverage (reviewer: Sihao Cui)

- **Finding.** Neither notebook contained the Week 9 practical workflow the rubric asks for:
  a train/test split, the five performance metrics, and a training-set-mean baseline.
- **Resolution.** `src/evaluate_week9.py` supplies all three for both models (issue #5).
- **Finding.** The LR2.2 work contained no exploratory analysis and no comparison against
  competing algorithms, both of which the rubric asks for.
- **Resolution.** `src/compare_models_lr22.py` supplies Pearson correlations and a
  comparison against ridge, lasso, elastic net, polynomial regression, random forest and
  gradient boosting. Linear regression beat every non-linear and ensemble competitor, which
  is reported in the conclusions as a recommendation (issue #6).

## 13 October — squad valuation dating (reviewer: Sihao Cui)

- **Finding.** The data dictionary flagged that one source dated the valuations 15 July 2026
  and another June 2026. A mid-tournament valuation would constitute leakage, and squad
  value is the strongest single predictor in both models.
- **Resolution.** The figures match a publication dated 18 June 2026 digit for digit for all
  48 teams; the July page republishes identical numbers, so July is the article date rather
  than the valuation date. Cross-check: the 11 June FIFA ranking explains tournament scoring
  better than squad value (r = −0.81 against +0.75), the reverse of what a mid-tournament
  revision would produce. Treated as a pre-tournament snapshot (issue #7).

## 14 October — report consolidation (reviewer: Sihao Cui)

- **Finding.** The two sections together ran to 1,132 words with two large tables, well over
  the two pages available for them.
- **Resolution.** Both condensed during consolidation. The finished report occupies about
  three and a half of the four permitted pages.
- **Finding.** The two sections drew opposite conclusions about in-tournament form.
- **Resolution.** Rather than hiding the tension, the conclusions section reconciles it:
  LR2.1 shows form's partial correlation collapsing to −0.06 once strength is controlled
  for, which explains the small and counter-intuitively signed form coefficients in LR2.2
  as collinearity rather than a tactical effect.
