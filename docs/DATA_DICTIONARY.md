# A3 Objective 2 – Data dictionary (Role 1)

Rebuild everything with `python build_datasets.py` (needs pandas, numpy). Raw inputs are in `raw/`.

## Sources
- Match results (all 104): OpenFootball `worldcup.json` (CC0), github.com/openfootball/worldcup.json
- FIFA ranking: release of 11 Jun 2026, as listed by MLSsoccer.com ("FIFA World Rankings: Every team at the World Cup")
- Squad market value: Transfermarkt via Planet Football / Everything Edinburgh ("World Cup 2026 squad values")
- Confederation: standard FIFA membership

## Targets
- `goal_diff` = team1 goals − team2 goals (team1 = first-named team in the schedule)
- `goals_scored` = goals by `team` in that match
- Goals counted **after extra time, excluding penalty shoot-outs** (5 matches went to extra time, 4 to penalties).

## LR2.1 (104 rows) – 8 explanatory variables, all team1 minus team2
| Column | Meaning |
|---|---|
| log_rank_diff | ln(rank of team2) − ln(rank of team1); positive = team1 better ranked |
| log_value_diff | ln(squad value team1) − ln(squad value team2) |
| host_home_diff | 1 if team1 is a host playing in its own country, minus same for team2 |
| ppg_diff | Points per game earlier in the tournament (shrunk), difference |
| gd_pg_diff | Goal difference per game earlier in the tournament (shrunk), difference |
| rest_days_diff | Days since previous match (capped at 7), difference |
| stage_ord | Group matchday 1–3; R32=4, R16=5, QF=6, SF/3rd/Final=7 |
| same_confed | 1 if both teams are in the same confederation |

## LR2.2 (208 rows) – 8 explanatory variables
| Column | Meaning |
|---|---|
| team_gf_pg | Team's goals scored per game earlier in the tournament (shrunk) |
| opp_ga_pg | Opponent's goals conceded per game earlier (shrunk) |
| opp_gf_pg | Opponent's goals scored per game earlier (shrunk) |
| team_log_rank / opp_log_rank | ln(FIFA rank) of team / opponent |
| team_log_value | ln(squad value, EUR m) of team |
| is_host_home | 1 if the team is a host playing in its own country |
| is_knockout | 1 for Round of 32 onwards |

Overlap between the two models (by concept): FIFA rank, squad value, host-at-home, stage = **4** (limit is 4). No column name is identical.

## Pre-match safeguards
- In-tournament form uses only matches on **earlier dates**; same-day matches never feed each other.
- "Shrunk" = (sum + 2 × prior) / (games + 2), priors: 1.3 goals scored/conceded, 1.0 points per game. This avoids missing/extreme values in matchday 1 (everyone starts at the prior).
- Rest days: first match assumed 7. Hosts: flag is venue-based (Canada played R32/R16 in the USA, so no host flag there).

## Caveats to resolve with the group
1. No Elo: complete Elo tables I could find were updated after the tournament (would leak results). If someone has a pre-11-June Elo table for all 48 teams, swap it in.
2. ~~Squad values: one source dates the valuation 15 Jul 2026, another June 2026 (identical numbers).~~
   **Resolved 13 Oct 2026 (Sihao).** Our figures match Everything Edinburgh (published 18 Jun 2026,
   sourced "Transfermarkt via PlanetFootball, June 2026") digit for digit for all 48 teams. The
   PlanetFootball page is dated 15 Jul 2026 but carries the identical numbers, so 15 July is the article
   date, not the valuation date. The figures did not move between 18 June and 15 July, a window covering
   almost the entire tournament, so they cannot have absorbed tournament performance. Cross-check: the
   11 June FIFA ranking explains tournament scoring better than squad value (r = -0.81 vs +0.75), the
   reverse of what a mid-tournament revision would produce. Treated as a pre-tournament snapshot.
3. Variable choice is a proposal; Roles 2/3 should confirm on 8 Oct. `candidate_features_*.csv` hold every engineered column for swapping.
4. `log_rank` and `log_value` correlate ~0.8 and `ppg_diff`/`gd_pg_diff` ~0.82 – check VIF in the models.
