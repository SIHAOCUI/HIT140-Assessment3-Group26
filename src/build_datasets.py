"""
A3 / Objective 2 - dataset builder (Role 1: Data & Features)

Inputs  (raw/):
  worldcup.json     OpenFootball (CC0) - all 104 WC2026 matches with scores
  team_inputs.csv   per-team PRE-TOURNAMENT inputs: confederation, FIFA rank (11 Jun 2026),
                    Transfermarkt squad value (EUR m)

Outputs:
  lr21_match_level_104.csv   one row per match   -> target: goal_diff (team1 - team2)
  lr22_team_match_208.csv    one row per team-match -> target: goals_scored
  candidate_features_*.csv   every engineered column (so Roles 2/3 can swap variables)

Leakage rule: every explanatory variable for a match uses only (a) pre-tournament inputs or
(b) results of matches played on EARLIER DATES. Nothing from the match itself.

Target definition: goals after extra time, shoot-out goals excluded.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"

# ----- shrinkage priors for in-tournament form (few games per team => raw averages are noisy)
K = 2.0                 # pseudo-games of prior
PRIOR_GF = 1.3          # goals scored per team per match (approx. long-run World Cup average)
PRIOR_GA = 1.3
PRIOR_PPG = 1.0         # points per game
REST_CAP = 7            # rest days capped at 7; first match of a team assumed = 7

HOST_COUNTRY = {"Mexico": "Mexico", "Canada": "Canada", "USA": "USA"}
VENUE_COUNTRY = {
    "Mexico City": "Mexico", "Guadalajara (Zapopan)": "Mexico", "Monterrey (Guadalupe)": "Mexico",
    "Toronto": "Canada", "Vancouver": "Canada",
}  # every other venue is in the USA

STAGE_ORD = {"Round of 32": 4, "Round of 16": 5, "Quarter-final": 6,
             "Semi-final": 7, "Match for third place": 7, "Final": 7}


def load_matches() -> pd.DataFrame:
    raw = json.load(open(RAW / "worldcup.json", encoding="utf-8"))["matches"]
    rows = []
    for i, m in enumerate(raw):
        s = m["score"]
        g1, g2 = s.get("et", s["ft"])            # cumulative score after extra time (if played)
        rows.append(dict(
            date=pd.Timestamp(m["date"]), round=m["round"], group=m.get("group"),
            team1=m["team1"], team2=m["team2"], goals1=g1, goals2=g2,
            ft1=s["ft"][0], ft2=s["ft"][1], went_to_pens="p" in s,
            venue=m["ground"], src_order=i,
        ))
    df = pd.DataFrame(rows).sort_values(["date", "src_order"]).reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))
    return df


def venue_country(v: str) -> str:
    return VENUE_COUNTRY.get(v, "USA")


def build():
    matches = load_matches()
    teams = pd.read_csv(RAW / "team_inputs.csv").set_index("team")
    assert len(matches) == 104, len(matches)
    assert set(matches.team1) | set(matches.team2) == set(teams.index), "team name mismatch"

    # running per-team state, updated AFTER each date has been processed
    state = {t: dict(n=0, pts=0, gf=0, ga=0, cs=0, last=None) for t in teams.index}
    team_rows, match_rows = [], []

    for date, day in matches.groupby("date", sort=True):
        pending = []
        for _, m in day.iterrows():
            feats = {}
            for side, team, opp, gf, ga in (("1", m.team1, m.team2, m.goals1, m.goals2),
                                            ("2", m.team2, m.team1, m.goals2, m.goals1)):
                st = state[team]
                n = st["n"]
                rest = REST_CAP if st["last"] is None else min((date - st["last"]).days, REST_CAP)
                f = dict(
                    matches_before=n,
                    gf_pg=(st["gf"] + K * PRIOR_GF) / (n + K),
                    ga_pg=(st["ga"] + K * PRIOR_GA) / (n + K),
                    ppg=(st["pts"] + K * PRIOR_PPG) / (n + K),
                    rest_days=rest,
                    log_rank=math.log(teams.loc[team, "fifa_rank_jun2026"]),
                    log_value=math.log(teams.loc[team, "squad_value_eur_m"]),
                    is_host_home=int(HOST_COUNTRY.get(team) == venue_country(m.venue)),
                    confed=teams.loc[team, "confederation"],
                )
                f["gd_pg"] = f["gf_pg"] - f["ga_pg"]
                feats[side] = f
                pending.append((team, gf, ga))

            ko = int(m["round"] in STAGE_ORD)
            stage_ord = STAGE_ORD.get(m["round"], feats["1"]["matches_before"] + 1)  # group: matchday 1-3
            f1, f2 = feats["1"], feats["2"]

            match_rows.append(dict(
                match_id=m.match_id, date=date.date(), round=m["round"],
                team1=m.team1, team2=m.team2, goals1=m.goals1, goals2=m.goals2,
                # ---- candidate explanatory variables (team1 minus team2)
                log_rank_diff=f2["log_rank"] - f1["log_rank"],          # + => team1 better ranked
                log_value_diff=f1["log_value"] - f2["log_value"],
                host_home_diff=f1["is_host_home"] - f2["is_host_home"],
                ppg_diff=f1["ppg"] - f2["ppg"],
                gd_pg_diff=f1["gd_pg"] - f2["gd_pg"],
                rest_days_diff=f1["rest_days"] - f2["rest_days"],
                stage_ord=stage_ord,
                same_confed=int(f1["confed"] == f2["confed"]),
                goal_diff=m.goals1 - m.goals2,
            ))
            for side, team, opp, gf, ga in (("1", m.team1, m.team2, m.goals1, m.goals2),
                                            ("2", m.team2, m.team1, m.goals2, m.goals1)):
                me, you = feats[side], feats["2" if side == "1" else "1"]
                team_rows.append(dict(
                    match_id=m.match_id, date=date.date(), round=m["round"], team=team, opponent=opp,
                    team_gf_pg=me["gf_pg"], opp_ga_pg=you["ga_pg"], opp_gf_pg=you["gf_pg"],
                    team_log_rank=me["log_rank"], opp_log_rank=you["log_rank"],
                    team_log_value=me["log_value"],
                    is_host_home=me["is_host_home"], is_knockout=ko,
                    goals_scored=gf,
                ))

        # update state only after every match of the day is read (no same-day leakage)
        for team, gf, ga in pending:
            st = state[team]
            st["n"] += 1; st["gf"] += gf; st["ga"] += ga; st["cs"] += int(ga == 0)
            st["pts"] += 3 if gf > ga else 1 if gf == ga else 0
            st["last"] = date

    return pd.DataFrame(match_rows), pd.DataFrame(team_rows)


LR21_X = ["log_rank_diff", "log_value_diff", "host_home_diff", "ppg_diff",
          "gd_pg_diff", "rest_days_diff", "stage_ord", "same_confed"]
LR22_X = ["team_gf_pg", "opp_ga_pg", "opp_gf_pg", "team_log_rank", "opp_log_rank",
          "team_log_value", "is_host_home", "is_knockout"]

# concept-level overlap check (columns measuring the same underlying idea count as shared)
CONCEPT = {
    "log_rank_diff": "fifa_rank", "team_log_rank": "fifa_rank", "opp_log_rank": "fifa_rank",
    "log_value_diff": "squad_value", "team_log_value": "squad_value",
    "host_home_diff": "host_home", "is_host_home": "host_home",
    "stage_ord": "stage", "is_knockout": "stage",
    "ppg_diff": "form_points", "gd_pg_diff": "form_goal_diff", "rest_days_diff": "rest",
    "same_confed": "confederation",
    "team_gf_pg": "form_goals_for", "opp_ga_pg": "form_goals_against", "opp_gf_pg": "form_opp_goals_for",
}


def main():
    m, t = build()
    out = HERE.parent / "data"
    m.to_csv(out / "candidate_features_match_level.csv", index=False)
    t.to_csv(out / "candidate_features_team_match.csv", index=False)

    lr21 = m[["match_id", "team1", "team2"] + LR21_X + ["goal_diff"]]
    lr22 = t[["match_id", "team", "opponent"] + LR22_X + ["goals_scored"]]
    lr21.to_csv(out / "lr21_match_level_104.csv", index=False)
    lr22.to_csv(out / "lr22_team_match_208.csv", index=False)

    # ---- validation
    assert lr21.shape == (104, 3 + 8 + 1), lr21.shape
    assert lr22.shape == (208, 3 + 8 + 1), lr22.shape
    assert len(LR21_X) == 8 and len(LR22_X) == 8
    assert lr21[LR21_X].notna().all().all() and lr22[LR22_X].notna().all().all()
    shared_cols = set(LR21_X) & set(LR22_X)
    shared_concepts = {CONCEPT[c] for c in LR21_X} & {CONCEPT[c] for c in LR22_X}
    assert len(shared_concepts) <= 4, shared_concepts
    assert lr22.goals_scored.sum() == m.goals1.sum() + m.goals2.sum()
    assert (lr21.goal_diff.values == (m.goals1 - m.goals2).values).all()
    print("OK  LR2.1:", lr21.shape, "| LR2.2:", lr22.shape)
    print("identical column names shared:", sorted(shared_cols))
    print("shared concepts (%d):" % len(shared_concepts), sorted(shared_concepts))
    print("total goals:", int(lr22.goals_scored.sum()), "| mean GD:", round(lr21.goal_diff.mean(), 3))
    print(lr21[LR21_X].describe().T[["mean", "std", "min", "max"]].round(2))
    print(lr22[LR22_X].describe().T[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
