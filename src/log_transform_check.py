"""
HIT140 A3 / Objective 2 - why FIFA ranking and squad value enter both models in logs.

Week 9 optimisation tip #1 (non-linear transformation): compare each variable with
its log, on skewness and on its correlation with the response of each model.

    python src/log_transform_check.py

Writes results/log_transform_check.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def main():
    teams = pd.read_csv(HERE / "data" / "raw" / "team_inputs.csv").set_index("team")
    rank, value = teams["fifa_rank_jun2026"], teams["squad_value_eur_m"]
    lr21 = pd.read_csv(HERE / "data" / "lr21_match_level_104.csv")
    lr22 = pd.read_csv(HERE / "data" / "lr22_team_match_208.csv")

    # LR2.1 uses team1-minus-team2 advantages: a better (smaller) rank is an advantage
    raw21 = {
        "rank": lr21.team2.map(rank) - lr21.team1.map(rank),
        "value": lr21.team1.map(value) - lr21.team2.map(value),
    }
    rows = []
    for name, series, logged, col in [
        ("rank", rank, np.log(rank), "log_rank_diff"),
        ("value", value, np.log(value), "log_value_diff"),
    ]:
        rows.append({
            "variable": name,
            "skew_raw": series.skew(), "skew_log": logged.skew(),
            "lr21_r_raw": lr21.goal_diff.corr(raw21[name]),
            "lr21_r_log": lr21.goal_diff.corr(lr21[col]),
            "lr22_r_raw": lr22.goals_scored.corr(lr22.team.map(series)),
            "lr22_r_log": lr22.goals_scored.corr(lr22[f"team_log_{name}"]),
        })

    table = pd.DataFrame(rows)
    table.round(3).to_csv(HERE / "results" / "log_transform_check.csv", index=False)
    print(table.round(3).to_string(index=False))
    print("\nwritten: results/log_transform_check.csv")


if __name__ == "__main__":
    main()
