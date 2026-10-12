"""
HIT140 A3 / Objective 2 - Week 9 holdout evaluation.

Reproduces the standard workflow from the Week 9 practical (multiple_linreg.py)
for both models: an 80/20 train/test split, a multiple linear regression fitted
with scikit-learn, the five required performance metrics (MAE, MSE, RMSE,
Normalised RMSE, R-squared), and a comparison against a baseline that predicts
the mean of the training set.

    python src/evaluate_week9.py

Writes results/week9_holdout_metrics.csv and prints the full regression
equation for each model.
"""
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

HERE = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
TEST_SIZE = 0.2

LR21_X = ["log_rank_diff", "log_value_diff", "host_home_diff", "ppg_diff",
          "gd_pg_diff", "rest_days_diff", "stage_ord", "same_confed"]
LR22_X = ["team_gf_pg", "opp_ga_pg", "opp_gf_pg", "team_log_rank",
          "opp_log_rank", "team_log_value", "is_host_home", "is_knockout"]
# The six variables retained by the LR2.2 subset search.
LR22_SUBSET = ["team_gf_pg", "opp_ga_pg", "team_log_rank", "opp_log_rank",
               "team_log_value", "is_host_home"]


def score(y_true, y_pred, y_range):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": rmse,
        "Normalised RMSE": rmse / y_range,
        "R2": metrics.r2_score(y_true, y_pred),
    }


def evaluate(df, features, target, label):
    x = df[features].values
    y = df[target].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Baseline: predict the mean of the response in the TRAINING set.
    y_base = [y_train.mean()] * len(y_test)

    y_range = y.max() - y.min()
    rows = [
        {"model": label, "k": len(features), "predictor": "linear regression",
         **score(y_test, y_pred, y_range)},
        {"model": label, "k": len(features), "predictor": "training-set mean",
         **score(y_test, y_base, y_range)},
    ]

    equation = f"{target} = {model.intercept_:.3f}" + "".join(
        f" {'+' if c >= 0 else '-'} {abs(c):.3f} x {f}"
        for c, f in zip(model.coef_, features))

    print(f"\n{label}  (n={len(y)}, k={len(features)}, "
          f"train={len(y_train)}, test={len(y_test)}, "
          f"training mean={y_train.mean():.3f})")
    for r in rows:
        print(f"  {r['predictor']:<20}" + "".join(
            f"{r[m]:>10.3f}" for m in ("MAE", "MSE", "RMSE", "Normalised RMSE", "R2")))
    print(f"  equation: {equation}")
    return rows


def main():
    print(f"{'':22}{'MAE':>10}{'MSE':>10}{'RMSE':>10}{'NRMSE':>10}{'R2':>10}")
    lr21 = pd.read_csv(HERE / "data" / "lr21_match_level_104.csv")
    lr22 = pd.read_csv(HERE / "data" / "lr22_team_match_208.csv")

    rows = []
    rows += evaluate(lr21, LR21_X, "goal_diff", "LR2.1 (all 8 variables)")
    rows += evaluate(lr22, LR22_X, "goals_scored", "LR2.2 (all 8 variables)")
    rows += evaluate(lr22, LR22_SUBSET, "goals_scored", "LR2.2 (6-variable subset)")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / "week9_holdout_metrics.csv"
    pd.DataFrame(rows).round(4).to_csv(path, index=False)
    print(f"\nwritten: {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
