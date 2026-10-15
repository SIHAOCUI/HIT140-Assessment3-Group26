"""
HIT140 A3 / Objective 2 - LR2.2 exploratory analysis and competing-model comparison.

Added by Sihao Cui (Role 4) during report consolidation, to cover two things the
LR2.2 notebook did not include:

  1. Exploratory data analysis - Pearson correlation of each explanatory
     variable with the response.
  2. A comparison of linear regression against competing regression algorithms
     (regularised linear, polynomial, and tree ensembles), scored on the same
     80/20 holdout used in evaluate_week9.py.
  3. The same comparison under repeated 10-fold cross-validation (30 repeats),
     so the ranking does not rest on a single 42-row test set.

    python src/compare_models_lr22.py

Writes results/lr22_eda_correlations.csv, results/lr22_competing_models.csv and
results/lr22_competing_models_cv.csv.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

HERE = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = ["team_gf_pg", "opp_ga_pg", "opp_gf_pg", "team_log_rank",
            "opp_log_rank", "team_log_value", "is_host_home", "is_knockout"]
TARGET = "goals_scored"

MODELS = {
    "Linear regression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05)),
    "Elastic net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5)),
    "Polynomial (degree 2)": make_pipeline(
        PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0)),
    "Random forest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
    "Gradient boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    # Week 9 optimisation tips 3 and 4, applied to the linear model
    "Linear regression, z-score scaled": make_pipeline(StandardScaler(), LinearRegression()),
    "Linear regression, Yeo-Johnson": make_pipeline(
        PowerTransformer(method="yeo-johnson"), LinearRegression()),
}


def score(y_true, y_pred, y_range):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {"MAE": metrics.mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": rmse, "Normalised RMSE": rmse / y_range,
            "R2": metrics.r2_score(y_true, y_pred)}


def main():
    df = pd.read_csv(HERE / "data" / "lr22_team_match_208.csv")
    out = HERE / "results"
    out.mkdir(exist_ok=True)

    # ---- 1. EDA: how each explanatory variable relates to the response
    eda = pd.DataFrame({
        "variable": FEATURES,
        "pearson_r": [np.corrcoef(df[c], df[TARGET])[0, 1] for c in FEATURES],
    })
    eda["abs_r"] = eda.pearson_r.abs()
    eda = eda.sort_values("abs_r", ascending=False).drop(columns="abs_r")
    eda.round(4).to_csv(out / "lr22_eda_correlations.csv", index=False)
    print("Pearson correlation with", TARGET)
    for _, r in eda.iterrows():
        print(f"  {r.variable:<16}{r.pearson_r:+.3f}")

    # ---- 2. Competing algorithms on the same holdout
    x, y = df[FEATURES].values, df[TARGET].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_range = y.max() - y.min()

    rows = [{"model": "Training-set mean (naive baseline)",
             **score(y_test, [y_train.mean()] * len(y_test), y_range)}]
    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        rows.append({"model": name, **score(y_test, model.predict(x_test), y_range)})

    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "lr22_competing_models.csv", index=False)

    print(f"\nCompeting models, 80/20 holdout (random_state={RANDOM_STATE})")
    print(f"{'model':<36}{'MAE':>8}{'RMSE':>8}{'NRMSE':>8}{'R2':>8}")
    for _, r in table.iterrows():
        print(f"{r.model:<36}{r.MAE:>8.3f}{r.RMSE:>8.3f}{r['Normalised RMSE']:>8.3f}{r.R2:>8.3f}")

    # ---- 3. The same competitors under repeated 10-fold cross-validation
    competitors = ["Linear regression", "Ridge", "Lasso", "Elastic net",
                   "Polynomial (degree 2)", "Random forest", "Gradient boosting"]
    folds = RepeatedKFold(n_splits=10, n_repeats=30, random_state=RANDOM_STATE)
    fold_rmse = {name: [] for name in ["Training-set mean (naive baseline)"] + competitors}
    for train_idx, test_idx in folds.split(x):
        y_fold = y[test_idx]
        fold_rmse["Training-set mean (naive baseline)"].append(
            math.sqrt(np.mean((y_fold - y[train_idx].mean()) ** 2)))
        for name in competitors:
            pred = clone(MODELS[name]).fit(x[train_idx], y[train_idx]).predict(x[test_idx])
            fold_rmse[name].append(math.sqrt(np.mean((y_fold - pred) ** 2)))

    cv = pd.DataFrame({
        "model": list(fold_rmse),
        "cv_rmse": [np.mean(v) for v in fold_rmse.values()],
        "cv_rmse_se": [np.std(v) / math.sqrt(len(v)) for v in fold_rmse.values()],
    }).sort_values("cv_rmse")
    cv.round(4).to_csv(out / "lr22_competing_models_cv.csv", index=False)

    print("\nCompeting models, repeated 10-fold cross-validation (30 repeats)")
    for _, r in cv.iterrows():
        print(f"  {r.model:<36}{r.cv_rmse:>8.3f}  (se {r.cv_rmse_se:.3f})")
    print("\nwritten: results/lr22_eda_correlations.csv, results/lr22_competing_models.csv, "
          "results/lr22_competing_models_cv.csv")


if __name__ == "__main__":
    main()
