"""
Two figures for the group report, added by Sihao Cui (Role 4) during consolidation.

  figures/report_figure_form.png      what both models say about pre-match strength
                                      versus in-tournament form
  figures/report_figure_models.png    linear regression against competing algorithms

    python src/make_report_figures.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "figures"
STRENGTH, FORM, OTHER = "#1f4e79", "#c55a11", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#555555",
                     "axes.linewidth": 0.8, "xtick.color": "#333333",
                     "ytick.color": "#333333"})


def partial_corr(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(rx, ry)[0, 1]


def figure_form():
    d22 = pd.read_csv(HERE / "data" / "lr22_team_match_208.csv")
    d21 = pd.read_csv(HERE / "data" / "lr21_match_level_104.csv")
    d21["strength"] = (d21.log_rank_diff + d21.log_value_diff) / 2

    cols = ["team_log_value", "team_log_rank", "opp_log_rank", "opp_ga_pg",
            "is_host_home", "team_gf_pg", "opp_gf_pg", "is_knockout"]
    kind = {"team_log_value": STRENGTH, "team_log_rank": STRENGTH,
            "opp_log_rank": STRENGTH, "is_host_home": STRENGTH,
            "opp_ga_pg": FORM, "team_gf_pg": FORM, "opp_gf_pg": FORM,
            "is_knockout": OTHER}
    r = [np.corrcoef(d22[c], d22.goals_scored)[0, 1] for c in cols]
    order = np.argsort(np.abs(r))
    cols_s = [cols[i] for i in order]
    r_s = [r[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    ax1.barh(range(len(cols_s)), r_s, color=[kind[c] for c in cols_s], height=0.62)
    ax1.set_yticks(range(len(cols_s)))
    ax1.set_yticklabels(cols_s, fontsize=11)
    ax1.axvline(0, color="#333333", lw=0.9)
    for i, v in enumerate(r_s):
        ax1.text(v + (0.018 if v >= 0 else -0.018), i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=10.5)
    ax1.set_xlim(-0.52, 0.56)
    ax1.set_xlabel("Pearson correlation with goals scored", fontsize=11.5)
    ax1.set_title("(a)  Model 2.2: pre-tournament strength carries the signal",
                  fontsize=12.5, loc="left", pad=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=10.5)

    pairs = [("ppg_diff", "form: points per game"),
             ("gd_pg_diff", "form: goal difference"),
             ("log_value_diff", "squad value"),
             ("log_rank_diff", "FIFA ranking")]
    raw = [np.corrcoef(d21[c], d21.goal_diff)[0, 1] for c, _ in pairs]
    par = [partial_corr(d21[c].values.astype(float),
                        d21.goal_diff.values.astype(float),
                        d21.strength.values) for c, _ in pairs]
    ypos = np.arange(len(pairs))[::-1]
    for y, a, b, (c, _) in zip(ypos, raw, par, pairs):
        col = FORM if "pg_diff" in c else STRENGTH
        ax2.annotate("", xy=(b, y), xytext=(a, y),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                     shrinkA=3, shrinkB=3))
        ax2.scatter([a], [y], s=46, color=col, zorder=3)
        ax2.scatter([b], [y], s=46, color="white", edgecolor=col, lw=1.6, zorder=3)
    ax2.axvline(0, color="#333333", lw=0.9)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([lbl for _, lbl in pairs], fontsize=11)
    ax2.set_xlim(-0.2, 0.82)
    ax2.set_xlabel("correlation with goal difference", fontsize=11.5)
    ax2.set_title("(b)  Model 2.1: controlling for strength collapses form",
                  fontsize=12.5, loc="left", pad=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=10.5)
    ax2.text(0.79, 1.50, "filled  =  raw correlation\nopen  =  partial, given strength",
             fontsize=9.5, ha="right", va="center", color="#555555")

    fig.tight_layout(pad=1.1)
    fig.savefig(OUT / "report_figure_form.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written: figures/report_figure_form.png")


def figure_models():
    df = pd.read_csv(HERE / "data" / "lr22_team_match_208.csv")
    X = ["team_gf_pg", "opp_ga_pg", "opp_gf_pg", "team_log_rank",
         "opp_log_rank", "team_log_value", "is_host_home", "is_knockout"]
    x, y = df[X].values, df.goals_scored.values
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=400, random_state=42),
        "Polynomial (degree 2)": make_pipeline(PolynomialFeatures(2, include_bias=False),
                                               StandardScaler(), Ridge(alpha=1.0)),
        "Linear regression": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05)),
        "Elastic net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5)),
    }
    base = math.sqrt(metrics.mean_squared_error(yte, [ytr.mean()] * len(yte)))
    names, rmses = [], []
    for n, m in models.items():
        m.fit(xtr, ytr)
        names.append(n)
        rmses.append(math.sqrt(metrics.mean_squared_error(yte, m.predict(xte))))

    fig, ax = plt.subplots(figsize=(11.6, 4.7))
    colours = []
    for n, v in zip(names, rmses):
        if v > base:            colours.append("#c00000")
        elif n == "Linear regression": colours.append(STRENGTH)
        else:                   colours.append("#7f9fc4")
    ax.barh(range(len(names)), rmses, color=colours, height=0.60)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11.5)
    ax.axvline(base, color="#c00000", lw=1.7, ls="--")
    ax.text(base + 0.006, len(names) - 0.42,
            f"naive baseline\n(training mean) {base:.3f}",
            color="#c00000", fontsize=10.5, va="top")
    for i, v in enumerate(rmses):
        ax.text(v - 0.012, i, f"{v:.3f}", va="center", ha="right",
                fontsize=10.5, color="white", fontweight="bold")
    ax.set_xlim(1.0, base + 0.16)
    ax.set_xlabel("Hold-out RMSE in goals  (lower is better)", fontsize=11.5)
    ax.set_title("Model 2.2: no non-linear or ensemble method beats ordinary linear regression",
                 fontsize=12.5, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10.5)
    fig.tight_layout(pad=1.1)
    fig.savefig(OUT / "report_figure_models.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written: figures/report_figure_models.png")


if __name__ == "__main__":
    figure_form()
    figure_models()
