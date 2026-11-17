"""Report figures and tables (matplotlib, static PNG) from walk-forward predictions.

Model-selection protocol: the headline model of each experiment is the best fitted model on the DEVELOPMENT period
(2013-2020); its holdout (2021-2026) numbers are then reported as they come. Holdout results are never used to pick
models.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import common_sample  # noqa: E402
from .paths import REPORTS  # noqa: E402

FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, aqua (first three validate all-pairs)
DEEMPH = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "text.parse_math": False, "legend.frameon": False,
})

LABEL = {
    "B0_random_walk": "Random walk (last quarter)", "B1_seasonal_rw": "Seasonal random walk",
    "B2_hist_avg_growth": "Historical avg growth", "B2_hist_avg_level": "Historical avg level",
    "B3_hist_seasonal_avg": "Historical seasonal avg", "L1_linear_baseline": "Simple linear baseline",
    "G0_guidance_mid": "Guidance midpoint", "G0_gaap_guidance_mid": "GAAP guidance midpoint",
    "G3_guidance+pooled_beat": "Guidance + market-wide avg beat",
    "G1_guidance+hist_beat": "Guidance + avg past beat", "G2_guidance+EB_beat": "Guidance + Bayesian beat",
    "C0_consensus": "Analyst consensus",
}


def label(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    base, _, fs = m.rpartition("_")
    names = {"M1_ols": "OLS", "M2_ridge": "Ridge", "M3_lasso": "Lasso", "M4_bayes_ridge": "Bayesian ridge",
             "M5_random_forest": "Random forest", "M6_gbm": "Gradient boosting", "M7_xgboost": "XGBoost"}
    return f"{names.get(base, base)} ({'history' if fs == 'hist' else 'with guidance'})"


def best_dev_model(dev_table: pd.DataFrame, metric: str, prefix: str = "M") -> str:
    t = dev_table[dev_table["model"].str.startswith(prefix)]
    return t.sort_values(metric).iloc[0]["model"]


def bar_chart(t: pd.DataFrame, metric: str, title: str, xlabel: str, path, highlight: dict[str, str]):
    """Ranked horizontal bars, one hue per role: highlighted models in their hue, the rest de-emphasised."""
    t = t.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(t) + 1.3))
    y = np.arange(len(t))[::-1]
    colors = [highlight.get(m, DEEMPH) for m in t["model"]]
    ax.barh(y, t[metric], height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [label(m) for m in t["model"]])
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, t[metric]):
        ax.text(v, yi, f" {v:.2f}", va="center", ha="left", color=INK2, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, t[metric].max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def yearly_error(pred: pd.DataFrame, experiment: str, models: list[str], revenue: bool) -> pd.DataFrame:
    p = common_sample(pred[pred["experiment"] == experiment], models).copy()
    p["year"] = pd.to_datetime(p["origin_utc"]).dt.year
    err = p["yhat"] - p["y"]
    p["loss"] = (np.exp(err) - 1).abs() * 100 if revenue else err.abs()
    return p.pivot_table(index="year", columns="model", values="loss", aggfunc="mean")[models]


def line_chart(df: pd.DataFrame, title: str, ylabel: str, path, colors: list[str], holdout_year: int | None = 2021):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for (m, s), c in zip(df.items(), colors):
        ax.plot(s.index, s.values, color=c, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2,
                label=label(m))
        ax.annotate(label(m), (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    if holdout_year and holdout_year <= df.index.max():
        ax.axvline(holdout_year - 0.5, color=AXIS, linewidth=1)
        ax.text(holdout_year - 0.4, ax.get_ylim()[1] * 0.97, "holdout", va="top", fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xticks(list(range(int(df.index.min()), int(df.index.max()) + 1, 2)))
    ax.set_xlim(df.index.min() - 0.3, df.index.max() + 2.8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def beat_histogram(beats: pd.Series, path):
    b = (np.exp(beats.dropna()) - 1) * 100
    b = b.clip(-20, 30)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.hist(b, bins=np.arange(-20, 31, 1), color=S1, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK2, linewidth=1)
    med = float(np.median(b))
    ax.annotate(f"median {med:+.1f}%", (med, ax.get_ylim()[1] * 0.92), xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=INK2)
    ax.set_xlabel("Reported revenue vs guidance midpoint (%, clipped at -20 / +30)")
    ax.set_ylabel("Company-quarters")
    ax.set_title("Semiconductor companies beat their own revenue guidance far more often than they miss",
                 loc="left", fontsize=11, pad=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(pred: pd.DataFrame, experiment: str, model: str, path, period: str, n_bins: int = 10):
    p = pred[(pred["experiment"] == experiment) & (pred["model"] == model) & (pred["period"] == period)].copy()
    p["s_hat"] = (p["yhat"] - p["guid_g"]) * 100
    p["s"] = (p["y"] - p["guid_g"]) * 100
    p["bin"] = pd.qcut(p["s_hat"], n_bins, labels=False, duplicates="drop")
    g = p.groupby("bin").agg(pred=("s_hat", "mean"), real=("s", "mean"), n=("s", "size"))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    lo, hi = min(g["pred"].min(), g["real"].min()) - 1, max(g["pred"].max(), g["real"].max()) + 1
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1)
    ax.plot(g["pred"], g["real"], color=S1, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlabel("Predicted beat vs guidance (%), decile mean")
    ax.set_ylabel("Realised beat (%), decile mean")
    ax.set_title(f"Calibration, {label(model)}, {period}", loc="left", fontsize=10.5, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return g


def company_chart(pred: pd.DataFrame, ticker: str, model: str, path):
    """Realised vs predicted revenue beat (% vs guidance midpoint) for one company, quarter by quarter."""
    p = pred[(pred["experiment"] == "rev_guided") & (pred["ticker"] == ticker) & (pred["model"] == model)].copy()
    if p.empty:
        return
    p = p.sort_values("period_end")
    real = (np.exp(p["y"] - p["guid_g"]) - 1) * 100
    hat = (np.exp(p["yhat"] - p["guid_g"]) - 1) * 100
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(p["period_end"], real, color=INK2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Realised beat vs guidance midpoint")
    ax.plot(p["period_end"], hat, color=S1, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=f"Predicted beat ({label(model)})")
    ax.set_ylabel("% vs guidance midpoint")
    ax.set_title(f"{ticker}: predicted vs realised revenue beat, forecast at the close before each release",
                 loc="left", fontsize=11, pad=10)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(t: pd.DataFrame, cols: list[str], fmt: dict[str, str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|\n"
    rows = []
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "model":
                cells.append(label(v))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else fmt.get(c, "{:.2f}").format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows)
