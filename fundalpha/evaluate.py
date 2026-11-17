"""Out-of-sample statistics.

Revenue errors are in log points (log(rev_hat / rev_actual)); MAPE = mean |rev_hat / rev_actual - 1|.
Gross-margin errors are in percentage points.

Significance: a panel Diebold-Mariano test. For each calendar window the cross-sectional mean loss differential
d_q = mean_i [L(model) - L(reference)] is formed (so cross-sectional correlation within a reporting season is
respected), and the time series of d_q is tested with Newey-West (4 lags). t < -2 means the model's loss is
significantly lower than the reference's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def nw_tstat(x: np.ndarray, lags: int = 4) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    u = x - x.mean()
    s = u @ u / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def _ic(df: pd.DataFrame, pred: str, real: str) -> tuple[float, float, float]:
    ics = []
    for _, g in df.groupby("window"):
        g = g[[pred, real]].dropna()
        if len(g) >= 5 and g[pred].nunique() > 1:
            ics.append(spearmanr(g[pred], g[real]).statistic)
    ics = np.array(ics, dtype=float)
    pooled = spearmanr(df[pred], df[real], nan_policy="omit").statistic if df[pred].nunique() > 1 else np.nan
    if len(ics) < 4:
        return np.nan, np.nan, pooled
    return float(np.nanmean(ics)), float(np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(len(ics)))), pooled


def common_sample(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    p = pred[pred["model"].isin(models)]
    ok = p.groupby("row")["yhat"].apply(lambda s: s.notna().all() and len(s) == len(models))
    return p[p["row"].isin(ok[ok].index)]


GUIDANCE_ONLY = ("G", "C")      # benchmarks that exist only where guidance / consensus exists


def default_models(p: pd.DataFrame, experiment: str) -> list[str]:
    """Models compared in an experiment. The all-quarters experiments exclude guidance/consensus benchmarks, which
    are undefined for unguided quarters and would otherwise shrink the common sample to guided quarters only."""
    ms = sorted(p["model"].unique())
    if experiment in ("rev_all", "gm_all"):
        ms = [m for m in ms if not m.startswith(GUIDANCE_ONLY)]
    return ms


def table(pred: pd.DataFrame, experiment: str, period: str, reference: str, models: list[str] | None = None,
          beat_anchor: bool = False) -> pd.DataFrame:
    p = pred[(pred["experiment"] == experiment) & (pred["period"] == period)]
    models = models or default_models(p, experiment)
    p = common_sample(p, models)
    is_rev = experiment.startswith("rev")
    p = p.copy()
    p["err"] = p["yhat"] - p["y"]
    p["abs"] = p["err"].abs()
    p["sq"] = p["err"] ** 2
    if is_rev:
        p["ape"] = (np.exp(p["err"]) - 1).abs()
    ref = p[p["model"] == reference].set_index("row")
    rows = []
    for m in models:
        q = p[p["model"] == m]
        if q.empty:
            continue
        r = {"model": m, "n": len(q), "n_windows": q["window"].nunique()}
        if is_rev:
            r.update({"MAPE_%": 100 * q["ape"].mean(), "MdAPE_%": 100 * q["ape"].median(),
                      "RMSE_log_%": 100 * np.sqrt(q["sq"].mean())})
        else:
            r.update({"MAE_pp": q["abs"].mean(), "MdAE_pp": q["abs"].median(), "RMSE_pp": np.sqrt(q["sq"].mean())})
        nz = q["y"] != 0
        r["dir_acc_%"] = 100 * float((np.sign(q.loc[nz, "yhat"]) == np.sign(q.loc[nz, "y"])).mean())
        if beat_anchor:
            surp_hat = q["yhat"] - q["guid_g"]
            surp = q["y"] - q["guid_g"]
            k = surp_hat.abs() > 1e-9
            r["beat_dir_acc_%"] = 100 * float((np.sign(surp_hat[k]) == np.sign(surp[k])).mean()) if k.any() else np.nan
            r["base_rate_beat_%"] = 100 * float((surp > 0).mean())      # accuracy of "always predict a beat"
            qq = q.assign(s_hat=surp_hat, s=surp)
            r["IC_beat"], r["IC_beat_t"], r["IC_beat_pooled"] = _ic(qq, "s_hat", "s")
        else:
            r["IC"], r["IC_t"], r["IC_pooled"] = _ic(q, "yhat", "y")
        if m != reference and not ref.empty:
            j = q.set_index("row").join(ref[["abs", "sq"]], rsuffix="_ref")
            d_abs = (j["abs"] - j["abs_ref"]).groupby(j["window"]).mean()
            d_sq = (j["sq"] - j["sq_ref"]).groupby(j["window"]).mean()
            r["DM_t_abs"] = nw_tstat(d_abs.values)
            r["DM_t_sq"] = nw_tstat(d_sq.values)
            r["MAE_vs_ref_%"] = 100 * (j["abs"].mean() / j["abs_ref"].mean() - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def by_group(pred: pd.DataFrame, experiment: str, period: str, models: list[str], group: str) -> pd.DataFrame:
    p = pred[(pred["experiment"] == experiment) & (pred["period"] == period)]
    p = common_sample(p, models).copy()
    p["abs"] = (p["yhat"] - p["y"]).abs()
    if experiment.startswith("rev"):
        p["abs"] = (np.exp(p["yhat"] - p["y"]) - 1).abs() * 100
    return p.pivot_table(index=group, columns="model", values="abs", aggfunc="mean")
