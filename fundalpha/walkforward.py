"""Strict walk-forward (expanding window) evaluation.

For each calendar quarter Q from START: cutoff = first instant of Q (UTC).
  training rows : release_utc < cutoff   (their outcome was public before the cutoff)
  test rows     : origin_utc in [cutoff, next cutoff)
Every forecast is therefore made with models fitted only on outcomes that were public before the forecast origin,
and with features that were themselves point-in-time at the forecast origin (features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .models import (BASELINE_GM, BASELINE_REV, GM_GUID, GM_HIST, REV_GUID, REV_HIST, bench_gm, bench_revenue,
                     design, make_models)

START = "2013Q1"
DEV_END = pd.Timestamp("2021-01-01", tz="UTC")       # dev: origins before; holdout: origins on/after


def samples(panel: pd.DataFrame) -> dict[str, pd.Series]:
    base_rev = panel["eligible"] & panel["y_rev"].notna() & panel["g_seq1"].notna() & panel["seas_g1"].notna() \
        & panel["g_mean8"].notna() & panel["g_yoy1"].notna()
    base_gm = panel["eligible"] & panel["y_gm"].notna() & panel["gm_l1"].notna() & panel["gm_seas1"].notna() \
        & panel["gm_mean4"].notna() & panel["d_gm1"].notna()
    out = {
        "rev_all": base_rev,
        "rev_guided": base_rev & panel["has_guid"] & panel["y_beat"].notna(),
        "gm_all": base_gm,
        "gm_guided": base_gm & panel["guid_gm_gaap"].notna(),
    }
    if "cons_revenue" in panel and panel["cons_revenue"].notna().any():
        out["rev_consensus"] = base_rev & panel["cons_revenue"].notna()
    return out


def _eb_beat(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Empirical-Bayes (normal-normal) shrinkage of each firm's trailing mean beat toward the pooled mean."""
    t = train.dropna(subset=["y_beat"])
    mu = t["y_beat"].clip(*t["y_beat"].quantile([0.01, 0.99])).mean()
    g = t.groupby("ticker")["y_beat"]
    sigma2 = float(np.nanmean(g.var()))                       # within-firm variance
    tau2 = max(float(np.nanvar(g.mean())) - sigma2 / max(float(g.size().mean()), 1.0), 1e-6)   # between firms
    n = test["beat_n"].fillna(0).clip(lower=0)
    w = n * tau2 / (n * tau2 + sigma2)
    return mu + w * (test["beat_mean8"].fillna(mu) - mu)


def _fit_predict(models: dict, Xtr, ytr, Xte) -> dict[str, np.ndarray]:
    out = {}
    for name, m in models.items():
        try:
            m.fit(Xtr.values, ytr.values)
            out[name] = m.predict(Xte.values)
        except Exception as exc:                               # a failed fit is recorded, never silently filled
            print(f"    fit failed {name}: {type(exc).__name__}: {exc}")
            out[name] = np.full(len(Xte), np.nan)
    return out


def _window(panel, masks, q):
    cut = q.start_time.tz_localize("UTC")
    nxt = (q + 1).start_time.tz_localize("UTC")
    recs = []
    test_all = (panel["origin_utc"] >= cut) & (panel["origin_utc"] < nxt)
    for exp, mask in masks.items():
        te = panel[test_all & mask]
        if te.empty:
            continue
        tr = panel[(panel["release_utc"] < cut) & mask]
        if len(tr) < 150:
            continue
        preds = {}
        if exp.startswith("rev"):
            preds.update({k: v.loc[te.index].values for k, v in bench_revenue(te).items()})
            preds.update(_fit_predict(make_models("baseline"), design(tr, BASELINE_REV), tr["y_rev"],
                                      design(te, BASELINE_REV)))
            hist = _fit_predict(make_models("full"), design(tr, REV_HIST), tr["y_rev"], design(te, REV_HIST))
            preds.update({f"{k}_hist": v for k, v in hist.items()})
            if exp == "rev_guided":
                preds["G2_guidance+EB_beat"] = (te["guid_g"] + _eb_beat(tr, te)).values
                gd = _fit_predict(make_models("full"), design(tr, REV_GUID), tr["y_beat"], design(te, REV_GUID))
                preds.update({f"{k}_guid": te["guid_g"].values + v for k, v in gd.items()})
            target = te["y_rev"].values
        else:
            preds.update({k: v.loc[te.index].values for k, v in bench_gm(te).items()})
            preds.update(_fit_predict(make_models("baseline"), design(tr, BASELINE_GM), tr["y_gm"],
                                      design(te, BASELINE_GM)))
            hist = _fit_predict(make_models("full"), design(tr, GM_HIST), tr["y_gm"], design(te, GM_HIST))
            preds.update({f"{k}_hist": v for k, v in hist.items()})
            gd = _fit_predict(make_models("full"), design(tr, GM_GUID), tr["y_gm"], design(te, GM_GUID))
            preds.update({f"{k}_guid": v for k, v in gd.items()})
            target = te["y_gm"].values
        for name, p in preds.items():
            for idx, yhat, y in zip(te.index, p, target):
                recs.append((exp, name, idx, float(yhat) if yhat is not None else np.nan, float(y), str(q), len(tr)))
    return recs


def importance(panel: pd.DataFrame, cutoff: str, experiment: str = "rev_guided") -> pd.DataFrame:
    """Gain importance and standardized linear coefficients of models fitted on data public before `cutoff`."""
    import xgboost as xgb
    from sklearn.linear_model import RidgeCV
    masks = samples(panel)
    cut = pd.Timestamp(cutoff, tz="UTC")
    tr = panel[(panel["release_utc"] < cut) & masks[experiment]]
    cols, target = {"rev_all": (REV_HIST, "y_rev"), "rev_guided": (REV_GUID, "y_beat"),
                    "gm_all": (GM_GUID, "y_gm"), "gm_guided": (GM_GUID, "y_gm")}[experiment]
    X = design(tr, cols)
    y = tr[target].clip(*tr[target].quantile([0.01, 0.99]))
    m = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=1.0, n_jobs=4, random_state=0, verbosity=0).fit(X, y)
    gain = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0.0)
    Xs = (X - X.median()).fillna(0.0)
    Xs = Xs / Xs.std().replace(0, 1)
    r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs, y)
    out = pd.DataFrame({"feature": X.columns, "xgb_gain": gain.values, "ridge_std_coef": r.coef_,
                        "missing_share": X.isna().mean().values})
    out["xgb_gain_share"] = out["xgb_gain"] / out["xgb_gain"].sum()
    return out.sort_values("xgb_gain", ascending=False).reset_index(drop=True)


MODEL_VERSION = "2026-11-14.1"      # bump when models.py / feature lists change (invalidates window checkpoints)

SPEC = {   # model-name suffix -> (experiment mask, feature list, target, anchor column added back to the prediction)
    ("rev", "hist"): ("rev_all", REV_HIST, "y_rev", None),
    ("rev", "guid"): ("rev_guided", REV_GUID, "y_beat", "guid_g"),
    ("gm", "hist"): ("gm_all", GM_HIST, "y_gm", None),
    ("gm", "guid"): ("gm_all", GM_GUID, "y_gm", None),
}


def add_pooled_beat(pred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Benchmark G3: guidance x (1 + the market-wide average beat of the training window).

    The skeptic's question: does a model know anything company-specific, or only that semiconductor companies beat
    their own guidance by ~2% on average? mu is the winsorised mean beat of quarters public before the window."""
    masks = samples(panel)
    g0 = pred[(pred["experiment"] == "rev_guided") & (pred["model"] == "G0_guidance_mid")]
    out = []
    for w, g in g0.groupby("window"):
        cut = pd.Period(w, freq="Q").start_time.tz_localize("UTC")
        tr = panel.loc[(panel["release_utc"] < cut) & masks["rev_guided"], "y_beat"]
        mu = tr.clip(*tr.quantile([0.01, 0.99])).mean()
        gg = g.copy()
        gg["model"] = "G3_guidance+pooled_beat"
        gg["yhat"] = gg["guid_g"] + mu
        out.append(gg)
    pred = pred[pred["model"] != "G3_guidance+pooled_beat"]
    return pd.concat([pred] + out, ignore_index=True)


def predict_live(panel: pd.DataFrame, live: pd.DataFrame, model: str, kind: str) -> np.ndarray:
    """Refit `model` (e.g. 'M7_xgboost_guid') on every quarter whose outcome is public now and predict live rows.

    Returns predictions of y_rev (kind='rev') or y_gm (kind='gm')."""
    base, _, fs = model.rpartition("_")
    exp, cols, target, anchor = SPEC[(kind, fs)]
    tr = panel[samples(panel)[exp]]
    est = make_models("full")[base]
    est.fit(design(tr, cols).values, tr[target].values)
    yhat = est.predict(design(live, cols).values)
    if anchor:
        yhat = live[anchor].values + yhat
    return yhat


def _cached_window(panel, masks, q, cache_dir):
    """One walk-forward window, checkpointed so an interrupted run resumes instead of starting over."""
    import pickle
    path = cache_dir / f"{q}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    recs = _window(panel, masks, q)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(recs))
    tmp.replace(path)
    return recs


def run(panel: pd.DataFrame, start: str = START, end: str | None = None, n_jobs: int = 4) -> pd.DataFrame:
    import hashlib
    from .paths import OUT
    masks = samples(panel)
    last = pd.Timestamp(panel["origin_utc"].max()).tz_convert(None).to_period("Q")
    qs = pd.period_range(start, end or last, freq="Q")
    key = hashlib.sha1((MODEL_VERSION + str(pd.util.hash_pandas_object(
        panel.drop(columns=["origin_q"], errors="ignore").astype(str), index=True).sum())).encode()).hexdigest()[:12]
    cache_dir = OUT / "wf_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"walk-forward {qs[0]}..{qs[-1]} ({len(qs)} windows), checkpoints in {cache_dir.name}", flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_cached_window)(panel, masks, q, cache_dir) for q in qs)
    recs = [r for rr in res for r in rr]
    pred = pd.DataFrame(recs, columns=["experiment", "model", "row", "yhat", "y", "window", "n_train"])
    meta = panel[["ticker", "period_end", "origin_utc", "release_utc", "subindustry", "fiscal_qtr", "rev_l1",
                  "gm_l1", "guid_g", "guid_mid", "has_guid", "rev_t", "gm_t"]]
    pred = pred.join(meta, on="row")
    pred["period"] = np.where(pred["origin_utc"] < DEV_END, "dev", "holdout")
    return pred
