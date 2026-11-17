"""Robustness of the headline comparisons (dev-selected models vs their benchmarks), dev and holdout.

Checks: per-company win rate, excluding M&A-affected quarters, excluding NVIDIA, by sub-industry, by year,
and a model-vs-Bayesian-beat DM test. Writes out/robustness.csv and prints a compact summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundalpha.evaluate import common_sample, nw_tstat  # noqa: E402
from fundalpha.paths import OUT  # noqa: E402

pred = pd.read_pickle(OUT / "predictions_all.pkl")
panel = pd.read_pickle(OUT / "panel.pkl")
flags = panel[["goodwill_jump", "y_rev"]].rename(columns={"y_rev": "y_rev_panel"})
pred = pred.join(flags, on="row")
COMPARISONS = [  # (experiment, model, reference, revenue?)
    ("rev_all", "M7_xgboost_hist", "B0_random_walk", True),
    ("rev_all", "M7_xgboost_hist", "L1_linear_baseline", True),
    ("rev_guided", "M5_random_forest_guid", "G0_guidance_mid", True),
    ("rev_guided", "M5_random_forest_guid", "G3_guidance+pooled_beat", True),
    ("rev_guided", "M5_random_forest_guid", "G2_guidance+EB_beat", True),
    ("gm_all", "M3_lasso_guid", "B0_random_walk", False),
    ("gm_guided", "M3_lasso_guid", "G0_gaap_guidance_mid", False),
]


def loss(df, revenue):
    e = df["yhat"] - df["y"]
    return (np.exp(e) - 1).abs() * 100 if revenue else e.abs()


rows = []
for exp, m, ref, rev in COMPARISONS:
    for per in ("dev", "holdout"):
        p = pred[(pred["experiment"] == exp) & (pred["period"] == per)]
        p = common_sample(p, [m, ref])
        a = p[p["model"] == m].set_index("row")
        b = p[p["model"] == ref].set_index("row")
        d = pd.DataFrame({"lm": loss(a, rev), "lr": loss(b, rev), "ticker": a["ticker"], "window": a["window"],
                          "sub": a["subindustry"], "gw": a["goodwill_jump"], "y": a["y"], "yr": pd.to_datetime(a["origin_utc"]).dt.year})

        def summ(x, tag):
            if len(x) < 20:
                return
            dq = (x["lm"] - x["lr"]).groupby(x["window"]).mean()
            firm = x.groupby("ticker")[["lm", "lr"]].mean()
            rows.append({"experiment": exp, "model": m, "reference": ref, "period": per, "subset": tag, "n": len(x),
                         "model_err": x["lm"].mean(), "ref_err": x["lr"].mean(),
                         "rel_change_%": 100 * (x["lm"].mean() / x["lr"].mean() - 1), "DM_t": nw_tstat(dq.values),
                         "firm_win_rate_%": 100 * float((firm["lm"] < firm["lr"]).mean()), "n_firms": len(firm),
                         "quarter_win_rate_%": 100 * float((dq < 0).mean())})

        summ(d, "all")
        ma = (d["gw"].abs() > 0.2) | (d["y"].abs() > 0.4 if rev else False)
        summ(d[~ma], "excl. M&A quarters (goodwill jump >20% or |seq growth| >40%)")
        summ(d[d["ticker"] != "NVDA"], "excl. NVDA")
        for s, g in d.groupby("sub"):
            summ(g, f"sub-industry: {s}")
        for y, g in d.groupby("yr"):
            summ(g, f"year {y}")

out = pd.DataFrame(rows)
out.to_csv(OUT / "robustness.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
show = out[~out["subset"].str.startswith("year")]
print(show.round(2).to_string(index=False))
yr = out[out["subset"].str.startswith("year") & (out["reference"].isin(["G0_guidance_mid", "B0_random_walk"]))]
print(yr.pivot_table(index="subset", columns=["experiment", "reference"], values="rel_change_%").round(1).to_string())
