"""Pipeline steps wired to the CLI."""
from __future__ import annotations

import pandas as pd

from . import db
from .paths import OUT
from .universe import load_universe


def fetch(refresh_hours: float = 24.0):
    from .edgar import companyfacts, submissions
    from .http import SECClient
    client = SECClient()
    univ = load_universe()
    rows = []
    for _, u in univ.iterrows():
        for cik in u["cik_list"]:
            head, df = submissions(client, cik, refresh_hours)
            js = companyfacts(client, cik, refresh_hours)
            n_facts = 0 if js is None else sum(len(v) for v in js.get("facts", {}).values())
            forms = df["form"].value_counts()
            rows.append({"ticker": u["ticker"], "cik": cik, "edgar_name": head["name"], "sic": head["sic"],
                         "fy_end": head["fiscalYearEnd"], "n_filings": len(df),
                         "n_10q": int(forms.get("10-Q", 0)), "n_10k": int(forms.get("10-K", 0)),
                         "n_8k": int(forms.get("8-K", 0)), "n_concepts": n_facts,
                         "first_filing": df["filingDate"].min(), "last_filing": df["filingDate"].max()})
            print(f"{u['ticker']:9s} {cik:>8d} {head['name'][:40]:40s} sic={head['sic']} filings={len(df):5d} "
                  f"10Q={rows[-1]['n_10q']:3d} concepts={n_facts}")
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / "universe_check.csv", index=False)
    print(f"network={client.n_network} cache={client.n_cache}")


CACHE_ONLY = 1e9   # hours: rebuild from the raw cache without touching the network


def events():
    """Recompute release events from the built filings and quarter calendar (no XBRL replay)."""
    from .events import earnings_events
    filings = pd.read_pickle(OUT / "filings.pkl")
    cal = pd.read_pickle(OUT / "quarters.pkl")
    ev = earnings_events(filings, cal)
    ev.to_pickle(OUT / "earnings_events.pkl")
    db.write(ev, "earnings_events")
    print(ev["release_source"].value_counts().to_string())


def releases(tickers=None):
    from .releases import download
    filings = pd.read_pickle(OUT / "filings.pkl")
    download(filings, tickers)


def guidance(tickers=None):
    from .guidance_build import build_guidance, first_reported_panel
    filings = pd.read_pickle(OUT / "filings.pkl")
    ev = pd.read_pickle(OUT / "earnings_events.pkl")
    vint = pd.read_pickle(OUT / "fin_vintages.pkl")
    cal = pd.read_pickle(OUT / "quarters.pkl")
    if tickers:
        ev = ev[ev["ticker"].isin(tickers)]
    raw, final = build_guidance(filings, ev, vint, cal)
    if tickers:                       # subset runs are for parser checks; never overwrite the full table
        raw.to_pickle(OUT / "guidance_raw_test.pkl")
        final.to_pickle(OUT / "guidance_test.pkl")
    else:
        raw.to_pickle(OUT / "guidance_raw.pkl")
        final.to_pickle(OUT / "guidance.pkl")
        db.write(raw, "guidance_raw")
        db.write(final, "guidance")
    # coverage + QA (realised outcome vs guidance midpoint is for inspection only, never used in parsing)
    panel = first_reported_panel(vint).reset_index()
    rev = final[final["metric"] == "revenue"].merge(
        panel[["ticker", "period_end", "revenue"]].rename(columns={"period_end": "target_period_end"}),
        on=["ticker", "target_period_end"], how="left")
    rev["surprise"] = rev["revenue"] / rev["mid"] - 1
    evn = ev.groupby("ticker").size().rename("events")
    cov = rev.groupby("ticker").agg(n_rev=("mid", "size"), med_surprise=("surprise", "median"),
                                   p_abs_gt10=("surprise", lambda s: float((s.abs() > 0.10).mean())),
                                   first=("target_period_end", "min"), last=("target_period_end", "max"))
    gmc = final[final["metric"].str.startswith("gm_")].pivot_table(index="ticker", columns="metric",
                                                                    values="mid", aggfunc="size")
    cov = pd.concat([evn, cov, gmc], axis=1).fillna({"n_rev": 0})
    cov["rev_coverage"] = cov["n_rev"] / cov["events"]
    pd.set_option("display.width", 220)
    print(cov.round(3).to_string())
    print(f"revenue guidance rows: {len(rev):,}; kinds: {rev['kind'].value_counts().to_dict()}")
    print(f"|surprise| > 10%: {(rev['surprise'].abs() > 0.10).sum()} of {rev['surprise'].notna().sum()}")


def features():
    from .features import build_panel
    vint = pd.read_pickle(OUT / "fin_vintages.pkl")
    cal = pd.read_pickle(OUT / "quarters.pkl")
    ev = pd.read_pickle(OUT / "earnings_events.pkl")
    guid = pd.read_pickle(OUT / "guidance.pkl")
    from .consensus import attach
    panel = attach(build_panel(vint, cal, ev, guid))          # cons_revenue stays empty until licensed data is added
    panel.to_pickle(OUT / "panel.pkl")
    panel.to_csv(OUT / "panel.csv", index=False)
    db.write(panel.drop(columns=["origin_q"]), "feature_panel")
    from .features import build_live_panel
    filings = pd.read_pickle(OUT / "filings.pkl")
    live = build_live_panel(vint, cal, ev, guid, filings)
    live.to_pickle(OUT / "panel_live.pkl")
    print("live targets:", live["status"].value_counts().to_dict() if len(live) else 0)
    yr = pd.to_datetime(panel["origin_utc"]).dt.year
    summ = panel.assign(year=yr).groupby("year").agg(rows=("ticker", "size"), eligible=("eligible", "sum"),
                                                     guided=("has_guid", "sum"),
                                                     gm_gaap_guided=("guid_gm_gaap", lambda s: s.notna().sum()),
                                                     firms=("ticker", "nunique"))
    print(summ.to_string())
    print(f"rows={len(panel):,} eligible={int(panel['eligible'].sum()):,} guided={int(panel['has_guid'].sum()):,}")


def finalize(show: str = "dev"):
    """Merge dev + holdout window predictions, add the pooled-beat benchmark, print metrics for `show` only."""
    from .evaluate import table
    from .walkforward import add_pooled_beat
    panel = pd.read_pickle(OUT / "panel.pkl")
    dev = pd.read_pickle(OUT / "predictions_dev.pkl")
    hold = pd.read_pickle(OUT / "predictions_holdout_raw.pkl")
    assert dev["window"].max() < hold["window"].min(), "dev and holdout windows overlap"
    pred = add_pooled_beat(pd.concat([dev, hold], ignore_index=True), panel)
    assert not pred.duplicated(["experiment", "model", "row"]).any()
    pred.to_pickle(OUT / "predictions_all.pkl")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    for exp, ref, anchor in REPORT_SPECS:
        t = table(pred, exp, show, ref, beat_anchor=anchor)
        print(f"\n=== {exp} | {show} | reference {ref}")
        print(t.round(3).to_string(index=False))


REPORT_SPECS = [("rev_all", "B0_random_walk", False), ("rev_guided", "G0_guidance_mid", True),
                ("rev_guided", "G1_guidance+hist_beat", True), ("rev_guided", "G3_guidance+pooled_beat", True),
                ("gm_all", "B0_random_walk", False), ("gm_guided", "G0_gaap_guidance_mid", False)]


def report():
    """Figures, auto-generated metric tables (reports/tables.md), live forecasts and the dashboard."""
    import numpy as np
    from . import dashboard
    from . import report as R
    from .evaluate import by_group, table
    from .paths import REPORTS
    from .universe import load_universe
    from .walkforward import importance, predict_live
    path = OUT / "predictions_all.pkl"
    pred = pd.read_pickle(path if path.exists() else OUT / "predictions_dev.pkl")
    periods = [p for p in ("dev", "holdout") if p in set(pred["period"])]
    panel = pd.read_pickle(OUT / "panel.pkl")
    live = pd.read_pickle(OUT / "panel_live.pkl")
    T = {}
    for exp, ref, anchor in REPORT_SPECS:
        for per in periods:
            T[(exp, per, ref)] = table(pred, exp, per, ref, beat_anchor=anchor)
    # ---- headline models chosen on the development period only
    dev_rev = T[("rev_all", "dev", "B0_random_walk")]
    dev_guid = T[("rev_guided", "dev", "G0_guidance_mid")]
    dev_gm = T[("gm_all", "dev", "B0_random_walk")]
    best_hist = R.best_dev_model(dev_rev[dev_rev["model"].str.endswith("_hist")], "MAPE_%")
    best_guid = R.best_dev_model(dev_guid[dev_guid["model"].str.endswith("_guid")], "MAPE_%")
    best_gm = R.best_dev_model(dev_gm, "MAE_pp")
    sel = {"rev_hist": best_hist, "rev_guid": best_guid, "gm": best_gm}
    print("dev-selected models:", sel)
    # ---- figures
    F = R.FIG
    for (exp, per, ref), t in T.items():
        if exp == "rev_guided" and not ref.startswith("G0"):
            continue
        metric = "MAPE_%" if exp.startswith("rev") else "MAE_pp"
        hl = {ref: R.S2, sel["rev_guid"] if exp == "rev_guided" else (sel["rev_hist"] if exp == "rev_all" else sel["gm"]): R.S1}
        titles = {"rev_all": "Revenue, all eligible quarters", "rev_guided": "Revenue, quarters with guidance",
                  "gm_all": "Gross margin, all eligible quarters", "gm_guided": "Gross margin, quarters with GAAP guidance"}
        span = "development 2013-2020" if per == "dev" else "holdout 2021-2026"
        R.bar_chart(t, metric, f"{titles[exp]} · {span} (orange = reference, blue = dev-selected model)",
                    "Mean absolute % error" if metric == "MAPE_%" else "Mean absolute error (pp)",
                    F / f"bars_{exp}_{per}.png", hl)
    yr = R.yearly_error(pred, "rev_guided", ["B0_random_walk", "G0_guidance_mid", sel["rev_guid"]], True)
    R.line_chart(yr, "Revenue: mean absolute % error by year (guided quarters)", "MAPE (%)", F / "rev_error_by_year.png",
                 [R.DEEMPH, R.S2, R.S1])
    yg = R.yearly_error(pred, "gm_all", ["B0_random_walk", "B2_hist_avg_level", sel["gm"]], False)
    R.line_chart(yg, "Gross margin: mean absolute error by year", "MAE (pp)", F / "gm_error_by_year.png",
                 [R.DEEMPH, R.S2, R.S1])
    R.beat_histogram(panel.loc[panel["eligible"] & panel["has_guid"], "y_beat"], F / "beat_histogram.png")
    for per in periods:
        R.calibration(pred, "rev_guided", sel["rev_guid"], F / f"calibration_{per}.png", per)
    for tk in ("NVDA", "TXN", "LRCX"):
        R.company_chart(pred, tk, sel["rev_guid"], F / f"{tk.lower()}_beats.png")
    # ---- tables.md
    lines = ["# Auto-generated metric tables", "",
             f"Headline models (selected on 2013-2020 development data): {sel}", ""]
    for (exp, per, ref), t in T.items():
        cols = ["model", "n"] + (["MAPE_%", "MdAPE_%", "RMSE_log_%"] if exp.startswith("rev") else
                                 ["MAE_pp", "MdAE_pp", "RMSE_pp"]) + ["dir_acc_%"]
        cols += ["beat_dir_acc_%", "IC_beat", "IC_beat_t"] if exp == "rev_guided" else ["IC", "IC_t"]
        cols += ["MAE_vs_ref_%", "DM_t_abs", "DM_t_sq"]
        lines += [f"## {exp} · {per} · reference {R.label(ref)}", "", R.md_table(t, [c for c in cols if c in t], {}), ""]
        t.to_csv(OUT / f"metrics_{exp}_{per}_vs_{ref.split('_')[0]}.csv", index=False)
    for per in periods:
        g = by_group(pred, "rev_guided", per, ["G0_guidance_mid", "G1_guidance+hist_beat", sel["rev_guid"]],
                     "subindustry").round(2)
        lines += [f"## MAPE by sub-industry · guided revenue · {per}", "", "```", g.to_string(), "```", ""]
    imp = importance(panel, "2021-01-01", "rev_guided")
    imp.to_csv(OUT / "importance_rev_guided_2020.csv", index=False)
    lines += ["## Feature importance, guided revenue (XGBoost gain, fit on data public before 2021-01-01)", "",
              imp.head(15).round(4).to_string(index=False), ""]
    (REPORTS / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    # ---- live forecasts (models refit on every quarter whose outcome is public now)
    lv = live[live["eligible"] | live["status"].eq("awaiting_release")].copy()
    if len(lv):
        lv["y_rev_hist"] = predict_live(panel, lv, sel["rev_hist"], "rev")
        lv["y_rev_guid"] = np.where(lv["has_guid"], predict_live(panel, lv.assign(guid_g=lv["guid_g"].fillna(0)),
                                                                   sel["rev_guid"], "rev"), np.nan)
        lv["y_gm"] = predict_live(panel, lv, sel["gm"], "gm")
        lv["model_rev"] = lv["rev_l1"] * np.exp(np.where(lv["has_guid"], lv["y_rev_guid"], lv["y_rev_hist"]))
        lv["pred_beat_pct"] = np.where(lv["has_guid"], 100 * (lv["model_rev"] / lv["guid_mid"] - 1), np.nan)
        lv["model_gm"] = lv["gm_l1"] + lv["y_gm"]
        lv.to_csv(OUT / "live_forecasts.csv", index=False)
    # ---- dashboard
    univ = load_universe().set_index("ticker")
    status_label = {"awaiting_release": "Awaiting release", "in_progress": "Quarter in progress",
                    "reported_pending_10q": "Reported, 10-Q pending"}
    live_rows = []
    for _, r in lv.sort_values(["status", "pred_beat_pct"], ascending=[True, False]).iterrows():
        live_rows.append({"ticker": r["ticker"], "status_label": status_label.get(r["status"], r["status"]),
                          "est_release": str(r["est_release_date"]), "guid_mid": r["guid_mid"],
                          "model_rev": r["model_rev"], "pred_beat_pct": r["pred_beat_pct"],
                          "hist_beat_pct": 100 * (np.exp(r["beat_mean8"]) - 1) if pd.notna(r["beat_mean8"]) else None,
                          "model_gm": r["model_gm"], "guid_gm": r["guid_gm_gaap"]})
    board, labels, notes = {}, {}, {}
    names = {"rev_all": "Revenue, all eligible quarters (history-only information)",
             "rev_guided": "Revenue, quarters with guidance (reference = guidance midpoint)",
             "gm_all": "Gross margin (GAAP), all eligible quarters", "gm_guided": "Gross margin, quarters with GAAP guidance"}
    for (exp, per, ref), t in T.items():
        if exp == "rev_guided" and not ref.startswith("G0"):
            continue
        tt = t.copy()
        tt["label"] = tt["model"].map(R.label)
        board.setdefault(exp, {})[per] = tt.to_dict("records")
        labels[exp] = names[exp]
        notes[exp] = f"Reference model: {R.label(ref)}. Negative 'vs reference' = lower error. DM t < -2: significantly better."
    comp = {}
    pg = pred[(pred["experiment"] == "rev_guided") & pred["model"].isin(["G0_guidance_mid", sel["rev_guid"]])]
    pa = pred[(pred["experiment"] == "rev_all") & (pred["model"] == sel["rev_hist"])]
    pm = pred[(pred["experiment"] == "gm_all") & (pred["model"] == sel["gm"])]
    for tk, grp in panel[panel["eligible"]].groupby("ticker"):
        qs = []
        for _, r in grp.sort_values("period_end").iterrows():
            rh = pg[(pg["ticker"] == tk) & (pg["period_end"] == r["period_end"]) & (pg["model"] == sel["rev_guid"])]
            if rh.empty:
                rh = pa[(pa["ticker"] == tk) & (pa["period_end"] == r["period_end"])]
            gh = pm[(pm["ticker"] == tk) & (pm["period_end"] == r["period_end"])]
            qs.append({"pe": str(r["period_end"])[:10], "rev": r["rev_t"], "guid": r["guid_mid"],
                       "rev_hat": float(r["rev_l1"] * np.exp(rh["yhat"].iloc[0])) if len(rh) else None,
                       "gm": r["gm_t"], "gm_guid": r["guid_gm_gaap"],
                       "gm_hat": float(r["gm_l1"] + gh["yhat"].iloc[0]) if len(gh) else None})
        comp[tk] = {"name": univ.loc[tk, "name"] if tk in univ.index else tk,
                    "subindustry": univ.loc[tk, "subindustry"] if tk in univ.index else "",
                    "status": univ.loc[tk, "status"] if tk in univ.index else "",
                    "revModelLabel": "Model forecast", "gmModelLabel": f"Model ({R.label(sel['gm'])})",
                    "quarters": qs}
    ev = pd.read_pickle(OUT / "earnings_events.pkl")
    active = set(univ.index[univ["status"] == "active"])
    recent = ev[ev["ticker"].isin(active)].sort_values("release_utc").groupby("ticker").tail(1) \
        .sort_values("release_utc", ascending=False)
    cal_rows = []
    for _, r in recent.iterrows():
        prow = panel[(panel["ticker"] == r["ticker"]) & (panel["period_end"] == r["period_end"])]
        beat = 100 * (np.exp(prow["y_beat"].iloc[0]) - 1) if len(prow) and pd.notna(prow["y_beat"].iloc[0]) else None
        cal_rows.append({"ticker": r["ticker"], "pe": str(r["period_end"])[:10],
                         "release_et": str(r["release_et"])[:16], "session": r["session"],
                         "days": int(r["days_after_period"]), "rev": prow["rev_t"].iloc[0] if len(prow) else None,
                         "beat": beat})
    s = panel[panel["eligible"] & panel["has_guid"] & panel["y_beat"].notna()]
    hold = T.get(("rev_guided", "holdout", "G0_guidance_mid"))
    k_best = hold[hold["model"] == sel["rev_guid"]]["MAPE_%"].iloc[0] if hold is not None else None
    k_g0 = hold[hold["model"] == "G0_guidance_mid"]["MAPE_%"].iloc[0] if hold is not None else None
    kpis = [{"label": "Companies in sample", "value": str(panel.loc[panel["eligible"], "ticker"].nunique()),
             "detail": "US semis & semicap incl. acquired, 2010-2026"},
            {"label": "Quarters with guidance", "value": f"{len(s):,}", "detail": "eligible, parsed from 8-K releases"},
            {"label": "Beat the guidance midpoint", "value": f"{100 * (s['y_beat'] > 0).mean():.0f}%",
             "detail": f"median beat {100 * (np.exp(s['y_beat'].median()) - 1):+.1f}%"}]
    if k_best is not None:
        kpis.append({"label": "Holdout revenue MAPE", "value": f"{k_best:.2f}%",
                     "detail": f"model; guidance midpoint {k_g0:.2f}%"})
    data = {"meta": {"subtitle": f"US semiconductors · point-in-time SEC data · built {pd.Timestamp.now():%Y-%m-%d %H:%M}",
                     "footer": "Sources: SEC EDGAR XBRL companyfacts, submissions and 8-K Item 2.02 exhibits. "
                               "No analyst consensus (not licensed). Research output, not investment advice."},
            "kpis": kpis, "live": live_rows, "board": board, "boardLabels": labels, "boardNotes": notes,
            "companies": comp, "calendar": cal_rows}
    import json
    (OUT / "dashboard_data.json").write_text(json.dumps(dashboard._clean(data)), encoding="utf-8")
    print("dashboard:", dashboard.build(data))


def dashboard_only():
    """Rebuild reports/dashboard.html from out/dashboard_data.json (after template changes)."""
    import json
    from . import dashboard
    data = json.loads((OUT / "dashboard_data.json").read_text(encoding="utf-8"))
    print("dashboard:", dashboard.build(data))


def backtest(period: str = "dev"):
    """period=dev runs windows through 2020Q4 only; holdout/all run every window (final, run once)."""
    from .evaluate import table
    from .walkforward import run
    panel = pd.read_pickle(OUT / "panel.pkl")
    end = "2020Q4" if period == "dev" else None
    pred = run(panel, end=end)
    pred.to_pickle(OUT / f"predictions_{'dev' if period == 'dev' else 'all'}.pkl")
    periods = ["dev"] if period == "dev" else (["holdout"] if period == "holdout" else ["dev", "holdout"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for per in periods:
        for exp, ref, anchor in REPORT_SPECS:
            t = table(pred, exp, per, ref, beat_anchor=anchor)
            t.to_csv(OUT / f"metrics_{exp}_{per}_vs_{ref.split('_')[0]}.csv", index=False)
            print(f"\n=== {exp} | {per} | reference {ref}")
            print(t.round(3).to_string(index=False))


def build():
    from .edgar import build_facts, build_filings
    from .events import earnings_events
    from .http import SECClient
    from .items import ALL_CONCEPTS
    from .pit import build_vintages, quarter_calendar
    client = SECClient()
    univ = load_universe()
    filings, heads = build_filings(client, univ, CACHE_ONLY)
    facts = build_facts(client, univ, ALL_CONCEPTS, CACHE_ONLY)
    print(f"filings={len(filings):,} facts={len(facts):,}")
    vint = build_vintages(facts, filings)
    cal = quarter_calendar(vint)
    ev = earnings_events(filings, cal)
    con = db.connect()
    db.write(filings.drop(columns=["acceptance_raw"]), "filings", con)
    db.write(heads, "entities", con)
    db.write(facts, "xbrl_facts", con)
    db.write(vint, "fin_vintages", con)
    db.write(cal, "quarters", con)
    db.write(ev, "earnings_events", con)
    con.close()
    vint.to_pickle(OUT / "fin_vintages.pkl")
    cal.to_pickle(OUT / "quarters.pkl")
    ev.to_pickle(OUT / "earnings_events.pkl")
    filings.to_pickle(OUT / "filings.pkl")
    print(f"vintages={len(vint):,} quarters={len(cal):,} events={len(ev):,}")
    print(ev["release_source"].value_counts().to_string())
    print(ev["session"].value_counts().to_string())
