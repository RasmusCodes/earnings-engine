"""Forecast-origin feature panel: one row per (company, target fiscal quarter) with an earnings release.

Origin tau = last regular-session close before the release (events.py). Every feature is computed from data
accepted on EDGAR before tau:
  * own financials : latest XBRL vintage of each quarter with avail_utc < tau (restatements included as known)
  * guidance       : latest outlook for the target quarter issued before tau
  * own track record (beats) : quarters whose release was before tau
  * peer season    : other companies' quarters released in [tau - 75d, tau) whose period ends within 50 days of
                     the target quarter (same reporting season); a peer's revenue counts as known at its release
Targets (never used as features):
  y_rev  = log(first-reported revenue_t / revenue_{t-1} as known at tau)
  y_beat = log(first-reported revenue_t / guidance midpoint_t)
  y_gm   = first-reported gross margin_t - gross margin_{t-1} as known at tau   (percentage points)

The fiscal calendar (length of the target quarter, 13 vs 14 weeks) is treated as known at tau: companies publish
their 52/53-week calendars in advance and the quarter has ended when the forecast is made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .guidance_build import first_reported_panel
from .pit import gross_margin
from .universe import load_universe

ITEMS = ["revenue", "gross_profit", "cogs", "inventory", "receivables", "deferred_revenue", "goodwill",
         "operating_income", "rnd", "sga"]
ELIGIBLE_TTM_REV = 500e6          # point-in-time size screen: trailing-4Q revenue known at the origin
PEER_LOOKBACK_DAYS = 75
PEER_PERIOD_TOL_DAYS = 50


def _snapshot(vt: pd.DataFrame, tau: pd.Timestamp, qe: list) -> pd.DataFrame:
    v = vt[vt["avail_utc"] < tau]
    v = v.sort_values("avail_utc").drop_duplicates(["item", "period_end"], keep="last")
    w = v.pivot(index="period_end", columns="item", values="value").reindex(qe)
    for c in ITEMS:
        if c not in w:
            w[c] = np.nan
    w["gm"] = gross_margin(w["revenue"], w["gross_profit"], w["cogs"]).values
    return w


def _lg(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))
    return float(out) if np.isfinite(out) else np.nan


CLEAN_REV_BASES = ("total", "seq", "yoy")


def build_panel(vint: pd.DataFrame, cal: pd.DataFrame, ev: pd.DataFrame, guid: pd.DataFrame,
                targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for `targets` (default: every released quarter in `ev`). `ev` always supplies the realised
    outcomes used for track-record and peer features."""
    univ = load_universe().set_index("ticker")
    fr = first_reported_panel(vint)
    gfinal = guid.copy()
    # company-total revenue guidance only (segment-like or discontinued-ops-inclusive outlooks are not comparable)
    grev = gfinal[(gfinal["metric"] == "revenue") & gfinal["basis"].isin(CLEAN_REV_BASES)] \
        .set_index(["ticker", "target_period_end"])
    ggaap = gfinal[gfinal["metric"] == "gm_gaap"].set_index(["ticker", "target_period_end"])
    gng = gfinal[gfinal["metric"] == "gm_nongaap"].set_index(["ticker", "target_period_end"])
    gun = gfinal[gfinal["metric"] == "gm_unspecified"].set_index(["ticker", "target_period_end"])

    # event-level outcomes used for track-record and peer features (known at each release)
    evo = ev[["ticker", "period_end", "release_utc", "origin_utc"]].copy()
    evo["A"] = [fr["revenue"].get((t, p), np.nan) for t, p in zip(evo["ticker"], evo["period_end"])]
    evo["gm_A"] = [fr["gm"].get((t, p), np.nan) for t, p in zip(evo["ticker"], evo["period_end"])]
    evo["G"] = [grev["mid"].get((t, p), np.nan) for t, p in zip(evo["ticker"], evo["period_end"])]
    evo["G_gm_gaap"] = [ggaap["mid"].get((t, p), np.nan) for t, p in zip(evo["ticker"], evo["period_end"])]
    evo["beat"] = np.log(evo["A"] / evo["G"])
    evo["gm_beat_gaap"] = evo["gm_A"] - evo["G_gm_gaap"]
    evo = evo.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    yoy, yoy_prev, seq = [], [], []
    for tk, g in evo.groupby("ticker", sort=False):
        qe = list(cal[cal["ticker"] == tk].sort_values("period_end")["period_end"])
        pos = {p: i for i, p in enumerate(qe)}
        for p in g["period_end"]:
            i = pos.get(p)
            def A(k):
                if i is None or i - k < 0:
                    return np.nan
                return fr["revenue"].get((tk, qe[i - k]), np.nan)
            yoy.append(_lg(A(0), A(4)))
            yoy_prev.append(_lg(A(1), A(5)))
            seq.append(_lg(A(0), A(1)))
    evo["g_yoy"], evo["g_yoy_prev"], evo["g_seq"] = yoy, yoy_prev, seq
    evo["subindustry"] = evo["ticker"].map(univ["subindustry"])

    rows = []
    for tk, e in (ev if targets is None else targets).groupby("ticker"):
        vt = vint[(vint["ticker"] == tk) & vint["item"].isin(ITEMS)]
        c = cal[cal["ticker"] == tk].sort_values("period_end").reset_index(drop=True)
        qe = list(c["period_end"])
        days = dict(zip(c["period_end"], c["period_days"]))
        fq = dict(zip(c["period_end"], c["fiscal_qtr"]))
        own = evo[evo["ticker"] == tk]
        for _, r in e.sort_values("period_end").iterrows():
            t_end = r["period_end"]
            if t_end not in qe:
                continue
            j = qe.index(t_end)
            if j < 2:
                continue
            tau = r["origin_utc"]
            w = _snapshot(vt, tau, qe)
            R = w["revenue"].values
            GM = w["gm"].values

            def lag(arr, k):
                return arr[j - k] if j - k >= 0 else np.nan

            R1, R2, R3, R4, R5, R6 = (lag(R, k) for k in range(1, 7))
            gl = [_lg(lag(R, k), lag(R, k + 1)) for k in range(1, 9)]
            seas = [_lg(lag(R, 4 * k), lag(R, 4 * k + 1)) for k in (1, 2, 3)]
            gm_seas = [lag(GM, 4 * k) - lag(GM, 4 * k + 1) for k in (1, 2, 3)]
            A_t = fr["revenue"].get((tk, t_end), np.nan)
            gm_t = fr["gm"].get((tk, t_end), np.nan)
            ttm = np.nansum([R1, R2, R3, R4]) if np.isfinite([R1, R2, R3, R4]).all() else np.nan
            row = {
                "ticker": tk, "period_end": t_end, "fiscal_qtr": fq.get(t_end), "origin_utc": tau,
                "release_utc": r["release_utc"], "session": r["session"],
                "subindustry": univ.loc[tk, "subindustry"] if tk in univ.index else "",
                "rev_t": A_t, "rev_l1": R1, "gm_t": gm_t, "gm_l1": lag(GM, 1),
                "ttm_rev": ttm, "eligible": bool(np.isfinite(ttm) and ttm >= ELIGIBLE_TTM_REV),
                # ---------------- own revenue history
                "log_rev_l1": np.log(R1) if R1 and R1 > 0 else np.nan,
                "g_seq1": gl[0], "g_seq2": gl[1], "g_seq3": gl[2],
                "g_yoy1": _lg(R1, R5), "g_yoy2": _lg(R2, R6),
                "seas_g1": seas[0], "seas_g_avg": np.nanmean(seas) if np.isfinite(seas).any() else np.nan,
                "g_mean4": np.nanmean(gl[:4]) if np.isfinite(gl[:4]).sum() >= 3 else np.nan,
                "g_mean8": np.nanmean(gl) if np.isfinite(gl).sum() >= 6 else np.nan,
                "g_std8": np.nanstd(gl) if np.isfinite(gl).sum() >= 6 else np.nan,
                "days_ratio": _lg(days.get(t_end), days.get(qe[j - 1])),
                "days_ratio_l4": _lg(days.get(qe[j - 4]), days.get(qe[j - 5])) if j >= 5 else np.nan,
                # ---------------- own gross margin history
                "gm_l2": lag(GM, 2), "gm_l4": lag(GM, 4), "gm_l5": lag(GM, 5),
                "d_gm1": lag(GM, 1) - lag(GM, 2),
                "gm_mean4": np.nanmean([lag(GM, k) for k in range(1, 5)]),
                "gm_mean8": np.nanmean([lag(GM, k) for k in range(1, 9)]),
                "gm_seas1": gm_seas[0], "gm_seas_avg": np.nanmean(gm_seas) if np.isfinite(gm_seas).any() else np.nan,
                # ---------------- balance sheet (quarter t-1)
                "inv_to_rev": lag(w["inventory"].values, 1) / R1 if R1 else np.nan,
                "inv_to_rev_chg": (lag(w["inventory"].values, 1) / R1 - lag(w["inventory"].values, 5) / R5)
                if R1 and R5 else np.nan,
                "dso": 91 * lag(w["receivables"].values, 1) / R1 if R1 else np.nan,
                "defrev_g": _lg(lag(w["deferred_revenue"].values, 1), lag(w["deferred_revenue"].values, 2)),
                "goodwill_jump": _lg(lag(w["goodwill"].values, 1), lag(w["goodwill"].values, 2)),
                "opm_l1": 100 * lag(w["operating_income"].values, 1) / R1 if R1 else np.nan,
            }
            # ---------------- guidance for the target quarter (issued before tau by construction)
            key = (tk, t_end)
            if key in grev.index:
                gr = grev.loc[key]
                ok = pd.notna(gr["mid"]) and gr["issued_utc"] < tau
                row.update({"has_guid": ok, "guid_mid": gr["mid"] if ok else np.nan,
                            "guid_g": _lg(gr["mid"], R1) if ok else np.nan,
                            "guid_width": (gr["high"] - gr["low"]) / gr["mid"] if ok and gr["mid"] else np.nan,
                            "guid_is_point": ok and gr["kind"] == "point",
                            "guid_is_pct": ok and gr["kind"] in ("seq_pct", "yoy_pct"),
                            "guid_is_update": ok and bool(gr.get("is_update", False))})
            else:
                row.update({"has_guid": False, "guid_mid": np.nan, "guid_g": np.nan, "guid_width": np.nan,
                            "guid_is_point": False, "guid_is_pct": False, "guid_is_update": False})
            for name, tab in (("gaap", ggaap), ("nongaap", gng), ("unspec", gun)):
                if key in tab.index and tab.loc[key, "issued_utc"] < tau:
                    row[f"guid_gm_{name}"] = tab.loc[key, "mid"]
                else:
                    row[f"guid_gm_{name}"] = np.nan
            prev_key = (tk, qe[j - 1])
            # change in guided GM (same basis) from the previous quarter's guidance, applied to last actual GM
            dg = np.nan
            for tab in (ggaap, gng, gun):
                if key in tab.index and prev_key in tab.index and tab.loc[key, "issued_utc"] < tau:
                    dg = tab.loc[key, "mid"] - tab.loc[prev_key, "mid"]
                    break
            row["guid_gm_delta"] = dg
            row["guid_gm_gaap_d"] = row["guid_gm_gaap"] - row["gm_l1"] if pd.notna(row["guid_gm_gaap"]) else np.nan
            # ---------------- own track record (quarters already released before tau)
            hist = own[(own["release_utc"] < tau) & (own["period_end"] < t_end)].sort_values("period_end").tail(8)
            b = hist["beat"].dropna()
            row.update({"beat_n": len(b), "beat_mean8": b.mean() if len(b) >= 2 else np.nan,
                        "beat_med8": b.median() if len(b) >= 2 else np.nan,
                        "beat_last": b.iloc[-1] if len(b) else np.nan,
                        "beat_std8": b.std() if len(b) >= 3 else np.nan})
            gb = hist["gm_beat_gaap"].dropna()
            row["gm_beat_gaap_mean8"] = gb.mean() if len(gb) >= 2 else np.nan
            # ---------------- peer season (other companies already released this season)
            peers = evo[(evo["ticker"] != tk) & (evo["release_utc"] < tau) &
                        (evo["release_utc"] >= tau - pd.Timedelta(days=PEER_LOOKBACK_DAYS)) &
                        ((evo["period_end"] - t_end).abs() <= pd.Timedelta(days=PEER_PERIOD_TOL_DAYS))]
            grp = peers[peers["subindustry"] == row["subindustry"]]
            row.update({
                "peer_n": len(peers), "peer_beat_mean": peers["beat"].mean() if peers["beat"].notna().sum() >= 2 else np.nan,
                "peer_yoy_med": peers["g_yoy"].median() if len(peers) >= 2 else np.nan,
                "peer_yoy_accel": (peers["g_yoy"] - peers["g_yoy_prev"]).median() if len(peers) >= 2 else np.nan,
                "peer_seq_med": peers["g_seq"].median() if len(peers) >= 2 else np.nan,
                "grp_n": len(grp), "grp_beat_mean": grp["beat"].mean() if grp["beat"].notna().sum() >= 1 else np.nan,
                "grp_yoy_accel": (grp["g_yoy"] - grp["g_yoy_prev"]).median() if len(grp) >= 1 else np.nan,
                "peer_gm_beat_gaap": peers["gm_beat_gaap"].mean() if peers["gm_beat_gaap"].notna().sum() >= 2 else np.nan,
            })
            # ---------------- targets
            row["y_rev"] = _lg(A_t, R1)
            row["y_beat"] = _lg(A_t, row["guid_mid"]) if row["has_guid"] else np.nan
            row["y_gm"] = gm_t - row["gm_l1"] if pd.notna(gm_t) and pd.notna(row["gm_l1"]) else np.nan
            rows.append(row)
    panel = pd.DataFrame(rows)
    # ---------------- industry indicator: hyperscaler capex (latest quarter all four had filed before tau)
    from .industry import capex_feature
    from .paths import OUT
    cv_path = OUT / "hyperscaler_capex_vintages.pkl"
    if cv_path.exists():
        cf = capex_feature(pd.read_pickle(cv_path), panel["origin_utc"].drop_duplicates())
        panel = panel.merge(cf, on="origin_utc", how="left")
    else:
        panel["hyper_capex_yoy"] = np.nan
        panel["hyper_capex_accel"] = np.nan
    panel["origin_q"] = pd.to_datetime(panel["origin_utc"]).dt.tz_convert(None).dt.to_period("Q")
    for c in ("has_guid", "guid_is_point", "guid_is_pct", "guid_is_update"):
        panel[c] = panel[c].astype(bool)
    return panel.sort_values(["origin_utc", "ticker"]).reset_index(drop=True)


EST_QUARTER_DAYS = 91


def live_targets(cal: pd.DataFrame, ev: pd.DataFrame, filings: pd.DataFrame, asof: pd.Timestamp):
    """The next, not-yet-in-XBRL quarter of every active company, as of `asof` (UTC).

    status: awaiting_release (quarter ended, results not out: the forecast that matters before a print),
            in_progress (quarter still running), reported_pending_10q (results released, XBRL not filed yet)."""
    univ = load_universe()
    active = set(univ.loc[univ["status"] == "active", "ticker"])
    is202 = filings["items"].fillna("").apply(lambda s: any(x.strip() == "2.02" for x in s.split(",")))
    rel = filings[(filings["form"] == "8-K") & is202]
    tgt, cal_add = [], []
    for tk in sorted(active):
        c = cal[cal["ticker"] == tk].sort_values("period_end")
        e = ev[ev["ticker"] == tk].sort_values("period_end")
        if c.empty or e.empty or c["period_end"].iloc[-1] != e["period_end"].iloc[-1]:
            continue
        last = c.iloc[-1]
        est_end = last["period_end"] + pd.Timedelta(days=EST_QUARTER_DAYS)
        est_end_utc = (est_end + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
        later = rel[(rel["ticker"] == tk) & (rel["acceptance_utc"] > est_end_utc)]
        if est_end_utc > asof:
            status = "in_progress"
        elif len(later):
            status = "reported_pending_10q"
        else:
            status = "awaiting_release"
        fq_next = int(last["fiscal_qtr"]) % 4 + 1 if pd.notna(last["fiscal_qtr"]) else np.nan
        # expected release: days-after-quarter-end of the same fiscal quarter in the last three years
        # (fourth quarters run later: annual audit), falling back to the recent median
        efq = e.merge(c[["period_end", "fiscal_qtr"]], on="period_end", how="left")
        same = efq[efq["fiscal_qtr"] == fq_next].tail(3)
        dap = float(same["days_after_period"].median()) if len(same) else float(e["days_after_period"].tail(8).median())
        cal_add.append({"ticker": tk, "period_end": est_end, "period_start": last["period_end"] + pd.Timedelta(days=1),
                        "fy_start": pd.NaT, "period_days": EST_QUARTER_DAYS, "fiscal_qtr": fq_next,
                        "fiscal_year": np.nan, "fiscal_label": "next (est.)"})
        tgt.append({"ticker": tk, "period_end": est_end, "release_utc": pd.NaT, "origin_utc": asof,
                    "session": status, "est_release_date": (est_end + pd.Timedelta(days=dap)).date(),
                    "last_release_et": e["release_et"].iloc[-1]})
    return pd.DataFrame(tgt), pd.DataFrame(cal_add)


def build_live_panel(vint, cal, ev, guid, filings, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    asof = asof or pd.Timestamp.now(tz="UTC")
    tgt, cal_add = live_targets(cal, ev, filings, asof)
    if tgt.empty:
        return tgt
    cal2 = pd.concat([cal, cal_add], ignore_index=True)
    live = build_panel(vint, cal2, ev, guid, targets=tgt)
    live = live.merge(tgt[["ticker", "est_release_date", "last_release_et"]], on="ticker", how="left")
    live["status"] = live["session"]
    return live
