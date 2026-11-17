"""Assemble point-in-time guidance per target quarter from parsed releases.

For reported quarter t (release_t), every 8-K Item 2.02 accepted in [release_t, release_{t+1}) may carry outlook for
quarter t+1: the regular release plus any mid-quarter update. When release_t looks like a pre-announcement (a second
Item 2.02 followed before the 10-Q), the pre-announcement document is skipped because its "expected revenue" refers to
quarter t itself. The guidance used for a target quarter is the latest one issued before the target's forecast
origin (last close before its release).

Live quarter: for a company's latest reported quarter the next quarter is not in the XBRL calendar yet, so its period
end is estimated as t + 91 days (`target_estimated`), and only documents accepted before that estimated end count as
guidance (anything later is the next results release, not an outlook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .guidance import PARSER_VERSION, choose_gm, choose_revenue, extract
from .pit import first_reported, gross_margin
from .releases import load_release, release_filings

EST_QUARTER_DAYS = 91


def first_reported_panel(vint: pd.DataFrame) -> pd.DataFrame:
    fr = first_reported(vint)
    p = fr.pivot_table(index=["ticker", "period_end"], columns="item", values="value", aggfunc="first")
    p["gm"] = gross_margin(p["revenue"], p.get("gross_profit"), p.get("cogs")).values
    return p


def _ticker_guidance(tk: str, e: pd.DataFrame, r: pd.DataFrame, qe: list, rev: dict, gm: dict,
                     now: pd.Timestamp) -> list[dict]:
    out = []
    e = e.sort_values("period_end").reset_index(drop=True)
    ev_by_pe = {pe: row for pe, row in zip(e["period_end"], e.to_dict("records"))}
    for i, row in e.iterrows():
        t_end = row["period_end"]
        j = qe.index(t_end) if t_end in qe else None
        target = qe[j + 1] if j is not None and j + 1 < len(qe) else pd.NaT
        estimated = False
        if pd.isna(target) and i == len(e) - 1:
            target = t_end + pd.Timedelta(days=EST_QUARTER_DAYS)
            estimated = True
        if pd.isna(target):
            continue
        yago = qe[j - 3] if j is not None and j - 3 >= 0 else pd.NaT
        start = row["release_utc"] - pd.Timedelta(minutes=1)
        if not estimated and target in ev_by_pe:
            end = ev_by_pe[target]["release_utc"]
        elif not estimated and i + 1 < len(e):
            end = e.loc[i + 1, "release_utc"]
        else:
            end = min(now, (target + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC"))
        docs = r[(r["acceptance_utc"] >= start) & (r["acceptance_utc"] < end)]
        if row["possible_preannouncement"] and len(docs) > 1:
            docs = docs[docs["accn"] != row["release_accn"]]
        base = rev.get(t_end, np.nan)
        base = None if pd.isna(base) else float(base)
        pgm = gm.get(t_end, np.nan)
        pgm = None if pd.isna(pgm) else float(pgm)
        base_yoy = rev.get(yago, np.nan) if pd.notna(yago) else np.nan
        for k, (_, d) in enumerate(docs.iterrows()):
            rec = load_release(d["accn"])
            if rec is None:
                continue
            revs, gms = extract(rec["docs"], base, pgm)
            common = {"ticker": tk, "reported_period_end": t_end, "target_period_end": target,
                      "target_estimated": estimated, "issued_accn": d["accn"], "issued_utc": d["acceptance_utc"],
                      "is_regular_release": k == 0, "base_revenue": base, "base_gm": pgm,
                      "parser_version": PARSER_VERSION, "n_rev_candidates": len(revs), "n_gm_candidates": len(gms)}
            g = choose_revenue(revs)
            if g is not None:
                lo, hi, mid = g.low, g.high, g.mid
                if g.kind == "seq_pct":
                    lo, hi, mid = (base * (1 + x / 100) for x in (g.low, g.high, g.mid))
                elif g.kind == "yoy_pct":
                    if pd.isna(base_yoy):
                        lo = hi = mid = np.nan
                    else:
                        lo, hi, mid = (base_yoy * (1 + x / 100) for x in (g.low, g.high, g.mid))
                pct = g.kind in ("seq_pct", "yoy_pct")
                out.append({**common, "metric": "revenue", "basis": g.basis, "kind": g.kind, "low": lo, "high": hi,
                            "mid": mid, "pct_low": g.low if pct else np.nan, "pct_high": g.high if pct else np.nan,
                            "doc_type": g.doc_type, "source": g.source, "snippet": g.snippet})
            for basis, gg in choose_gm(gms).items():
                out.append({**common, "metric": f"gm_{basis}", "basis": gg.basis, "kind": gg.kind, "low": gg.low,
                            "high": gg.high, "mid": gg.mid, "pct_low": np.nan, "pct_high": np.nan,
                            "doc_type": gg.doc_type, "source": gg.source, "snippet": gg.snippet})
    return out


def build_guidance(filings: pd.DataFrame, ev: pd.DataFrame, vint: pd.DataFrame, cal: pd.DataFrame,
                   n_jobs: int = 6):
    panel = first_reported_panel(vint)
    rel = release_filings(filings)
    now = pd.Timestamp.now(tz="UTC")
    jobs = []
    for tk, e in ev.groupby("ticker"):
        r = rel[rel["ticker"] == tk].sort_values("acceptance_utc")
        qe = list(cal[cal["ticker"] == tk].sort_values("period_end")["period_end"])
        sl = panel.xs(tk, level="ticker") if tk in panel.index.get_level_values(0) else panel.iloc[:0]
        jobs.append(delayed(_ticker_guidance)(tk, e, r, qe, sl["revenue"].to_dict(), sl["gm"].to_dict(), now))
    res = Parallel(n_jobs=n_jobs)(jobs)
    raw = pd.DataFrame([x for rr in res for x in rr])
    if raw.empty:
        return raw, raw
    # final: latest guidance issued before the target quarter's forecast origin
    org = ev[["ticker", "period_end", "origin_utc"]].rename(columns={"period_end": "target_period_end",
                                                                       "origin_utc": "target_origin_utc"})
    raw = raw.merge(org, on=["ticker", "target_period_end"], how="left")
    usable = raw[raw["mid"].notna() & (raw["target_origin_utc"].isna() | (raw["issued_utc"] < raw["target_origin_utc"]))]
    final = (usable.sort_values("issued_utc")
             .drop_duplicates(["ticker", "target_period_end", "metric"], keep="last")
             .reset_index(drop=True))
    first_issue = usable.sort_values("issued_utc").drop_duplicates(["ticker", "target_period_end", "metric"])
    final = final.merge(first_issue[["ticker", "target_period_end", "metric", "issued_utc"]]
                        .rename(columns={"issued_utc": "first_issued_utc"}),
                        on=["ticker", "target_period_end", "metric"], how="left")
    final["is_update"] = final["issued_utc"] > final["first_issued_utc"]
    return raw, final
