"""Point-in-time quarterly financials from XBRL vintages.

Every filing (10-Q, 10-K and amendments) is replayed in acceptance-time order. After each filing the latest
value of every reported period is known; discrete fiscal quarters are then derived:

* direct      : a reported 3-month value (80-105 day duration)
* ytd_diff    : cumulative year-to-date differences with the same fiscal-year start (cash-flow statements are
                reported YTD only, and Q4 = full year - nine months)

Whenever a derived quarterly value appears or changes, one row is written to the vintage table with the
acceptance timestamp of the filing that produced it (`avail_utc`). Two views follow directly:

* first-reported value of a quarter = its earliest vintage (what the market saw, used as forecast target)
* value known at time tau            = latest vintage with avail_utc < tau (used for features)

The `fy` / `fp` fields of companyfacts describe the *filing*, not the fact (comparatives carry the filing's
labels), so periods are identified only by their start/end dates here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .items import DURATION_ITEMS, INSTANT_ITEMS, NONADDITIVE_ITEMS

FIN_FORMS = ("10-Q", "10-K", "10-KT", "10-Q/A", "10-K/A", "10-KT/A")
Q_MIN, Q_MAX = 80, 105          # 3-month duration (13- or 14-week quarters included)
REL_TOL = 1e-6


def _prep(facts: pd.DataFrame, filings: pd.DataFrame, concepts: list[str], unit: str) -> pd.DataFrame:
    f = facts[facts["concept"].isin(concepts) & (facts["unit"] == unit) & facts["form"].isin(FIN_FORMS)].copy()
    if f.empty:
        return f
    f["prio"] = f["concept"].map({c: i for i, c in enumerate(concepts)})
    acc = filings[["accn", "acceptance_utc"]].drop_duplicates("accn")
    f = f.merge(acc, on="accn", how="left")
    # fallback when the accession is not in the submissions index: end of the 'filed' day, New York time
    fb = (pd.to_datetime(f["filed"]) + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize("America/New_York")
    f["avail_utc"] = f["acceptance_utc"].fillna(fb.dt.tz_convert("UTC"))
    f["end"] = pd.to_datetime(f["end"])
    if "start" in f and f["start"].notna().any():
        f["start"] = pd.to_datetime(f["start"])
    f = f.sort_values(["avail_utc", "accn", "prio", "val"])
    return f


def _fy_label(fy_end: pd.Timestamp) -> int:
    return int((fy_end - pd.Timedelta(days=15)).year)


MAX_YTD_VINTAGE_GAP = pd.Timedelta(days=400)


def _derive_quarters(cur: dict) -> dict:
    """cur: {(start, end): (value, avail)} for duration facts -> {quarter_end: (start, value, derivation, fy_start)}

    A year-to-date difference is only formed when both cumulative values were filed within ~13 months of each
    other: a full year restated years later (discontinued operations) minus an original nine-month figure would
    mix bases (SLAB FY2019 recast in 2022)."""
    direct, cum = {}, defaultdict(dict)
    fy_starts = set()
    for (s, e), (v, _) in cur.items():
        d = (e - s).days + 1
        if Q_MIN <= d <= Q_MAX:
            direct[e] = (s, v)
        if d > Q_MAX:
            fy_starts.add(s)
    for (s, e), va in cur.items():
        if s in fy_starts:
            cum[s][e] = va         # includes Q1's own 3-month value (start == fiscal-year start)
    out = {}
    for s, pts in cum.items():
        ends = sorted(pts)
        prev_e = None
        for e in ends:
            if prev_e is None:
                d = (e - s).days + 1
                if Q_MIN <= d <= Q_MAX:
                    out[e] = (s, pts[e][0], "direct", s)
            else:
                gap = (e - prev_e).days
                if Q_MIN <= gap <= Q_MAX and abs(pts[e][1] - pts[prev_e][1]) <= MAX_YTD_VINTAGE_GAP:
                    out[e] = (prev_e + pd.Timedelta(days=1), pts[e][0] - pts[prev_e][0], "ytd_diff", s)
            prev_e = e
    for e, (s, v) in direct.items():
        fs = next((x for x in sorted(fy_starts) if x <= s and (e - x).days < 380), None)
        out[e] = (s, v, "direct", fs if fs is not None else (out.get(e, (None, None, None, None))[3]))
    return out


def duration_vintages(f: pd.DataFrame, item: str, additive: bool = True) -> list[dict]:
    rows = []
    if f.empty:
        return rows
    cur: dict = {}
    prev: dict = {}
    for (avail, accn), g in f.groupby(["avail_utc", "accn"], sort=True):
        g = g.dropna(subset=["start"]).drop_duplicates(["start", "end"], keep="first")
        for s, e, v in zip(g["start"], g["end"], g["val"]):
            cur[(s, e)] = (float(v), avail)
        if additive:
            q = _derive_quarters(cur)
        else:
            q = {e: (s, v, "direct", None) for (s, e), (v, _) in cur.items() if Q_MIN <= (e - s).days + 1 <= Q_MAX}
        form = g["form"].iloc[0]
        for e, (s, v, how, fys) in q.items():
            pv = prev.get(e)
            if pv is None or abs(pv - v) > REL_TOL * max(1.0, abs(pv)):
                rows.append({"item": item, "period_start": s, "period_end": e, "value": v, "derivation": how,
                             "fy_start": fys, "avail_utc": avail, "accn": accn, "form": form})
            prev[e] = v
    return rows


def instant_vintages(f: pd.DataFrame, item: str) -> list[dict]:
    rows, prev = [], {}
    if f.empty:
        return rows
    for (avail, accn), g in f.groupby(["avail_utc", "accn"], sort=True):
        g = g.drop_duplicates(["end"], keep="first")
        form = g["form"].iloc[0]
        for e, v in zip(g["end"], g["val"]):
            v = float(v)
            pv = prev.get(e)
            if pv is None or abs(pv - v) > REL_TOL * max(1.0, abs(pv)):
                rows.append({"item": item, "period_start": pd.NaT, "period_end": e, "value": v,
                             "derivation": "instant", "fy_start": pd.NaT, "avail_utc": avail, "accn": accn,
                             "form": form})
            prev[e] = v
    return rows


def canonical_boundaries(ft: pd.DataFrame, tol_days: int = 10) -> dict:
    """Map period boundary dates that differ by a few days onto one canonical date.

    Some filers tag the same quarter with two end dates (fiscal 52/53-week date in the 10-Q, calendar
    month-end in a later 10-K's quarterly note or comparatives, e.g. TER 2011-04-03 vs 2011-03-31). Boundary
    dates are period ends and (start - 1 day); dates within `tol_days` of each other are one boundary, and the
    most frequently used date wins.
    """
    ends = pd.to_datetime(ft["end"])
    starts = pd.to_datetime(ft["start"].dropna()) - pd.Timedelta(days=1)
    allb = pd.concat([ends, starts], ignore_index=True)
    counts = allb.value_counts()
    uniq = sorted(counts.index)
    mapping, cluster = {}, []

    def flush():
        if cluster:
            best = max(cluster, key=lambda d: (counts[d], -d.value))
            for d in cluster:
                mapping[d] = best

    for d in uniq:
        if cluster and (d - cluster[-1]).days > tol_days:
            flush()
            cluster = []
        cluster.append(d)
    flush()
    return mapping


def _apply_boundaries(ft: pd.DataFrame) -> pd.DataFrame:
    m = canonical_boundaries(ft)
    ft = ft.copy()
    e = pd.to_datetime(ft["end"])
    ft["end"] = e.map(lambda d: m.get(d, d)).dt.strftime("%Y-%m-%d")
    s = pd.to_datetime(ft["start"])
    has = s.notna()
    adj = (s[has] - pd.Timedelta(days=1)).map(lambda d: m.get(d, d)) + pd.Timedelta(days=1)
    ft["start"] = ft["start"].astype(object)
    ft.loc[has, "start"] = adj.dt.strftime("%Y-%m-%d")
    return ft


def build_vintages(facts: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    out = []
    for tk, ft in facts.groupby("ticker"):
        ft = _apply_boundaries(ft)
        fl = filings[filings["ticker"] == tk]
        for item, concepts in DURATION_ITEMS.items():
            for r in duration_vintages(_prep(ft, fl, concepts, "USD"), item):
                r["ticker"] = tk
                out.append(r)
        for item, concepts in NONADDITIVE_ITEMS.items():
            unit = "USD/shares" if item.startswith("eps") else "shares"
            for r in duration_vintages(_prep(ft, fl, concepts, unit), item, additive=False):
                r["ticker"] = tk
                out.append(r)
        for item, concepts in INSTANT_ITEMS.items():
            for r in instant_vintages(_prep(ft, fl, concepts, "USD"), item):
                r["ticker"] = tk
                out.append(r)
    v = pd.DataFrame(out)
    v["period_days"] = (v["period_end"] - v["period_start"]).dt.days + 1
    v = clean_vintages(v)
    return v.sort_values(["ticker", "item", "period_end", "avail_utc"]).reset_index(drop=True)


def quarter_calendar(vint: pd.DataFrame) -> pd.DataFrame:
    """Fiscal quarter grid per ticker from revenue quarters (latest vintage): fiscal year/quarter labels."""
    rev = vint[vint["item"] == "revenue"].sort_values("avail_utc").drop_duplicates(["ticker", "period_end"],
                                                                                   keep="last")
    rows = []
    for tk, g in rev.groupby("ticker"):
        g = g.sort_values("period_end")
        # fiscal-year start for each quarter: from vintages where known, else from the quarter sequence
        for _, r in g.iterrows():
            rows.append({"ticker": tk, "period_end": r["period_end"], "period_start": r["period_start"],
                         "fy_start": r["fy_start"]})
    cal = pd.DataFrame(rows)
    cal["period_days"] = (cal["period_end"] - cal["period_start"]).dt.days + 1
    fq, fy = [], []
    for _, r in cal.iterrows():
        if pd.notna(r["fy_start"]):
            q = int(round(((r["period_end"] - r["fy_start"]).days + 1) / 91.3))
            q = min(max(q, 1), 4)
            fy_end_guess = r["fy_start"] + pd.Timedelta(days=364)
            fq.append(q)
            fy.append(_fy_label(fy_end_guess))
        else:
            fq.append(np.nan)
            fy.append(np.nan)
    cal["fiscal_qtr"] = fq
    cal["fiscal_year"] = fy
    # fill missing labels by stepping from neighbours (quarters are consecutive)
    out = []
    for tk, g in cal.groupby("ticker"):
        g = g.sort_values("period_end").reset_index(drop=True)
        for i in range(len(g)):
            if pd.isna(g.loc[i, "fiscal_qtr"]):
                # look forward for the next labelled quarter
                j = next((k for k in range(i + 1, len(g)) if pd.notna(g.loc[k, "fiscal_qtr"])), None)
                if j is not None:
                    steps = j - i
                    q = int(g.loc[j, "fiscal_qtr"]) - steps
                    y = int(g.loc[j, "fiscal_year"])
                    while q < 1:
                        q += 4
                        y -= 1
                    g.loc[i, "fiscal_qtr"], g.loc[i, "fiscal_year"] = q, y
        out.append(g)
    cal = pd.concat(out, ignore_index=True)
    cal["fiscal_label"] = cal.apply(lambda r: f"FY{int(r.fiscal_year)}Q{int(r.fiscal_qtr)}"
                                    if pd.notna(r.fiscal_qtr) else "", axis=1)
    return cal


SCALE_CHECK_ITEMS = {"revenue", "cogs", "gross_profit", "rnd", "sga", "opex_total", "shares_diluted", "inventory",
                     "receivables", "total_assets", "ppe_net", "goodwill", "capex", "dna", "sbc", "cash",
                     "intangible_amort"}


def clean_vintages(v: pd.DataFrame) -> pd.DataFrame:
    """Remove XBRL tagging errors that never reflected what the company published.

    * revenue <= 0 (a pre-merger shell registrant's 10-Q, QRVO 2014; Q4 derivations across restated bases)
    * scale errors: a vintage ~10^3 or ~10^6 away from the level of the neighbouring quarters (median of the two
      quarters on each side, latest vintages), e.g. MKSI Q1-2018 revenue tagged as $554 billion
    """
    v = v[~((v["item"] == "revenue") & (v["value"] <= 0))].copy()
    latest = v.sort_values("avail_utc").drop_duplicates(["ticker", "item", "period_end"], keep="last")
    refs = []
    for (tk, it), g in latest[latest["item"].isin(SCALE_CHECK_ITEMS)].groupby(["ticker", "item"]):
        g = g.sort_values("period_end")
        a = g["value"].abs().to_numpy()
        ref = [np.median(np.r_[a[max(0, i - 2):i], a[i + 1:i + 3]]) if len(a) > 1 else np.nan for i in range(len(a))]
        refs.append(pd.DataFrame({"ticker": tk, "item": it, "period_end": g["period_end"].values, "ref": ref}))
    if refs:
        r = pd.concat(refs, ignore_index=True)
        v = v.merge(r, on=["ticker", "item", "period_end"], how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log10(v["value"].abs() / v["ref"])
        bad = v["ref"].gt(0) & (((lg - 3).abs() < 0.3) | ((lg + 3).abs() < 0.3) | ((lg - 6).abs() < 0.3) |
                                ((lg + 6).abs() < 0.3))
        v = v[~bad].drop(columns="ref")
    return v.reset_index(drop=True)


def gross_margin(rev, gp, cogs):
    """GM in percent: GrossProfit tag when present, else revenue - cost of revenue.

    A GrossProfit fact equal to minus (revenue - cost) is a sign error in the tag (old SanDisk Q4-2009) and is
    flipped; margins outside (-100%, 100%) are treated as missing."""
    rev = pd.Series(rev, dtype=float)
    gp = pd.Series(gp, dtype=float, index=rev.index)
    cogs = pd.Series(cogs, dtype=float, index=rev.index)
    alt = rev - cogs
    signerr = gp.notna() & alt.notna() & (gp < 0) & ((gp + alt).abs() < 0.02 * rev.abs())
    g = gp.where(gp.notna(), alt).where(~signerr, alt)
    gm = 100 * g / rev.where(rev > 0)
    return gm.where(gm.abs() < 100)


def first_reported(vint: pd.DataFrame) -> pd.DataFrame:
    return vint.sort_values("avail_utc").drop_duplicates(["ticker", "item", "period_end"], keep="first")


def asof(vint: pd.DataFrame, tau: pd.Timestamp) -> pd.DataFrame:
    v = vint[vint["avail_utc"] < tau]
    return v.sort_values("avail_utc").drop_duplicates(["ticker", "item", "period_end"], keep="last")
