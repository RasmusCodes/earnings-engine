"""Earnings-release events from 8-K Item 2.02 ("Results of Operations and Financial Condition").

For each fiscal quarter the release is the first 8-K with Item 2.02 accepted after the period end (within 100
days). EDGAR acceptance timestamps are UTC; they are converted to New York time and classified:

  BMO   : accepted before 09:30 ET (pre-market)
  INTRA : 09:30-16:00 ET
  AMC   : 16:00 ET or later (after hours)

The forecast origin is the last regular-session close (16:00 ET) strictly before the release, i.e. the last
moment a position could be set at the close before the news. Everything accepted on EDGAR before the origin is
in the information set; nothing after it is.

Caveat: the 8-K can lag the wire release by minutes (or, rarely, until the next morning), so the session label
is a proxy. A first Item 2.02 that is followed by a second one before the 10-Q is flagged as a possible
pre-announcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"


def _is_202(items: str) -> bool:
    return any(x.strip() in ("2.02", "12") for x in str(items or "").split(","))


def prev_close(ts_et: pd.Timestamp) -> pd.Timestamp:
    """Last 16:00 ET weekday close strictly before ts_et (exchange holidays ignored)."""
    d = ts_et.normalize()
    close = d + pd.Timedelta(hours=16)
    if ts_et > close and d.weekday() < 5:
        return close
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d = d - pd.Timedelta(days=1)
    return d + pd.Timedelta(hours=16)


def session_of(ts_et: pd.Timestamp) -> str:
    hm = ts_et.hour * 60 + ts_et.minute
    if hm < 9 * 60 + 30:
        return "BMO"
    if hm < 16 * 60:
        return "INTRA"
    return "AMC"


def earnings_events(filings: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    ek = filings[(filings["form"] == "8-K") & filings["items"].apply(_is_202)].copy()
    per = filings[filings["form"].isin(["10-Q", "10-K", "10-KT", "10-K405"])].copy()
    rows = []
    for tk, g in cal.groupby("ticker"):
        e8 = ek[ek["ticker"] == tk].sort_values("acceptance_utc")
        pr = per[per["ticker"] == tk]
        for _, q in g.sort_values("period_end").iterrows():
            pe = pd.Timestamp(q["period_end"]).tz_localize(ET) + pd.Timedelta(hours=23, minutes=59)
            win = e8[(e8["acceptance_et"] > pe) & (e8["acceptance_et"] <= pe + pd.Timedelta(days=100))]
            rep = pr[pr["report_date"] == pd.Timestamp(q["period_end"]).strftime("%Y-%m-%d")]
            rep = rep.sort_values("acceptance_utc")
            rep_ts = rep["acceptance_et"].iloc[0] if len(rep) else pd.NaT
            rep_accn = rep["accn"].iloc[0] if len(rep) else None
            source, first, n202, second = "none", None, len(win), pd.NaT
            if len(win):
                first = win.iloc[0]
                if pd.notna(rep_ts) and first["acceptance_et"] > rep_ts + pd.Timedelta(days=2):
                    first = None               # 2.02 only after the periodic report: not the release
                    source = "periodic_fallback"
                else:
                    source = "8-K 2.02"
                    later = win[(win["acceptance_et"] > first["acceptance_et"] + pd.Timedelta(hours=12))]
                    if pd.notna(rep_ts):
                        later = later[later["acceptance_et"] <= rep_ts + pd.Timedelta(days=1)]
                    if len(later):
                        second = later["acceptance_et"].iloc[0]
            if first is None and pd.notna(rep_ts):
                source = "periodic_fallback"
            rel_ts = first["acceptance_et"] if first is not None else rep_ts
            if first is not None and pd.notna(rep_ts) and rep_ts < rel_ts:
                # the 10-Q/10-K went out before the 8-K (DIOD 2012-2015): results were public at the earlier time
                rel_ts = rep_ts
                source = "periodic_before_8k"
            if pd.isna(rel_ts):
                continue
            rows.append({
                "ticker": tk, "period_end": q["period_end"], "fiscal_label": q.get("fiscal_label", ""),
                "release_source": source,
                "release_accn": first["accn"] if first is not None else rep_accn,
                "release_cik": int(first["cik"]) if first is not None else (int(rep["cik"].iloc[0]) if len(rep) else None),
                "release_et": rel_ts, "release_utc": rel_ts.tz_convert("UTC"),
                "session": session_of(rel_ts),
                "origin_et": prev_close(rel_ts),
                "periodic_accn": rep_accn, "periodic_et": rep_ts,
                "days_after_period": (rel_ts.tz_localize(None).normalize() - pd.Timestamp(q["period_end"])).days,
                "n_202_in_window": n202, "second_202_et": second,
                "possible_preannouncement": bool(pd.notna(second)),
            })
    ev = pd.DataFrame(rows)
    ev["origin_utc"] = pd.to_datetime(ev["origin_et"]).dt.tz_convert("UTC")
    return ev.sort_values(["ticker", "period_end"]).reset_index(drop=True)
