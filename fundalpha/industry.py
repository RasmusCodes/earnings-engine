"""Industry indicator: hyperscaler capital expenditure (MSFT, GOOGL, AMZN, META), point in time.

Quarterly capex comes from the same XBRL vintage engine as company financials (cash-flow facts are year-to-date,
so quarters are derived by differencing). For a forecast origin tau the indicator uses only quarters whose filing
was accepted before tau: the latest calendar quarter that all four companies had filed, and the year-over-year log
growth of their summed capex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .edgar import build_filings, build_facts
from .http import SECClient
from .pit import build_vintages

HYPERSCALERS = pd.DataFrame({
    "ticker": ["MSFT", "GOOGL", "AMZN", "META"],
    "name": ["Microsoft", "Alphabet (Google)", "Amazon", "Meta Platforms"],
    "ciks": ["789019", "1288776;1652044", "1018724", "1326801"],
    "subindustry": ["hyperscaler"] * 4, "status": ["active"] * 4,
})
HYPERSCALERS["cik_list"] = HYPERSCALERS["ciks"].str.split(";").apply(lambda xs: [int(x) for x in xs])
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]


def capex_vintages(refresh_hours: float = 24.0) -> pd.DataFrame:
    client = SECClient()
    filings, _ = build_filings(client, HYPERSCALERS, refresh_hours)
    facts = build_facts(client, HYPERSCALERS, set(CAPEX) | {"Revenues"}, refresh_hours)
    v = build_vintages(facts, filings)
    v = v[v["item"] == "capex"].copy()
    # calendar quarter of each fiscal quarter (all four end on calendar quarter-ends, +/- a few days)
    v["cal_q"] = (v["period_end"] - pd.Timedelta(days=20)).dt.to_period("Q")
    return v


def capex_feature(v: pd.DataFrame, origins: pd.Series) -> pd.DataFrame:
    """For each origin: latest common calendar quarter filed by all four and the YoY log growth of total capex."""
    out = []
    v = v.sort_values("avail_utc")
    for tau in origins:
        s = v[v["avail_utc"] < tau].drop_duplicates(["ticker", "cal_q"], keep="last")
        w = s.pivot_table(index="cal_q", columns="ticker", values="value", aggfunc="last")
        w = w.dropna(how="any")
        if len(w) < 5:
            out.append((tau, np.nan, np.nan, None))
            continue
        tot = w.sum(axis=1)
        q = tot.index.max()
        yoy = np.log(tot.loc[q] / tot.loc[q - 4]) if (q - 4) in tot.index else np.nan
        yoy_prev = np.log(tot.loc[q - 1] / tot.loc[q - 5]) if (q - 1) in tot.index and (q - 5) in tot.index else np.nan
        out.append((tau, yoy, yoy - yoy_prev if np.isfinite(yoy_prev) else np.nan, str(q)))
    return pd.DataFrame(out, columns=["origin_utc", "hyper_capex_yoy", "hyper_capex_accel", "hyper_capex_q"])
