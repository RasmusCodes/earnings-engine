"""Analyst consensus: licensed-data hook (no consensus is bundled with this project).

Drop one or more CSV exports into data/consensus/ with these columns (one row per estimate snapshot):

    ticker            engine ticker (config/universe.csv)
    period_end        fiscal quarter end the estimate is for (YYYY-MM-DD; +/- 10 days is matched)
    metric            revenue | gross_margin_pct | eps
    value             consensus mean (USD for revenue, percent for gross margin)
    as_of             date the consensus snapshot was taken (YYYY-MM-DD) -- REQUIRED for point-in-time use
    n_estimates       optional
    source            e.g. "CapIQ", "LSEG I/B/E/S"

Point in time: for a forecast origin tau the engine uses the latest snapshot with as_of < tau's calendar date, so a
history of snapshots (CapIQ "estimates history", I/B/E/S detail/summary history) is needed; a single current value
can only be used for the live quarter. Revenue consensus becomes benchmark C0 in the backtest.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .paths import DATA

CONS_DIR = DATA / "consensus"
COLUMNS = ["ticker", "period_end", "metric", "value", "as_of", "n_estimates", "source"]


def load() -> pd.DataFrame:
    CONS_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(CONS_DIR / "*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = {"ticker", "period_end", "metric", "value", "as_of"} - set(df.columns)
    if missing:
        raise ValueError(f"consensus files lack columns {sorted(missing)}")
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def attach(panel: pd.DataFrame, cons: pd.DataFrame | None = None, metric: str = "revenue") -> pd.DataFrame:
    """Add cons_<metric> = latest snapshot dated strictly before each row's origin date."""
    cons = load() if cons is None else cons
    out = panel.copy()
    col = f"cons_{metric}"
    out[col] = np.nan
    if cons.empty:
        return out
    c = cons[cons["metric"] == metric]
    origin_date = pd.to_datetime(out["origin_utc"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    vals = []
    for (tk, pe, od) in zip(out["ticker"], out["period_end"], origin_date):
        s = c[(c["ticker"] == tk) & ((c["period_end"] - pe).abs() <= pd.Timedelta(days=10)) & (c["as_of"] < od)]
        vals.append(s.sort_values("as_of")["value"].iloc[-1] if len(s) else np.nan)
    out[col] = vals
    return out
