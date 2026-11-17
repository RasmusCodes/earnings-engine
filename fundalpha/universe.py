"""Company universe: an editable CSV (config/universe.csv).

Each row is one economic entity. `ciks` is a ';'-separated chain, oldest first, for companies whose SEC
registrant changed (re-domiciliations: Avago -> Broadcom Ltd -> Broadcom Inc; Marvell Ltd -> Marvell Inc).
Acquired companies stay in the file so the research sample is not survivorship-biased: a firm-quarter is
eligible when the firm was large enough *at the time* (see eligibility rule in features.py), not because the
firm still exists today.
"""
from __future__ import annotations

import pandas as pd

from .paths import CONFIG


def load_universe(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or CONFIG / "universe.csv", dtype=str).fillna("")
    df["cik_list"] = df["ciks"].str.split(";").apply(lambda xs: [int(x) for x in xs if x.strip()])
    return df


def cik_to_ticker(univ: pd.DataFrame) -> dict[int, str]:
    out = {}
    for _, r in univ.iterrows():
        for c in r["cik_list"]:
            out[c] = r["ticker"]
    return out
