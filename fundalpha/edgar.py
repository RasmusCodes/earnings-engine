"""EDGAR acquisition: submissions (filing index with acceptance timestamps) and XBRL companyfacts."""
from __future__ import annotations

import json

import pandas as pd

from .http import SECClient
from .universe import load_universe

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_TXT = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}.txt"

API_MAX_AGE_H = 24.0


def submissions(client: SECClient, cik: int, refresh_hours: float = API_MAX_AGE_H) -> tuple[dict, pd.DataFrame]:
    """Return (header, filings DataFrame) for one CIK, including older paged files."""
    js = client.get_json(SUBMISSIONS.format(cik=cik), "submissions", max_age_hours=refresh_hours)
    frames = [pd.DataFrame(js["filings"]["recent"])]
    for f in js["filings"].get("files", []):
        page = client.get_json(SUBMISSIONS_PAGE.format(name=f["name"]), "submissions", max_age_hours=24 * 30)
        frames.append(pd.DataFrame(page))
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = cik
    header = {k: js.get(k) for k in ("cik", "name", "sic", "sicDescription", "fiscalYearEnd", "tickers",
                                      "exchanges", "entityType", "stateOfIncorporation")}
    header["formerNames"] = js.get("formerNames", [])
    return header, df


def companyfacts(client: SECClient, cik: int, refresh_hours: float = API_MAX_AGE_H) -> dict | None:
    return client.get_json(COMPANYFACTS.format(cik=cik), "companyfacts", max_age_hours=refresh_hours,
                           allow_404=True)


def build_filings(client: SECClient, univ: pd.DataFrame | None = None, refresh_hours: float = API_MAX_AGE_H):
    """Filing index for every CIK in the universe -> (filings DataFrame, entity header DataFrame)."""
    univ = load_universe() if univ is None else univ
    rows, heads = [], []
    for _, u in univ.iterrows():
        for cik in u["cik_list"]:
            head, df = submissions(client, cik, refresh_hours)
            df["ticker"] = u["ticker"]
            rows.append(df)
            head["ticker"] = u["ticker"]
            heads.append(head)
    filings = pd.concat(rows, ignore_index=True)
    keep = ["ticker", "cik", "accessionNumber", "form", "filingDate", "reportDate", "acceptanceDateTime", "items",
            "primaryDocument", "primaryDocDescription", "size", "isXBRL", "isInlineXBRL", "fileNumber"]
    filings = filings[[c for c in keep if c in filings.columns]].rename(columns={
        "accessionNumber": "accn", "filingDate": "filing_date", "reportDate": "report_date",
        "acceptanceDateTime": "acceptance_raw", "primaryDocument": "primary_doc",
        "primaryDocDescription": "primary_desc", "isXBRL": "is_xbrl", "isInlineXBRL": "is_ixbrl",
        "fileNumber": "file_number"})
    # EDGAR acceptanceDateTime carries a 'Z' suffix and is UTC (checked against known release times).
    filings["acceptance_utc"] = pd.to_datetime(filings["acceptance_raw"], utc=True, errors="coerce")
    filings["acceptance_et"] = filings["acceptance_utc"].dt.tz_convert("America/New_York")
    filings = filings.drop_duplicates(["cik", "accn"]).sort_values(["ticker", "acceptance_utc"]).reset_index(drop=True)
    heads_df = pd.DataFrame(heads)
    heads_df["formerNames"] = heads_df["formerNames"].apply(json.dumps)
    heads_df["tickers"] = heads_df["tickers"].apply(json.dumps)
    heads_df["exchanges"] = heads_df["exchanges"].apply(json.dumps)
    return filings, heads_df


def flatten_facts(js: dict, ticker: str, concepts: set[str] | None = None) -> pd.DataFrame:
    """companyfacts JSON -> long table, one row per (concept, unit, period, filing)."""
    out = []
    cik = int(js.get("cik", 0))
    for tax, block in js.get("facts", {}).items():
        for concept, cd in block.items():
            if concepts is not None and concept not in concepts:
                continue
            for unit, items in cd.get("units", {}).items():
                for it in items:
                    out.append((ticker, cik, tax, concept, unit, it.get("start"), it.get("end"), it.get("val"),
                                it.get("accn"), it.get("fy"), it.get("fp"), it.get("form"), it.get("filed"),
                                it.get("frame")))
    cols = ["ticker", "cik", "taxonomy", "concept", "unit", "start", "end", "val", "accn", "fy", "fp", "form",
            "filed", "frame"]
    return pd.DataFrame(out, columns=cols)


def build_facts(client: SECClient, univ: pd.DataFrame | None = None, concepts: set[str] | None = None,
                refresh_hours: float = API_MAX_AGE_H) -> pd.DataFrame:
    univ = load_universe() if univ is None else univ
    frames = []
    for _, u in univ.iterrows():
        for cik in u["cik_list"]:
            js = companyfacts(client, cik, refresh_hours)
            if js is None:
                continue
            frames.append(flatten_facts(js, u["ticker"], concepts))
    facts = pd.concat(frames, ignore_index=True)
    return facts
