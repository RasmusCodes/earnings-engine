"""Earnings-release documents: every 8-K with Item 2.02 -> EX-99 exhibits as text.

One request per filing: the EDGAR full submission file (<accession>.txt) holds every document with its
<TYPE> tag, so exhibits are found without guessing file names (NVDA q2fy27pr.htm, KLA
exhibit991earningsrelease0.htm, ...). Raw files sit in the immutable SEC cache; extracted text is written to
data/derived/releases/<accession>.json.gz together with the filing's acceptance timestamp.

HTML tables become one line per row with cells joined by " | ", so tabular outlooks
("Revenue | $4.20 Billion | +/- | $300 Million") stay parseable.
"""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .edgar import ARCHIVE_TXT
from .http import SECClient
from .paths import DATA

import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

REL_DIR = DATA / "derived" / "releases"
DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "center", "pre"]
TRANSLATE = str.maketrans({"\xa0": " ", " ": " ", " ": " ", " ": " ", "​": "",
                           "–": "-", "—": "-", "−": "-", "‘": "'", "’": "'",
                           "“": '"', "”": '"', "•": "*", "�": " "})


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}>([^\n<]*)", block, re.I)
    return m.group(1).strip() if m else ""


def normalize(text: str) -> str:
    text = text.translate(TRANSLATE)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out, blank = [], 0
    for ln in lines:
        if ln:
            out.append(ln)
            blank = 0
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def html_to_text(raw: str) -> str:
    low = raw[:5000].lower()
    if "<html" not in low and "<table" not in raw.lower() and "<p" not in low and "<div" not in low:
        return normalize(raw)
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style", "head", "title"]):
        t.decompose()
    for table in reversed(soup.find_all("table")):
        lines = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                lines.append(" | ".join(cells))
        if table.parent is not None:
            table.replace_with("\n" + "\n".join(lines) + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    return normalize(soup.get_text(" "))


def split_exhibits(sub_txt: str) -> list[dict]:
    docs = []
    for m in DOC_RE.finditer(sub_txt):
        block = m.group(1)
        typ = _tag(block, "TYPE").upper()
        if not typ.startswith("EX-99"):
            continue
        body = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S | re.I)
        raw = body.group(1) if body else ""
        docs.append({"type": typ, "filename": _tag(block, "FILENAME"), "description": _tag(block, "DESCRIPTION"),
                     "text": html_to_text(raw)})
    return docs


def release_filings(filings: pd.DataFrame, since: str = "2008-01-01") -> pd.DataFrame:
    def is202(items):
        return any(x.strip() in ("2.02", "12") for x in str(items or "").split(","))
    f = filings[(filings["form"] == "8-K") & filings["items"].apply(is202) & (filings["filing_date"] >= since)]
    return f.sort_values(["ticker", "acceptance_utc"]).reset_index(drop=True)


def _path(accn: str):
    return REL_DIR / accn[:10] / f"{accn}.json.gz"


def fetch_one(client: SECClient, row) -> tuple[str, str]:
    p = _path(row["accn"])
    if p.exists():
        return row["accn"], "cached"
    url = ARCHIVE_TXT.format(cik=int(row["cik"]), acc_nodash=row["accn"].replace("-", ""), acc=row["accn"])
    body = client.get(url, "archives", immutable=True, allow_404=True)
    if body is None:
        return row["accn"], "404"
    docs = split_exhibits(body.decode("utf-8", errors="replace"))
    rec = {"accn": row["accn"], "ticker": row["ticker"], "cik": int(row["cik"]),
           "acceptance_utc": str(row["acceptance_utc"]), "items": row["items"], "url": url, "docs": docs}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(rec).encode("utf-8")))
    return row["accn"], f"ok:{len(docs)}"


def download(filings: pd.DataFrame, tickers: list[str] | None = None, workers: int = 3, log_every: int = 100):
    REL_DIR.mkdir(parents=True, exist_ok=True)
    todo = release_filings(filings)
    if tickers:
        todo = todo[todo["ticker"].isin(tickers)]
    print(f"8-K Item 2.02 filings to process: {len(todo):,}", flush=True)
    client = SECClient(rate_per_sec=6.0)
    stats: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, client, r) for _, r in todo.iterrows()]
        for fu in as_completed(futs):
            try:
                _, st = fu.result()
            except Exception as exc:          # keep going; failures are logged and retried on the next run
                st = f"error:{type(exc).__name__}"
                print(f"  {st}: {exc}", flush=True)
            key = st.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            done += 1
            if done % log_every == 0:
                print(f"  {done:,}/{len(todo):,} {stats} network={client.n_network}", flush=True)
    print(f"finished {stats} network={client.n_network}", flush=True)


def load_release(accn: str) -> dict | None:
    p = _path(accn)
    if not p.exists():
        return None
    return json.loads(gzip.decompress(p.read_bytes()))
