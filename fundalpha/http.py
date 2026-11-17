"""Polite SEC EDGAR client with an on-disk raw cache.

* The User-Agent comes from the SEC_USER_AGENT environment variable at runtime (SEC requires a declared
  "Name email" contact). It is never written to disk or logs.
* Throttled below SEC's 10 requests/second fair-access limit.
* Every response body is stored gzip-compressed under data/raw/sec/<kind>/ and indexed in
  data/raw/sec/index.jsonl (url, path, UTC fetch time, status, bytes, sha256), so any number in the
  database can be traced to the exact bytes it came from.
* Archive documents are immutable (read back from cache forever); API JSON is refreshed after max_age_hours.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .paths import LOGS, RAW

SEC_DIR = RAW / "sec"
INDEX = SEC_DIR / "index.jsonl"
RETRY = {429, 500, 502, 503, 504}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FetchError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


class SECClient:
    def __init__(self, rate_per_sec: float = 6.0):
        ua = os.environ.get("SEC_USER_AGENT", "").strip()
        if "@" not in ua:
            raise RuntimeError("Set SEC_USER_AGENT to 'Name email' (SEC fair-access policy) before fetching.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / rate_per_sec
        self._last = 0.0
        self._lock = threading.Lock()
        self.n_network = 0
        self.n_cache = 0
        SEC_DIR.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _load_index() -> dict:
        idx = {}
        if INDEX.exists():
            with INDEX.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") == 200:
                        idx[rec["url"]] = rec
        return idx

    def _store(self, url: str, kind: str, status: int, body: bytes) -> dict:
        h = hashlib.sha1(url.encode()).hexdigest()
        folder = SEC_DIR / kind / h[:2]
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{h}.gz"
        p.write_bytes(gzip.compress(body))
        rec = {"url": url, "path": p.relative_to(RAW).as_posix(), "fetched_at": utcnow(), "status": status,
               "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        with self._lock:
            with INDEX.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if status == 200:
                self._index[url] = rec
        return rec

    def cached(self, url: str) -> bytes | None:
        rec = self._index.get(url)
        if rec is None:
            return None
        p = RAW / rec["path"]
        if not p.exists():
            return None
        return gzip.decompress(p.read_bytes())

    def fetched_at(self, url: str) -> str | None:
        rec = self._index.get(url)
        return rec["fetched_at"] if rec else None

    # ------------------------------------------------------------------ network
    def _throttle(self):
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, url: str, kind: str, *, immutable: bool = False, max_age_hours: float | None = None,
            allow_404: bool = False) -> bytes | None:
        rec = self._index.get(url)
        if rec is not None:
            fresh = immutable
            if not fresh and max_age_hours is not None:
                got = datetime.strptime(rec["fetched_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                fresh = datetime.now(timezone.utc) - got < timedelta(hours=max_age_hours)
            if fresh:
                body = self.cached(url)
                if body is not None:
                    self.n_cache += 1
                    return body
        last_exc = None
        for attempt in range(4):
            self._throttle()
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(3 * (attempt + 1))
                continue
            self.n_network += 1
            if r.status_code in RETRY and attempt < 3:
                ra = r.headers.get("Retry-After", "")
                time.sleep(min(int(ra) if ra.isdigit() else 5 * 2 ** attempt, 60))
                continue
            if r.status_code == 404 and allow_404:
                self._log_error(url, 404)
                return None
            if r.status_code != 200:
                self._log_error(url, r.status_code)
                raise FetchError(r.status_code, url)
            self._store(url, kind, 200, r.content)
            return r.content
        raise FetchError(-1, f"{url} ({type(last_exc).__name__ if last_exc else 'retries exhausted'})")

    @staticmethod
    def _log_error(url: str, status: int):
        with (LOGS / "fetch_errors.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": url, "status": status, "at": utcnow()}) + "\n")

    def get_json(self, url: str, kind: str, **kw):
        body = self.get(url, kind, **kw)
        return None if body is None else json.loads(body)
