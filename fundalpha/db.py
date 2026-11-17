"""SQLite store. Tables are rebuilt deterministically from the raw cache, so `replace` writes are safe."""
from __future__ import annotations

import sqlite3

import pandas as pd

from .paths import DB_PATH


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def write(df: pd.DataFrame, table: str, con: sqlite3.Connection | None = None, if_exists: str = "replace"):
    own = con is None
    con = con or connect()
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str).replace({"NaT": None})
    out.to_sql(table, con, if_exists=if_exists, index=False, chunksize=20000)
    con.commit()
    if own:
        con.close()


def read(sql: str, con: sqlite3.Connection | None = None, **kw) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    try:
        return pd.read_sql_query(sql, con, **kw)
    finally:
        if own:
            con.close()
