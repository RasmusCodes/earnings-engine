"""Consensus hook: only snapshots dated before the forecast origin may be used."""
import unittest

import pandas as pd

from fundalpha.consensus import attach


class ConsensusPointInTime(unittest.TestCase):
    def test_latest_snapshot_before_origin(self):
        panel = pd.DataFrame({"ticker": ["NVDA"], "period_end": [pd.Timestamp("2026-10-25")],
                              "origin_utc": [pd.Timestamp("2026-11-18 21:00", tz="UTC")]})
        cons = pd.DataFrame({"ticker": ["NVDA"] * 3, "metric": ["revenue"] * 3,
                             "period_end": pd.to_datetime(["2026-10-26"] * 3),
                             "as_of": pd.to_datetime(["2026-09-01", "2026-11-17", "2026-11-19"]),
                             "value": [50e9, 54e9, 99e9]})
        out = attach(panel, cons)
        self.assertEqual(out["cons_revenue"].iloc[0], 54e9)      # 11-19 snapshot is after the origin

    def test_no_files_means_empty_column(self):
        panel = pd.DataFrame({"ticker": ["NVDA"], "period_end": [pd.Timestamp("2026-10-25")],
                              "origin_utc": [pd.Timestamp("2026-11-18 21:00", tz="UTC")]})
        out = attach(panel, pd.DataFrame(columns=["ticker", "period_end", "metric", "value", "as_of"]))
        self.assertTrue(out["cons_revenue"].isna().all())


if __name__ == "__main__":
    unittest.main()
