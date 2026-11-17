"""Point-in-time audits on the built data (run after `fundalpha build/guidance/features/backtest`)."""
import unittest

import numpy as np
import pandas as pd

from fundalpha.paths import OUT


class PointInTime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / "panel.pkl").exists():
            raise unittest.SkipTest("built data not present: run the pipeline (see README) before the PIT audits")
        cls.panel = pd.read_pickle(OUT / "panel.pkl")
        cls.guid = pd.read_pickle(OUT / "guidance.pkl")
        cls.vint = pd.read_pickle(OUT / "fin_vintages.pkl")

    def test_origin_before_release(self):
        self.assertTrue((self.panel["origin_utc"] < self.panel["release_utc"]).all())

    def test_guidance_issued_before_origin(self):
        g = self.guid[self.guid["metric"] == "revenue"][["ticker", "target_period_end", "issued_utc"]]
        p = self.panel[self.panel["has_guid"]].merge(g, left_on=["ticker", "period_end"],
                                                     right_on=["ticker", "target_period_end"], how="left")
        self.assertTrue(p["issued_utc"].notna().all())
        self.assertTrue((p["issued_utc"] < p["origin_utc"]).all())

    def test_lagged_revenue_is_an_asof_value(self):
        """rev_l1 must equal the latest vintage of quarter t-1 accepted before the origin."""
        rng = np.random.default_rng(0)
        cal = pd.read_pickle(OUT / "quarters.pkl")
        rows = self.panel.dropna(subset=["rev_l1"]).sample(200, random_state=1)
        for _, r in rows.iterrows():
            qe = list(cal[cal["ticker"] == r["ticker"]].sort_values("period_end")["period_end"])
            prev = qe[qe.index(r["period_end"]) - 1]
            v = self.vint[(self.vint["ticker"] == r["ticker"]) & (self.vint["item"] == "revenue") &
                          (self.vint["period_end"] == prev) & (self.vint["avail_utc"] < r["origin_utc"])]
            self.assertFalse(v.empty)
            self.assertAlmostEqual(v.sort_values("avail_utc")["value"].iloc[-1], r["rev_l1"], delta=1.0)
        _ = rng

    def test_target_quarter_not_in_features(self):
        """No vintage of the target quarter itself may exist before the origin (else the target leaked)."""
        v = self.vint[self.vint["item"] == "revenue"][["ticker", "period_end", "avail_utc"]]
        first = v.groupby(["ticker", "period_end"])["avail_utc"].min().rename("first_avail")
        p = self.panel.join(first, on=["ticker", "period_end"])
        self.assertTrue((p["first_avail"].isna() | (p["first_avail"] > p["origin_utc"])).all())

    def test_walkforward_training_cutoff(self):
        """Every training row's outcome was public before its window began, and every test origin is inside it."""
        for name in ("predictions_all.pkl", "predictions_dev.pkl"):
            path = OUT / name
            if not path.exists():
                continue
            pred = pd.read_pickle(path)
            w = pred.groupby("window")["origin_utc"].min()
            for win, omin in w.items():
                cut = pd.Period(win, freq="Q").start_time.tz_localize("UTC")
                self.assertGreaterEqual(omin, cut)
            # n_train recomputation for one experiment
            from fundalpha.walkforward import samples
            mask = samples(self.panel)["rev_all"]
            sub = pred[(pred["experiment"] == "rev_all")].drop_duplicates("window")
            for _, r in sub.iterrows():
                cut = pd.Period(r["window"], freq="Q").start_time.tz_localize("UTC")
                n = int(((self.panel["release_utc"] < cut) & mask).sum())
                self.assertEqual(n, r["n_train"])
            break


if __name__ == "__main__":
    unittest.main()
