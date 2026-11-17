"""Parser unit tests on real phrasings seen in semiconductor earnings releases."""
import unittest

from fundalpha.guidance import _norm, extract, parse_gm, parse_revenue


def doc(text, typ="EX-99.1"):
    return [{"type": typ, "text": text}]


class RevenueGuidance(unittest.TestCase):
    def test_pm_pct(self):          # NVIDIA FY26
        revs, _ = extract(doc("Outlook\n* Revenue is expected to be $43.0 billion, plus or minus 2%."), 39.3e9)
        g = revs[0]
        self.assertEqual(g.kind, "pm_pct")
        self.assertAlmostEqual(g.mid, 43.0e9)
        self.assertAlmostEqual(g.low, 42.14e9)

    def test_range_millions(self):  # ON 2019
        revs, _ = extract(doc("Based on product booking trends, backlog levels, and estimated turns levels, the Company "
                              "anticipates that total revenue in first quarter of 2019 will be approximately $1,365 to "
                              "$1,415 million."), 1.44e9)
        self.assertEqual(revs[0].kind, "range")
        self.assertAlmostEqual(revs[0].low, 1365e6)
        self.assertAlmostEqual(revs[0].high, 1415e6)

    def test_table_pm_abs(self):    # Lam Research table
        text = ("Outlook for the March 2025 quarter\nU.S. GAAP | Reconciling Items | Non-GAAP\n"
                "Revenue | $4.65 Billion | +/- | $300 Million | - | $4.65 Billion | +/- | $300 Million\n"
                "Gross margin as a percentage of revenue | 47.9% | +/- | 1% | 2.8 | Million | 48.0% | +/- | 1%")
        revs, gms = extract(doc(text), 4.38e9, 47.4)
        self.assertEqual(revs[0].kind, "pm_abs")
        self.assertAlmostEqual(revs[0].low, 4.35e9)
        by = {g.basis: g for g in gms}
        self.assertAlmostEqual(by["gaap"].mid, 47.9)
        self.assertAlmostEqual(by["nongaap"].mid, 48.0)

    def test_seq_pct(self):         # NVIDIA FY12 style
        revs, _ = extract(doc("· | Revenue is expected to be up 6 to 8 percent from the fourth quarter."), 886.4e6)
        self.assertEqual(revs[0].kind, "seq_pct")
        self.assertEqual((revs[0].low, revs[0].high), (6.0, 8.0))

    def test_past_sentence_rejected(self):
        revs, _ = extract(doc("Revenue of $886.4 million was in line with our expectations."), 886.4e6)
        self.assertEqual(revs, [])

    def test_annual_rejected_by_ratio(self):
        revs, _ = extract(doc("For fiscal 2026 we expect revenue of approximately $5.6 billion."), 1.3e9)
        self.assertEqual(revs, [])

    def test_unit_inference(self):  # table without units, header says (in millions)
        r = parse_revenue(_norm("Revenue | $1,350 to $1,450"), 1.45e9)
        self.assertEqual(r[0][0], "range")
        self.assertAlmostEqual(r[0][1], 1350e6)


class FormatRegressions(unittest.TestCase):
    """Formats that broke the first full run (2026-11-13 QA)."""

    def test_klac_bullets_eps_not_revenue(self):
        text = ("First Quarter Fiscal 2024 Guidance The following details our guidance for the first quarter of fiscal "
                "2024 ending in September: * Total revenues is expected to be in a range of $2.35 billion +/- $125 "
                "million * GAAP gross margin is expected to be in a range of 59.0% +/- 1.0% * Non-GAAP gross margin "
                "is expected to be in a range of 61.0% +/- 1.0% * GAAP diluted EPS is expected to be in a range of "
                "$5.66 +/- $0.60")
        revs, gms = extract(doc(text), 2.36e9, 59.0)
        self.assertAlmostEqual(revs[0].mid, 2.35e9)
        by = {g.basis: g.mid for g in gms}
        self.assertAlmostEqual(by["gaap"], 59.0)
        self.assertAlmostEqual(by["nongaap"], 61.0)

    def test_msccc_eps_range_ignored(self):
        text = ("Business Outlook Microsemi currently expects net sales in the second quarter of fiscal year 2017 of "
                "between $430 million and $450 million, and expects non-GAAP diluted earnings per share of between "
                "$0.86 and $0.96.")
        revs, _ = extract(doc(text), 435.5e6)
        self.assertEqual(len(revs), 1)
        self.assertAlmostEqual(revs[0].mid, 440e6)

    def test_amat_table_without_dollar(self):
        text = ("Applied's total net revenue, non-GAAP gross margin and non-GAAP diluted EPS for the third quarter of "
                "fiscal 2025 are expected to be approximately as follows:\nQ3 FY2025\n"
                "(In millions, except percentage and per share amounts)\nTotal net revenue | 7,200 | +/- | 500")
        revs, _ = extract(doc(text), 7.1e9)
        self.assertEqual(revs[0].kind, "pm_abs")
        self.assertAlmostEqual(revs[0].low, 6.7e9)

    def test_nxp_low_mid_high(self):
        text = ("Guidance for the First Quarter 2022: ($ millions) (1)\nGuidance Range\nGAAP | Reconciliation | non-GAAP\n"
                "Low | Mid | High | Low | Mid | High\n"
                "Total Revenue | $3,025 | $3,100 | $3,175 | $3,025 | $3,100 | $3,175\n"
                "Gross Margin | 55.8% | 56.3% | 56.9% | 56.8% | 57.3% | 57.8%")
        revs, gms = extract(doc(text), 3.04e9, 56.0)
        self.assertAlmostEqual(revs[0].low, 3025e6)
        self.assertAlmostEqual(revs[0].high, 3175e6)
        by = {g.basis: (g.low, g.high) for g in gms}
        self.assertEqual(by["gaap"], (55.8, 56.9))
        self.assertEqual(by["nongaap"], (56.8, 57.8))

    def test_historical_table_is_not_guidance(self):
        text = ("Results Summary\nQ1 FY2025 | Q1 FY2024 | Change\n(In millions, except per share amounts)\n"
                "Net revenue | 7,166 | 6,707 | 7%")
        revs, _ = extract(doc(text), 7.166e9)
        self.assertEqual(revs, [])

    def test_qorvo_bullet_without_verb(self):
        text = ("Qorvo's current outlook for the March 2019 quarter:\n▪ | Quarterly revenue in the range of $660 "
                "million to $680 million\n▪ | Gross margin of approximately 47%")
        revs, _ = extract(doc(text), 7.9e8)
        self.assertAlmostEqual(revs[0].mid, 670e6)

    def test_highlight_bullets_are_not_guidance(self):
        text = ("Fourth Quarter Highlights and our outlook\n* Revenue of $9,295 million for the fourth quarter, up 4 "
                "percent from the prior year period")
        self.assertEqual(extract(doc(text), 9.295e9)[0], [])
        text = "Outlook commentary\n* | Revenue of $184.1 million and non-GAAP earnings per share of $0.13 met guidance"
        self.assertEqual(extract(doc(text), 184.1e6)[0], [])

    def test_past_percent_change_is_not_guidance(self):
        s = ("Rodgers said, We are pleased to report a sequential revenue increase of 4.3% in a quarter that normally "
             "sees a seasonal decrease of 4% to 8%.")
        self.assertEqual(extract(doc(s), 202e6)[0], [])
        self.assertEqual(extract(doc("Our outlook\n | Test revenue down 7% in Q122 from Q121"), 755e6)[0], [])

    def test_historical_triple_needs_low_high_header(self):
        text = ("Business outlook and forward-looking statements\nRevenue | $188,723 | $193,466 | $203,015 | $188,723 | "
                "$193,466 | $203,015")
        self.assertEqual(extract(doc(text), 188.7e6)[0], [])

    def test_targets_and_forecasting_verbs(self):
        revs, _ = extract(doc("For its third quarter of fiscal 2023, Wolfspeed targets revenue in a range of $210 "
                              "million to $230 million."), 216e6)
        self.assertAlmostEqual(revs[0].mid, 220e6)
        revs, _ = extract(doc("For the second quarter of fiscal 2025, we are forecasting revenue of $2.50 billion, "
                              "+/- $100 million."), 2.42e9)
        self.assertAlmostEqual(revs[0].mid, 2.5e9)


class GrossMarginTables(unittest.TestCase):
    HDR = "Outlook for the next quarter\nGAAP | Reconciliation | non-GAAP\n"

    def gm(self, row, hdr=None, prev=40.0):
        _, gms = extract(doc((hdr if hdr is not None else self.HDR) + row), None, prev)
        return {g.basis: (g.low, g.high) for g in gms}

    def test_range_cells(self):          # ON 2017
        self.assertEqual(self.gm("Gross Margin | 36.0% to 38.0% | 0.2% | 36.2% to 38.2%", prev=37.0),
                         {"gaap": (36.0, 38.0), "nongaap": (36.2, 38.2)})

    def test_lost_dash(self):            # LSI 2012
        self.assertEqual(self.gm("Gross Margin | 47% 51% | $20 million to $30 million | 52% 54%", prev=48.0),
                         {"gaap": (47.0, 51.0), "nongaap": (52.0, 54.0)})

    def test_comparison_row_rejected(self):   # Intel results table
        self.assertEqual(self.gm("Gross margin | 45.7% | 39.2% | up 6.5 ppts | 48.8% | 43.8% | up 5 ppts", prev=45.0), {})

    def test_past_row_rejected(self):          # AAOI highlights
        self.assertEqual(self.gm("· | GAAP gross margin was 21.6%, compared with 23.3% in the fourth quarter",
                                 prev=22.0), {})

    def test_historical_multi_column_rows_rejected(self):
        self.assertEqual(self.gm("Gross margin | 14.1% | 18.8% | 19.9% | 17.7% | 18.4%", prev=18.0), {})      # AMKR
        self.assertEqual(self.gm("Gross margin | 42.3% | 28.2% | 32.8% | 1,410bps | 950bps", prev=40.0), {})  # LITE
        self.assertEqual(self.gm("Gross Margin | $689.3 | 48.1% | $933.7 | 61.7%", prev=50.0), {})            # MCHP

    def test_pm_cells_with_reconciliation_kept(self):   # FormFactor
        self.assertEqual(self.gm("Gross margin | 52.3% +/- 1.5% | $4.9 million | 54.0% +/- 1.5%", prev=50.0),
                         {"gaap": (50.8, 53.8), "nongaap": (52.5, 55.5)})

    def test_point_with_zero_reconciliation(self):   # SLAB
        self.assertEqual(self.gm("Gross margin | 58.5% | 0.0% | 58.5%", prev=58.0),
                         {"gaap": (58.5, 58.5), "nongaap": (58.5, 58.5)})


class GrossMarginGuidance(unittest.TestCase):
    def test_respectively(self):
        out = parse_gm(_norm("GAAP and non-GAAP gross margins are expected to be 70.6% and 71.0%, respectively, "
                             "plus or minus 50 basis points."))
        d = {b: (lo, hi) for b, _, lo, hi in out}
        self.assertAlmostEqual(d["gaap"][0], 70.1)
        self.assertAlmostEqual(d["nongaap"][1], 71.5)

    def test_percent_words(self):
        out = parse_gm(_norm("GAAP gross margin is expected to be 48.5 to 49.5 percent."))
        self.assertEqual(out[0][:2], ("gaap", "range"))
        self.assertEqual(out[0][2:], (48.5, 49.5))

    def test_both_range(self):      # ON 2011
        out = parse_gm(_norm("For the first quarter of 2011, we expect GAAP and non-GAAP gross margin of approximately "
                             "40 to 41 percent."))
        self.assertEqual(sorted(b for b, *_ in out), ["gaap", "nongaap"])


if __name__ == "__main__":
    unittest.main()
