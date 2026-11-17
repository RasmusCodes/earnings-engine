# earnings-engine

A point-in-time research platform for forecasting reported fundamentals of US-listed semiconductor companies and
testing the forecasts honestly. It reconstructs what was knowable at each moment from SEC EDGAR alone:

* every XBRL financial fact with the timestamp of the filing that first carried it (restatements become new vintages);
* the exact earnings-release time from each 8-K Item 2.02 filing (pre-market, intraday or after-hours);
* management's next-quarter revenue and gross-margin guidance, parsed from 3,198 earnings press releases.

On top of that it runs walk-forward forecasts of next-quarter revenue and GAAP gross margin for 50 companies,
13 of them since acquired (so the sample is not survivorship-biased). The forecasts are compared with random walks,
historical averages, simple linear baselines and management guidance. Model choices were made on 2013-2020; the
2021-2026 holdout was run once.

**Full write-up: [reports/RESEARCH_REPORT.md](reports/RESEARCH_REPORT.md)** · interactive explorer:
`reports/dashboard.html` (open locally; data embedded).

## Results in brief

| | 2013-20 | 2021-26 holdout |
|---|---|---|
| Revenue, history-only XGBoost vs random walk (mean absolute % error) | 7.03% vs 8.36% | 7.37% vs 9.29% |
| Management guidance midpoint | 3.59% | 3.32% |
| Revenue, random forest predicting the beat vs guidance | 3.08% | 2.30% |
| ... vs "guidance + market-wide average beat" | 3.26% | 2.60% |
| Beat-vs-miss direction: model / "always a beat" | 76% / 78% | 90% / 90% |
| GAAP gross margin, lasso vs random walk (mean absolute error) | 2.14 vs 2.27 pp | 2.05 vs 2.40 pp |

* Fitted models beat every history-based benchmark in both periods. All gains are significant in panel
  Diebold-Mariano tests clustered by reporting season.
* Management guidance is still about twice as accurate as anything built from history.
* Semiconductor companies beat their own guidance midpoint 82% of the time (median +1.7%). The model predicts the
  size and ranking of the beat well, but not its direction.
* No analyst consensus or price data is used yet, so none of this is evidence of an earnings-surprise or return signal.

## Quickstart

Python 3.12. The first run downloads about 1.1 GB from SEC EDGAR, throttled to 6 requests/s, and takes roughly an
hour end to end on a laptop.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your@email"         # Windows PowerShell: $env:SEC_USER_AGENT="Your Name your@email"
```

SEC requires automated clients to declare a contact in the User-Agent. The engine reads it from `SEC_USER_AGENT` at
run time, never writes it to disk, and contacts no host other than SEC EDGAR.

```bash
python -m fundalpha fetch        # EDGAR submissions + XBRL companyfacts
python -m fundalpha build        # filings, facts, point-in-time vintages, quarter calendar, release events
python -m fundalpha releases     # every 8-K Item 2.02 earnings release (EX-99 text), immutable cache
python -m fundalpha guidance     # management outlook: revenue, GAAP / non-GAAP gross margin
python -m fundalpha features     # forecast-origin feature panel + next-quarter ("live") rows
python -m fundalpha backtest --period dev           # walk-forward 2013-2020 (development)
python scripts/run_holdout_windows.py               # 2021-2026 windows, predictions only, no metrics printed
python -m fundalpha finalize --period holdout       # merge + pooled-beat benchmark, print holdout metrics
python scripts/robustness.py                        # per-company, ex-M&A, ex-NVDA, sub-industry, by-year checks
python -m fundalpha report       # figures, reports/tables.md, out/live_forecasts.csv, dashboard
python -m unittest discover -s tests                # parser tests always; point-in-time audits once data is built
```

Every download is cached, so reruns are offline and fast. The walk-forward checkpoints each window under
`out/wf_cache/`.

## How it works

| Layer | Rule |
|---|---|
| Financials (`fundalpha/pit.py`) | 10-Qs/10-Ks (and amendments) are replayed in EDGAR acceptance order; each derived quarterly value is stored with the acceptance time of the filing that produced it. Features use the latest vintage accepted before the forecast origin; targets use the first-reported value. Q4 and cash-flow quarters come from year-to-date facts. Derivations across restated bases are refused, and tagging errors (scale x1000, sign flips, duplicate period ends) are cleaned. |
| Release events (`events.py`) | First 8-K Item 2.02 after the quarter end, or the 10-Q if it came first. Forecast origin = the last 16:00 ET close before the release. Pre-announcements are flagged. |
| Guidance (`guidance.py`) | Prose, bullet and table formats: ranges, ± absolute/percent, sequential-percent guidance, low/mid/high columns, GAAP and non-GAAP gross margin. The latest outlook issued before the target quarter's origin is used, mid-quarter updates included. Parses are validated only against the quarter reported in the same release, never against the outcome. |
| Peers and industry (`features.py`, `industry.py`) | A peer's quarter counts only after its own release. Hyperscaler capex (MSFT, GOOGL, AMZN, META) uses the same vintage engine. |
| Models (`models.py`, `walkforward.py`) | Benchmarks (random walk, seasonal random walk, historical averages, guidance, guidance + company / Bayesian / market-wide beat), a pooled linear baseline, and OLS, ridge, lasso, Bayesian ridge, random forest, gradient boosting and XGBoost. Refit every calendar quarter on outcomes public before the window. |
| Evaluation (`evaluate.py`) | MAPE / MAE / RMSE, direction and beat-direction hit rates against the base rate, rank IC, and panel Diebold-Mariano tests with Newey-West errors across reporting seasons. |

`tests/test_pit.py` audits the built data: origins precede releases, guidance precedes origins, lagged values equal
the vintage known at the origin, no vintage of a target quarter exists before its origin, and training windows end
before test origins. `tests/test_guidance.py` pins the parser on real release phrasings.

## Universe

`config/universe.csv`: 37 currently listed large/mid caps and 13 acquired companies (Broadcom Corp, Xilinx, Altera,
Mellanox, Cavium, LSI, Maxim, Linear, Freescale, Microsemi, Cypress, IDT, SanDisk). Qorvo is marked acquired (Skyworks,
2026-10-05). Re-domiciled registrants are chained by CIK (Avago → Broadcom Ltd → Broadcom Inc; Marvell Ltd → Marvell
Inc). A company-quarter enters the evaluation only if trailing-four-quarter revenue known at the origin was ≥ $500M.
Add a company by appending a row with its CIK.

## Analyst consensus

None is bundled (it is licensed data). Point-in-time consensus snapshots dropped into `data/consensus/*.csv` (schema in
`fundalpha/consensus.py`, example header in `config/consensus_template.csv`) become benchmark `C0_consensus`
automatically. Only snapshots dated before each forecast origin are used.

## Repository layout

| Path | Contents |
|---|---|
| `fundalpha/` | the package (pipeline, PIT engine, guidance parser, features, models, walk-forward, evaluation, report, dashboard) |
| `config/` | universe and consensus template |
| `scripts/` | holdout-window runner, robustness checks |
| `tests/` | parser tests, point-in-time audits, consensus hook test |
| `reports/` | research report, figures, auto-generated metric tables, dashboard |
| `out/*.csv` | committed results: metrics, robustness, feature importance, beat history, live forecasts, feature panel |
| `data/`, `out/*.pkl`, `logs/` | rebuilt by the pipeline, not committed |

## Limitations

* Guidance covers about 78% of releases. Companies that gave their outlook only on the call have little or none (KLA
  before 2016, Micron before FY2020, Altera, Cavium, IDT).
* Gross-margin targets are GAAP; most companies guide non-GAAP, so the GM guidance benchmark covers only GAAP-basis
  outlooks.
* The 8-K acceptance time can trail the wire release by minutes.
* The live forecasts in `out/live_forecasts.csv` are a snapshot from 2026-11-14.

## Disclaimer

Research code and results, provided as-is for educational purposes. Nothing here is investment advice or a
recommendation to trade any security.

## License

MIT, see [LICENSE](LICENSE). Underlying financial data comes from public SEC EDGAR filings.
