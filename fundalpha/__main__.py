"""Command line: python -m fundalpha <command> [options]

  fetch        download/refresh EDGAR submissions + companyfacts for the universe (cached)
  build        rebuild filings, facts, quarterly vintages, earnings events
  releases     download the 8-K Item 2.02 earnings releases (immutable cache)
  guidance     extract management outlook from the cached releases
  features     build the forecast-origin feature panel
  backtest     walk-forward evaluation (--period dev|holdout|all)
  report       write the research report and dashboard
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fundalpha")
    ap.add_argument("command")
    ap.add_argument("--refresh-hours", type=float, default=24.0)
    ap.add_argument("--period", default="dev")
    ap.add_argument("--tickers", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.command == "fetch":
        from .pipeline import fetch
        fetch(args.refresh_hours)
    elif args.command == "build":
        from .pipeline import build
        build()
    elif args.command == "events":
        from .pipeline import events
        events()
    elif args.command == "releases":
        from .pipeline import releases
        releases(args.tickers.split(",") if args.tickers else None)
    elif args.command == "guidance":
        from .pipeline import guidance
        guidance(args.tickers.split(",") if args.tickers else None)
    elif args.command == "features":
        from .pipeline import features
        features()
    elif args.command == "backtest":
        from .pipeline import backtest
        backtest(args.period)
    elif args.command == "finalize":
        from .pipeline import finalize
        finalize(args.period)
    elif args.command == "dashboard":
        from .pipeline import dashboard_only
        dashboard_only()
    elif args.command == "report":
        from .pipeline import report
        report()
    else:
        ap.error(f"unknown command {args.command}")
    print(f"[{args.command}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
