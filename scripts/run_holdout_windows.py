"""Compute holdout-window predictions (2021Q1 onward) WITHOUT printing any metric.

The holdout must not be inspected before the model specification is frozen on development data; this script only
writes out/predictions_holdout_raw.pkl (checkpointed per window under out/wf_cache/) so the final evaluation can
start immediately once the dev review is done.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundalpha.paths import OUT  # noqa: E402
from fundalpha.walkforward import run  # noqa: E402

if __name__ == "__main__":
    panel = pd.read_pickle(OUT / "panel.pkl")
    pred = run(panel, start="2021Q1", n_jobs=4)
    pred.to_pickle(OUT / "predictions_holdout_raw.pkl")
    print(f"saved {len(pred):,} holdout prediction rows (metrics deliberately not computed)")
