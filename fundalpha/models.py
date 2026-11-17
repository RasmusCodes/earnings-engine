"""Forecasting models: formula benchmarks (no fitting), simple linear baselines and ML models.

All models predict a *change* target so that every forecast is anchored on the last known quarter:
  revenue : y_rev  = log(rev_t / rev_{t-1})                -> rev_hat = rev_{t-1} * exp(y_hat)
  guided  : y_beat = log(rev_t / guidance_mid_t)           -> rev_hat = guidance_mid_t * exp(y_hat)
  margin  : y_gm   = gm_t - gm_{t-1} (percentage points)   -> gm_hat  = gm_{t-1} + y_hat
Hyperparameters are fixed a priori (documented here, not tuned on test data). Ridge/Lasso choose their penalty by
cross-validation inside each training window only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SUBINDUSTRIES = ["digital", "rf_mobile", "analog_mcu_power", "memory", "equipment_materials", "optical"]

REV_HIST = ["g_seq1", "g_seq2", "g_seq3", "g_yoy1", "g_yoy2", "seas_g1", "seas_g_avg", "g_mean4", "g_mean8",
            "g_std8", "days_ratio", "days_ratio_l4", "log_rev_l1", "inv_to_rev", "inv_to_rev_chg", "dso", "defrev_g",
            "goodwill_jump", "opm_l1", "gm_l1", "d_gm1", "peer_n", "peer_yoy_med", "peer_yoy_accel", "peer_seq_med",
            "grp_yoy_accel", "hyper_capex_yoy", "hyper_capex_accel"]
REV_GUID = REV_HIST + ["guid_g", "guid_width", "guid_is_point", "guid_is_pct", "guid_is_update", "beat_n",
                       "beat_mean8", "beat_med8", "beat_last", "beat_std8", "peer_beat_mean", "grp_beat_mean"]
GM_HIST = ["gm_l1", "gm_l2", "gm_l4", "gm_l5", "d_gm1", "gm_mean4", "gm_mean8", "gm_seas1", "gm_seas_avg",
           "g_seq1", "g_yoy1", "seas_g1", "days_ratio", "inv_to_rev", "inv_to_rev_chg", "opm_l1", "goodwill_jump",
           "peer_yoy_accel", "peer_seq_med"]
GM_GUID = GM_HIST + ["has_guid", "guid_g", "guid_gm_gaap_d", "guid_gm_delta", "gm_beat_gaap_mean8",
                     "peer_gm_beat_gaap", "peer_beat_mean"]


def design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float).copy()
    for q in (1, 2, 3, 4):
        X[f"fq_{q}"] = (df["fiscal_qtr"] == q).astype(float)
    for s in SUBINDUSTRIES:
        X[f"sub_{s}"] = (df["subindustry"] == s).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- formula benchmarks
def bench_revenue(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Predictions of y_rev = log(rev_t / rev_{t-1})."""
    out = {
        "B0_random_walk": pd.Series(0.0, index=df.index),
        "B1_seasonal_rw": df["seas_g1"],                                  # last year's same-quarter seq growth
        "B2_hist_avg_growth": df["g_mean8"],                              # mean of last 8 seq growth rates
        "B3_hist_seasonal_avg": df["seas_g_avg"],                         # mean same-quarter seq growth, 3 years
    }
    out["G0_guidance_mid"] = df["guid_g"].where(df["has_guid"])
    out["G1_guidance+hist_beat"] = (df["guid_g"] + df["beat_mean8"].fillna(0.0)).where(df["has_guid"])
    if "cons_revenue" in df and df["cons_revenue"].notna().any():          # licensed consensus, when supplied
        out["C0_consensus"] = np.log(df["cons_revenue"] / df["rev_l1"])
    return out


def bench_gm(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Predictions of y_gm = gm_t - gm_{t-1} (pp)."""
    return {
        "B0_random_walk": pd.Series(0.0, index=df.index),
        "B1_seasonal_rw": df["gm_seas1"],
        "B2_hist_avg_level": df["gm_mean4"] - df["gm_l1"],
        "B3_hist_seasonal_avg": df["gm_seas_avg"],
        "G0_gaap_guidance_mid": df["guid_gm_gaap_d"],
    }


# ----------------------------------------------------------------------------- fitted models
class Winsorized:
    """Clip the training target at its 1st/99th percentile before fitting (test targets are never touched)."""

    def __init__(self, est, q=(0.01, 0.99)):
        self.est, self.q = est, q

    def fit(self, X, y):
        lo, hi = np.nanquantile(y, self.q)
        self.est.fit(X, np.clip(y, lo, hi))
        return self

    def predict(self, X):
        return self.est.predict(X)


def _lin(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), est)


def make_models(kind: str) -> dict:
    """kind: 'baseline' (few-feature linear) or 'full'."""
    import xgboost as xgb
    if kind == "baseline":
        return {"L1_linear_baseline": Winsorized(_lin(LinearRegression()))}
    return {
        "M1_ols": Winsorized(_lin(LinearRegression())),
        "M2_ridge": Winsorized(_lin(RidgeCV(alphas=np.logspace(-2, 4, 25)))),
        "M3_lasso": Winsorized(_lin(LassoCV(alphas=40, cv=TimeSeriesSplit(4), max_iter=20000, random_state=0))),
        "M4_bayes_ridge": Winsorized(_lin(BayesianRidge())),
        "M5_random_forest": Winsorized(make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.4, n_jobs=1,
                                  random_state=0))),
        "M6_gbm": Winsorized(HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=300,
                                                           min_samples_leaf=25, l2_regularization=1.0,
                                                           random_state=0)),
        "M7_xgboost": Winsorized(xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                                  reg_lambda=1.0, n_jobs=1, random_state=0, verbosity=0)),
    }


BASELINE_REV = ["g_seq1", "g_yoy1", "seas_g1"]
BASELINE_GM = ["d_gm1", "gm_seas1", "gm_l1", "gm_mean8"]
