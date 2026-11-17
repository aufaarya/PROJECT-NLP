"""
data_prep.py  (Task 3)
----------------------
Trading-day table for the Bank Indonesia JISDOR USD/IDR series, with the same leakage-safe
conventions as Task 2:

  row t (a BI trading day) holds
    * prices up to P_t and news aligned to day t (published before 16:00 WIB on t, or on the
      weekend / holiday / after-cut-off period that ends on t)      -> known at end of day t
    * targets for the NEXT trading day:
        target_ret_next     r_{t+1} = ln(P_{t+1}/P_t)                (regression)
        target_up_next      1[r_{t+1} > 0]  (USD strengthens vs IDR) (primary, classification)
        target_highvol_next 1[|r_{t+1}| > median |r| of TRAIN]       (secondary, volatility)

The chronological 70 / 15 / 15 split is identical in spirit to Task 2 (oldest -> newest, no
shuffling); market features re-use the Task 2 functions so Model A is the Task 2 baseline.
"""
import numpy as np
import pandas as pd

from . import config
from .task2.features import add_market_features, market_feature_cols  # noqa: F401  (re-export)


def load_bi() -> pd.DataFrame:
    bi = pd.read_csv(config.BI_FILE, parse_dates=["date"])
    bi = bi.dropna(subset=["usd_idr"]).sort_values("date").reset_index(drop=True)
    return bi[["date", "usd_idr"]]


def build_base_table() -> pd.DataFrame:
    df = load_bi()
    df["ret"] = np.log(df["usd_idr"]).diff()
    df["target_ret_next"] = df["ret"].shift(-1)
    df["target_up_next"] = (df["target_ret_next"] > 0).astype(float)
    df.loc[df["target_ret_next"].isna(), "target_up_next"] = np.nan
    # calendar days covered by each trading-day news bucket (Mon = Sat + Sun + Mon = 3)
    df["n_calendar_days"] = df["date"].diff().dt.days.fillna(1).clip(lower=1)
    # Task 1 daily GDELT event count (all sources, all events) -> extra volume signal
    t1 = pd.read_csv(config.TASK1_DAILY_FILE, usecols=["date", "event_count"],
                     parse_dates=["date"])
    df = df.merge(t1, on="date", how="left")
    df = add_market_features(df)
    return df


def chronological_split(df: pd.DataFrame, train_frac=config.TRAIN_FRAC,
                        val_frac=config.VAL_FRAC) -> pd.DataFrame:
    n = len(df)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    out = df.copy()
    out["split"] = "test"
    out.iloc[:i_tr, out.columns.get_loc("split")] = "train"
    out.iloc[i_tr:i_va, out.columns.get_loc("split")] = "val"
    return out


def add_volatility_target(df: pd.DataFrame) -> pd.DataFrame:
    """High-volatility label; threshold = median |r| on TRAIN only (no leakage)."""
    thr = df.loc[df["split"] == "train", "target_ret_next"].abs().median()
    out = df.copy()
    out["target_highvol_next"] = (out["target_ret_next"].abs() > thr).astype(float)
    out.attrs["highvol_threshold"] = float(thr)
    return out
