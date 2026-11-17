"""
data_prep.py
------------
Loads the Task 1 outputs and turns them into a leakage-safe, trading-day dataset.

Why this step exists (a correction to the Task 1 table)
-------------------------------------------------------
Task 1's `daily_features_usd_target.csv` is on a *calendar* grid: the USD index was
forward-filled over weekends/holidays, so ~31% of rows have a "next-day change" of exactly
0 and were labelled DOWN (0). That inflates the DOWN class (only 34% UP) and makes a
"always predict DOWN" rule look like 66% accuracy. The FX market does not trade on those
days, so they are not real prediction targets.

Here we:
  1. keep only real trading days (dates where FRED has a value),
  2. bucket every calendar day's news onto the *next* trading day at or after it
     (weekend/holiday news is therefore added to Monday instead of being thrown away),
  3. define the target on trading-day log returns.

Timing convention (no look-ahead)
---------------------------------
Row t (a trading day) contains:
  * news published in (t_prev, t]            -> known by the end of day t
  * USD index up to and including P_t        -> known on day t
  * target  r_{t+1} = ln(P_{t+1} / P_t)       -> the NEXT trading day's move
So every feature on row t is available before the target period ends.
"""
import numpy as np
import pandas as pd

from . import config


def load_usd_trading_days(path=config.USD_RAW_FILE) -> pd.DataFrame:
    """FRED series -> one row per trading day (holidays, which FRED leaves blank, dropped)."""
    usd = pd.read_csv(path, parse_dates=["date"])
    usd["usd_index"] = pd.to_numeric(usd["usd_index"], errors="coerce")
    usd = usd.dropna(subset=["usd_index"]).sort_values("date").reset_index(drop=True)
    return usd


def load_daily_news_counts(path=config.NEWS_DAILY_FILE) -> pd.DataFrame:
    """Calendar-day article counts per geopolitical category (only the news columns are kept;
    the Task 1 target columns are rebuilt from scratch below)."""
    cols = ["date"] + config.CATEGORIES + ["total_articles"]
    news = pd.read_csv(path, parse_dates=["date"], usecols=cols)
    return news.sort_values("date").reset_index(drop=True)


def bucket_news_to_trading_days(news: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Assign each calendar day to the first trading day >= that day and sum the counts.

    Example: news from Sat, Sun and Mon all land on Monday's row, because Monday is the
    first moment a trader could act on them.
    """
    td = pd.DatetimeIndex(trading_dates).sort_values()
    pos = td.searchsorted(news["date"].values, side="left")
    valid = pos < len(td)                      # news after the last trading day is dropped
    news = news.loc[valid].copy()
    news["trade_date"] = td[pos[valid]]

    count_cols = config.CATEGORIES + ["total_articles"]
    agg = news.groupby("trade_date")[count_cols].sum()
    agg["n_calendar_days"] = news.groupby("trade_date").size()   # 1 normally, 3 on Mondays
    agg = agg.reindex(td, fill_value=0)
    agg.index.name = "date"
    return agg.reset_index()


def build_base_table() -> pd.DataFrame:
    """Trading-day table with USD level, returns, next-day targets and bucketed news counts."""
    usd = load_usd_trading_days()
    news = load_daily_news_counts()

    df = usd.copy()
    df["ret"] = np.log(df["usd_index"]).diff()                 # r_t  (known on day t)
    df["target_ret_next"] = df["ret"].shift(-1)                # r_{t+1}  (to predict)
    df["target_up_next"] = (df["target_ret_next"] > 0).astype(float)
    df.loc[df["target_ret_next"].isna(), "target_up_next"] = np.nan

    news_td = bucket_news_to_trading_days(news, df["date"])
    df = df.merge(news_td, on="date", how="left")

    # Only keep the period covered by the news collection (Task 1 news starts 2021-01-04).
    df = df[df["date"] >= news["date"].min()].reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame,
                        train_frac=config.TRAIN_FRAC,
                        val_frac=config.VAL_FRAC):
    """Strict time-ordered split: oldest 70% train, next 15% validation, newest 15% test.
    No shuffling, so no future observation can influence a past one."""
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    out = df.copy()
    out["split"] = "test"
    out.iloc[:i_train, out.columns.get_loc("split")] = "train"
    out.iloc[i_train:i_val, out.columns.get_loc("split")] = "val"
    return out
