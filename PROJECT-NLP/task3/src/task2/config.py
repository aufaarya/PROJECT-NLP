"""
config.py
---------
Central place for every path, constant and design decision the pipeline uses.
Changing a choice here (split ratios, rolling window, categories) changes it everywhere,
so the report and the code can't silently drift apart.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
# Task 3 repo layout: the Task 2 (FRED DTWEXBGS + NYT counts) experiment is kept unchanged
# under data/task2_fred_nyt, results/task2 and reports/task2 so it can be re-run for reference.
DATA_RAW = ROOT / "data" / "task2_fred_nyt" / "raw"
DATA_PROCESSED = ROOT / "data" / "task2_fred_nyt" / "processed"
RESULTS = ROOT / "results" / "task2"
FIGURES = ROOT / "reports" / "task2" / "figures"

USD_RAW_FILE = DATA_RAW / "usd_index_raw.csv"                    # FRED DTWEXBGS (Task 1)
NEWS_DAILY_FILE = DATA_RAW / "daily_features_usd_target.csv"     # daily category counts (Task 1)
# Article-level file produced by Task 1 notebook, section 2. Optional: if it is placed in
# data/raw/, run_pipeline.py automatically adds the text features (VADER + TF-IDF/LSA).
ARTICLES_FILE = DATA_RAW / "nyt_geopolitical_articles_clean.csv"

# ----------------------------------------------------------------------------
# Geopolitical categories (identical to the Task 1 keyword lexicon)
# ----------------------------------------------------------------------------
CATEGORIES = [
    "military_conflict",
    "sanctions",
    "diplomacy",
    "trade_conflict",
    "political_crisis",
]

# Keyword lexicon used in Task 1 to assign an article to a category.
# Kept here for documentation / for re-categorising the article file.
CATEGORY_KEYWORDS = {
    "military_conflict": ["war", "conflict", "military", "invasion", "attack", "nato"],
    "sanctions": ["sanction", "embargo", "trade restriction"],
    "diplomacy": ["diplomatic", "negotiation", "peace talk", "treaty"],
    "trade_conflict": ["trade war", "tariff", "trade dispute"],
    "political_crisis": ["political crisis", "coup", "election", "government crisis"],
}

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
SHOCK_WINDOW = 20        # trading days (~1 month) used as the "normal news level" baseline
RETURN_LAGS = 5          # r_t ... r_{t-4} as market features
VOL_WINDOWS = (5, 20)    # rolling std of returns

# ----------------------------------------------------------------------------
# Chronological split (fractions of usable trading days, in time order)
# ----------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # test = remaining 0.15

RANDOM_STATE = 42
