"""
features.py
-----------
Turns text-derived news data and the USD series into model features.

Three feature families
----------------------
1. MARKET features (history of the exchange rate only) -> used by the baseline.
2. NEWS-VOLUME features from the Task 1 keyword lexicon (always available):
     * log1p count per category          -> how much geopolitical news of each type
     * share of each category in total   -> the *mix* of news, independent of volume
     * "news shock" z-score per category -> today's volume vs. its own past 20-day norm
   The shock z-score is the key design choice: markets react to *surprises*, not to the
   fact that the NYT writes ~28 war articles every day. It is computed with a
   shift(1) rolling window, so day t is compared only with days strictly before it.
3. TEXT features from article titles/abstracts (optional, needs the article-level file):
     * VADER sentiment (lexicon-based)            -> tone of the day's geopolitical news
     * TF-IDF on titles -> TruncatedSVD (LSA)     -> which *topics* dominate the day
   The TF-IDF vocabulary and SVD are fit on TRAIN-period articles only (no leakage).

No pre-trained embeddings or transformer models are used (course rule).
"""
import numpy as np
import pandas as pd

from . import config

EPS = 1e-9


# ----------------------------------------------------------------------------
# 1. Market features
# ----------------------------------------------------------------------------
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in range(config.RETURN_LAGS):
        out[f"ret_lag{k}"] = out["ret"].shift(k)            # ret_lag0 = r_t (today)
    for w in config.VOL_WINDOWS:
        out[f"vol_{w}"] = out["ret"].rolling(w).std()
    out["mom_5"] = out["ret"].rolling(5).sum()               # 1-week momentum
    out["mom_20"] = out["ret"].rolling(20).sum()             # 1-month momentum
    return out


def market_feature_cols():
    cols = [f"ret_lag{k}" for k in range(config.RETURN_LAGS)]
    cols += [f"vol_{w}" for w in config.VOL_WINDOWS]
    cols += ["mom_5", "mom_20"]
    return cols


# ----------------------------------------------------------------------------
# 2. News-volume features (lexicon category counts)
# ----------------------------------------------------------------------------
def add_news_volume_features(df: pd.DataFrame, window=config.SHOCK_WINDOW) -> pd.DataFrame:
    out = df.copy()
    cats = config.CATEGORIES
    # Per-calendar-day average, so a Monday bucket (Sat+Sun+Mon) is comparable to a Tuesday.
    ndays = out["n_calendar_days"].clip(lower=1)

    for c in cats + ["total_articles"]:
        per_day = out[c] / ndays
        out[f"log_{c}"] = np.log1p(per_day)
        # Past-only baseline: mean/std of the previous `window` trading days (shift(1)).
        hist = out[f"log_{c}"].shift(1).rolling(window, min_periods=window)
        out[f"shock_{c}"] = (out[f"log_{c}"] - hist.mean()) / (hist.std() + EPS)
        # Rare categories (sanctions, diplomacy) can have a zero std window -> clip extremes.
        out[f"shock_{c}"] = out[f"shock_{c}"].clip(-5, 5)

    total = out["total_articles"].replace(0, np.nan)
    for c in cats:
        out[f"share_{c}"] = (out[c] / total).fillna(0)
    return out


def news_volume_feature_cols():
    cats = config.CATEGORIES
    cols = [f"log_{c}" for c in cats + ["total_articles"]]
    cols += [f"share_{c}" for c in cats]
    cols += [f"shock_{c}" for c in cats + ["total_articles"]]
    return cols


# ----------------------------------------------------------------------------
# 3. Text features from the article-level file (VADER + TF-IDF/LSA)
# ----------------------------------------------------------------------------
def build_text_features(articles: pd.DataFrame,
                        trading_dates: pd.Series,
                        train_end_date,
                        n_topics: int = 10,
                        text_col: str = "title",
                        min_df: int = 5):
    """Aggregate article-level text into one feature row per trading day.

    Parameters
    ----------
    articles : DataFrame with at least ['date', text_col] (Task 1 output
               `nyt_geopolitical_articles_clean.csv`; 'abstract' is used by VADER if present).
    trading_dates : the trading-day calendar the models use.
    train_end_date : last date of the TRAIN split. TF-IDF + SVD are fit only on articles
                     up to this date, so val/test vocabulary cannot leak into training.

    Returns
    -------
    DataFrame indexed by trading date with columns:
        vader_mean, vader_neg_share, vader_pos_share, topic_0..topic_{k-1}
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    art = articles.copy()
    art["date"] = pd.to_datetime(art["date"]).dt.tz_localize(None).dt.normalize()
    art[text_col] = art[text_col].fillna("").astype(str)

    # --- map each article to the first trading day >= its publication date ---
    td = pd.DatetimeIndex(trading_dates).sort_values()
    pos = td.searchsorted(art["date"].values, side="left")
    keep = pos < len(td)
    # Articles long before the first modelled day would otherwise all pile onto day 0.
    keep &= art["date"].values > (td[0] - pd.Timedelta(days=4)).to_datetime64()
    art = art.loc[keep].copy()
    pos = td.searchsorted(art["date"].values, side="left")
    keep = np.ones(len(art), dtype=bool)
    art["trade_date"] = td[pos[keep]]

    # --- (a) lexicon sentiment: VADER on title (+ abstract when available) ---
    sia = SentimentIntensityAnalyzer()
    sent_text = art[text_col]
    if "abstract" in art.columns:
        sent_text = sent_text + ". " + art["abstract"].fillna("").astype(str)
    art["vader"] = [sia.polarity_scores(t)["compound"] for t in sent_text]
    art["vader_neg"] = (art["vader"] <= -0.05).astype(float)
    art["vader_pos"] = (art["vader"] >= 0.05).astype(float)
    sent = art.groupby("trade_date").agg(
        vader_mean=("vader", "mean"),
        vader_neg_share=("vader_neg", "mean"),
        vader_pos_share=("vader_pos", "mean"),
    )

    # --- (b) TF-IDF on titles -> LSA topics, fit on TRAIN period only ---
    is_train = art["date"] <= pd.Timestamp(train_end_date)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=min_df, max_df=0.5, sublinear_tf=True)
    X_train = vec.fit_transform(art.loc[is_train, text_col])
    svd = TruncatedSVD(n_components=n_topics, random_state=config.RANDOM_STATE)
    svd.fit(X_train)
    topics = svd.transform(vec.transform(art[text_col]))
    topic_cols = [f"topic_{i}" for i in range(n_topics)]
    topic_df = pd.DataFrame(topics, columns=topic_cols, index=art.index)
    topic_df["trade_date"] = art["trade_date"]
    topic_daily = topic_df.groupby("trade_date")[topic_cols].mean()

    feats = sent.join(topic_daily, how="outer").reindex(td).fillna(0.0)
    feats.index.name = "date"

    # Top terms per topic, so the team can name/interpret each LSA component in the report.
    terms = np.array(vec.get_feature_names_out())
    top_terms = {f"topic_{i}": list(terms[np.argsort(comp)[::-1][:8]])
                 for i, comp in enumerate(svd.components_)}
    return feats.reset_index(), top_terms


def text_feature_cols(n_topics: int = 10):
    return ["vader_mean", "vader_neg_share", "vader_pos_share"] + \
           [f"topic_{i}" for i in range(n_topics)]
