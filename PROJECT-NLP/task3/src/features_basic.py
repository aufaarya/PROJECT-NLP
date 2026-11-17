"""
features_basic.py  -  Model B: the Task 2 "basic NLP" recipe, now run on real text
--------------------------------------------------------------------------------
Task 2 defined (but could not run on article text) three classical feature families.
Here they are computed on the GDELT headline stream so Model B is exactly the Task 2
recipe and the Task 2 -> Task 3 ablation only changes the *representation*:

1. Lexicon category volume / mix / shock  (Task 2 lexicon, unchanged)
     counted over ALL aligned headlines of the day (volume needs the full stream)
2. VADER sentiment (mean compound, share negative, share positive)
3. TF-IDF (1-2 grams) -> TruncatedSVD / LSA, 10 components, fit on TRAIN headlines only
   2 + 3 are computed on the same top-K salient corpus the embeddings use.
"""
import re

import numpy as np
import pandas as pd

from . import config
from .task2.features import add_news_volume_features  # same shock / share / log maths

EPS = 1e-9


def _lexicon_regex(words):
    return re.compile(r"\b(?:" +"|".join(re.escape(w) for w in words) + r")", re.I)


def lexicon_daily_counts(aligned: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """Per trading day: number of headlines matching each Task 2 category (multi-label, a
    headline can count in several categories - Task 2 plan item 2) + total headlines."""
    h = aligned["headline"]
    out = pd.DataFrame({"date": aligned["date"]})
    for cat, words in config.CATEGORY_KEYWORDS.items():
        out[cat] = h.str.contains(_lexicon_regex(words)).astype(int)
    out["total_articles"] = 1
    daily = out.groupby("date").sum().reindex(pd.DatetimeIndex(trading_dates), fill_value=0)
    daily.index.name = "date"
    return daily.reset_index()


def add_lexicon_features(df: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    counts = lexicon_daily_counts(aligned, df["date"])
    out = df.merge(counts, on="date", how="left")
    out = add_news_volume_features(out)          # Task 2 function: log / share / shock
    # Task 1 all-source GDELT event count as one more volume + shock signal
    lv = np.log1p(out["event_count"].fillna(0) / out["n_calendar_days"])
    hist = lv.shift(1).rolling(config.SHOCK_WINDOW, min_periods=config.SHOCK_WINDOW)
    out["log_gdelt_events"] = lv
    out["shock_gdelt_events"] = ((lv - hist.mean()) / (hist.std() + EPS)).clip(-5, 5)
    return out


def lexicon_feature_cols():
    cats = config.CATEGORIES
    cols = [f"log_{c}" for c in cats + ["total_articles"]]
    cols += [f"share_{c}" for c in cats]
    cols += [f"shock_{c}" for c in cats + ["total_articles"]]
    return cols + ["log_gdelt_events", "shock_gdelt_events"]


def vader_scores(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def tfidf_lsa(corpus: pd.DataFrame, train_end, n_topics=config.N_LSA_TOPICS):
    """Returns (per-headline LSA matrix, top terms per component). Fit on TRAIN rows only."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    is_tr = corpus["date"] <= pd.Timestamp(train_end)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.5,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(corpus.loc[is_tr, "headline"])
    svd = TruncatedSVD(n_components=n_topics, random_state=config.RANDOM_STATE).fit(Xtr)
    Z = svd.transform(vec.transform(corpus["headline"]))
    terms = np.array(vec.get_feature_names_out())
    top = {f"lsa_{i}": list(terms[np.argsort(c)[::-1][:8]]) for i, c in enumerate(svd.components_)}
    return Z, top


def basic_text_daily(corpus: pd.DataFrame, trading_dates, train_end):
    """VADER + LSA aggregated per trading day (daily mean)."""
    c = corpus[["date", "headline"]].copy()
    c["vader"] = vader_scores(c["headline"].tolist())
    c["vader_neg"] = (c["vader"] <= -0.05).astype(float)
    c["vader_pos"] = (c["vader"] >= 0.05).astype(float)
    Z, top = tfidf_lsa(corpus, train_end)
    lsa_cols = [f"lsa_{i}" for i in range(Z.shape[1])]
    c[lsa_cols] = Z
    daily = c.groupby("date").agg(vader_mean=("vader", "mean"),
                                  vader_neg_share=("vader_neg", "mean"),
                                  vader_pos_share=("vader_pos", "mean"),
                                  **{k: (k, "mean") for k in lsa_cols})
    daily = daily.reindex(pd.DatetimeIndex(trading_dates)).fillna(0.0)
    daily.index.name = "date"
    return daily.reset_index(), top, c["vader"].to_numpy()


def basic_text_cols():
    return ["vader_mean", "vader_neg_share", "vader_pos_share"] + \
        [f"lsa_{i}" for i in range(config.N_LSA_TOPICS)]
