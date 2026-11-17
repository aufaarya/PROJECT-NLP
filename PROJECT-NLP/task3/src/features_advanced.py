"""
features_advanced.py  -  Model C features: from headline vectors to daily signals
-------------------------------------------------------------------------------
A trading day has up to K=60 headline vectors; the classifier needs one fixed-length row.
With only ~830 training days, feeding 384/768 raw dimensions would massively over-fit, so
every representation is compressed into a small, interpretable set of daily features:

  pooling (Task 3 req. 1b-v)
    emb_pc{i}        PCA (fit on TRAIN days) of the salience-weighted daily mean vector.
                     weight = log(1 + NumSources): stories covered by many outlets count more.
  semantic signals
    anchor_{name}    mean cosine similarity of the day's headlines to a hand-written
                     "concept anchor" (e.g. 'military escalation', 'ceasefire talks',
                     'central bank rate hike'), minus its train mean. Directly interpretable.
    novelty          1 - cos(today's mean vector, mean of the previous 20 days' vectors):
                     a *semantic* news shock (Task 2 measured shocks only in counts).
    dispersion       1 - mean cos(headline, day centroid): one dominant story vs. scattered news.
  temporal decay (req. 1b-v)
    every daily feature can be replaced by an exponentially-weighted mean over past trading
    days (half-life h) so a big story keeps influencing the next days with decaying weight;
    h is chosen on VALIDATION.
  topics (req. 1b-ii)
    lda_{k}          LDA topic shares (12 topics, CountVectorizer, fit on TRAIN headlines)
    clu_{k}          BERTopic-style topics: k-means (k=12) on MiniLM vectors of TRAIN headlines,
                     each cluster named by its class-based TF-IDF terms; daily cluster shares.
  entities (req. 1b-iii)
    ent_{group}      log volume + share of headlines mentioning a nation / central bank /
                     economic-event group (gazetteer over headline + GDELT actor names).
    entemb_pc{i}     PCA of the mean vector of ONLY the entity-relevant headlines.
"""
import re

import numpy as np
import pandas as pd

from . import config

EPS = 1e-9

# Concept anchors: short headline-style descriptions of FX-relevant geopolitical states.
ANCHORS = {
    "escalation": "military escalation as troops launch missile strikes and invasion",
    "deescalation": "ceasefire agreed as leaders hold peace talks and sign deal",
    "sanctions_trade": "new sanctions and tariffs imposed in escalating trade war",
    "monetary": "central bank raises interest rates to fight inflation",
    "energy": "oil prices surge on supply disruption fears opec",
    "em_stress": "emerging market currency slumps as investors pull capital",
    "us_politics": "us congress government shutdown debt ceiling standoff",
    "china_taiwan": "china taiwan tensions military drills strait",
}


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def daily_pool(dates, X, weights, trading_dates):
    """Weighted mean of rows of X per date -> [n_days, d] (zeros for empty days)."""
    td = pd.DatetimeIndex(trading_dates)
    pos = td.get_indexer(pd.DatetimeIndex(dates))
    ok = pos >= 0
    out = np.zeros((len(td), X.shape[1]))
    wsum = np.zeros(len(td))
    np.add.at(out, pos[ok], X[ok] * weights[ok, None])
    np.add.at(wsum, pos[ok], weights[ok])
    has = wsum > 0
    out[has] /= wsum[has, None]
    return out, has


def embedding_daily_features(name, corpus, X, trading_dates, is_train_day, anchor_vecs=None,
                             n_pc=config.N_EMB_PCA):
    """Build the compact daily feature block for one representation."""
    from sklearn.decomposition import PCA
    td = pd.DatetimeIndex(trading_dates)
    w = np.log1p(corpus["max_sources"].fillna(1).to_numpy().astype(float))
    w = np.where(w > 0, w, 0.5)
    Xu = _unit(X)
    M, has = daily_pool(corpus["date"], Xu, w, td)
    feats = pd.DataFrame(index=td)

    # PCA of daily mean vectors (fit on TRAIN days that have news)
    fit = is_train_day & has
    pca = PCA(n_components=n_pc, random_state=config.RANDOM_STATE).fit(M[fit])
    P = pca.transform(M)
    for i in range(n_pc):
        feats[f"{name}_pc{i}"] = P[:, i]

    # novelty: distance of today's centroid to the trailing 20-day centroid
    Mu = _unit(M)
    roll = pd.DataFrame(M).shift(1).rolling(config.SHOCK_WINDOW, min_periods=5).mean().to_numpy()
    nov = 1 - (Mu * _unit(np.nan_to_num(roll))).sum(1)
    feats[f"{name}_novelty"] = np.where(np.isnan(roll).any(1), np.nan, nov)

    # dispersion: how spread the day's headlines are around their centroid
    pos = td.get_indexer(pd.DatetimeIndex(corpus["date"]))
    ok = pos >= 0
    cos_to_c = np.full(len(corpus), np.nan)
    cos_to_c[ok] = (Xu[ok] * Mu[pos[ok]]).sum(1)
    disp = pd.Series(1 - cos_to_c).groupby(pos).mean()
    feats[f"{name}_dispersion"] = pd.Series(disp.reindex(range(len(td))).to_numpy(), index=td)

    # concept anchors
    if anchor_vecs is not None:
        A = _unit(anchor_vecs)
        S = Xu @ A.T                                    # [n_headlines, n_anchors]
        Sd, _ = daily_pool(corpus["date"], S, w, td)
        for j, a in enumerate(ANCHORS):
            col = Sd[:, j]
            col = col - col[fit].mean()                 # centre on TRAIN mean
            feats[f"{name}_anchor_{a}"] = np.where(has, col, 0.0)

    feats.loc[~has] = feats.loc[~has].fillna(0.0)
    feats.index.name = "date"
    return feats.reset_index(), pca


def finbert_sentiment_daily(corpus, probs, trading_dates):
    """probs = [n, 3] (pos, neg, neu) -> daily mean pos / neg / net tone / share negative."""
    td = pd.DatetimeIndex(trading_dates)
    w = np.ones(len(corpus))
    net = (probs[:, 0] - probs[:, 1])[:, None]
    negshare = (probs[:, 1] > 0.5).astype(float)[:, None]
    M, has = daily_pool(corpus["date"], np.hstack([probs[:, :2], net, negshare]), w, td)
    f = pd.DataFrame(M, index=td, columns=["fb_pos", "fb_neg", "fb_net", "fb_neg_share"])
    f.index.name = "date"
    return f.reset_index()


# ----------------------------------------------------------------------------
# Topic models
# ----------------------------------------------------------------------------
def lda_topics(corpus, is_train_row, k=config.N_LDA_TOPICS):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words="english", min_df=10, max_df=0.3)
    Ctr = vec.fit_transform(corpus.loc[is_train_row, "headline"])
    lda = LatentDirichletAllocation(n_components=k, learning_method="online", batch_size=4096,
                                    max_iter=10, random_state=config.RANDOM_STATE, n_jobs=1)
    lda.fit(Ctr)
    T = lda.transform(vec.transform(corpus["headline"]))
    terms = np.array(vec.get_feature_names_out())
    top = {f"lda_{i}": list(terms[np.argsort(c)[::-1][:8]]) for i, c in enumerate(lda.components_)}
    return T, top


def embedding_clusters(corpus, X, is_train_row, k=config.N_LDA_TOPICS):
    """BERTopic-style: k-means on embeddings (fit on TRAIN) + class-based TF-IDF labels."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    Xu = _unit(X)
    km = KMeans(n_clusters=k, n_init=5, random_state=config.RANDOM_STATE).fit(Xu[is_train_row])
    lab = km.predict(Xu)
    # c-TF-IDF: concatenate all train headlines of a cluster into one "document"
    docs = (pd.Series(corpus.loc[is_train_row, "headline"].to_numpy())
            .groupby(lab[is_train_row]).apply(" ".join))
    cv = CountVectorizer(stop_words="english", min_df=2).fit(docs)
    tf = cv.transform(docs).toarray().astype(float)
    tf = tf / (tf.sum(1, keepdims=True) + EPS)
    idf = np.log(1 + tf.shape[0] / (1 + (tf > 0).sum(0)))
    ctfidf = tf * idf
    terms = np.array(cv.get_feature_names_out())
    top = {f"clu_{i}": list(terms[np.argsort(r)[::-1][:8]]) for i, r in zip(docs.index, ctfidf)}
    onehot = np.eye(k)[lab]
    return onehot, lab, top


def topic_share_daily(corpus, T, prefix, trading_dates):
    td = pd.DatetimeIndex(trading_dates)
    M, has = daily_pool(corpus["date"], T, np.ones(len(corpus)), td)
    f = pd.DataFrame(M, index=td, columns=[f"{prefix}_{i}" for i in range(T.shape[1])])
    f.index.name = "date"
    return f.reset_index()


# ----------------------------------------------------------------------------
# Entity filtering
# ----------------------------------------------------------------------------
def entity_matrix(df_text: pd.DataFrame) -> pd.DataFrame:
    """Boolean [n_headlines, n_groups]: does the headline (or a GDELT actor) mention the group?"""
    txt = (df_text["headline"].fillna("") + " " + df_text.get("actor1", "").fillna("").str.lower()
           + " " + df_text.get("actor2", "").fillna("").str.lower())
    out = {}
    for g, words in config.ENTITY_GROUPS.items():
        rx = re.compile(r"\b(?:" +"|".join(re.escape(w) for w in words) + r")\b")
        out[f"ent_{g}"] = txt.str.contains(rx)
    return pd.DataFrame(out, index=df_text.index)


def entity_daily_features(aligned, trading_dates):
    """Volume (log per calendar day) and share of each entity group, over ALL aligned headlines."""
    E = entity_matrix(aligned).astype(int)
    E["date"] = aligned["date"].to_numpy()
    E["n"] = 1
    d = E.groupby("date").sum().reindex(pd.DatetimeIndex(trading_dates), fill_value=0)
    out = pd.DataFrame(index=d.index)
    for g in config.ENTITY_GROUPS:
        out[f"ent_{g}_share"] = d[f"ent_{g}"] / d["n"].replace(0, np.nan)
        lv = np.log1p(d[f"ent_{g}"])
        hist = lv.shift(1).rolling(config.SHOCK_WINDOW, min_periods=config.SHOCK_WINDOW)
        out[f"ent_{g}_shock"] = ((lv - hist.mean()) / (hist.std() + EPS)).clip(-5, 5)
    out = out.fillna(0.0)
    out.index.name = "date"
    return out.reset_index()


# ----------------------------------------------------------------------------
# Temporal decay
# ----------------------------------------------------------------------------
def apply_decay(df: pd.DataFrame, cols, halflife):
    """Exponentially-weighted mean over past trading days (includes today, never the future)."""
    out = df.copy()
    if halflife and halflife > 0:
        out[cols] = out[cols].ewm(halflife=halflife, adjust=True).mean()
    return out
