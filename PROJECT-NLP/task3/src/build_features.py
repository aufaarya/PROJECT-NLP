"""
build_features.py  -  one leakage-safe daily table with every feature family
--------------------------------------------------------------------------
    python -m src.build_features        (after python -m src.news_corpus)

Output: data/processed/{train,val,test}.csv, full_features.csv, feature_groups.json
        + data/interim/cache/*.npy (headline embeddings, reused by run_task3 / notebook)

Leakage rules (all unsupervised fits use TRAIN-period text only)
  * TF-IDF/LSA vocabulary + SVD, LDA, k-means topics, Word2Vec, every PCA -> TRAIN only.
  * shock / novelty windows look strictly backwards (shift(1)).
  * pre-trained encoders (GloVe, MiniLM, FinBERT) are frozen; they are never fine-tuned on
    our labels, so they cannot leak the target.
"""
import json

import numpy as np
import pandas as pd

from . import config
from . import embeddings as E
from . import features_advanced as FA
from .data_prep import add_volatility_target, build_base_table, chronological_split
from .features_basic import add_lexicon_features, basic_text_cols, basic_text_daily, \
    lexicon_feature_cols
from .task2.features import market_feature_cols

CAMEO_ROOT = {"10": "demands", "11": "criticizes", "12": "rejects", "13": "threatens",
              "14": "protests", "15": "mobilizes forces against", "16": "cuts ties with",
              "17": "coerces", "18": "assaults", "19": "fights", "20": "commits mass violence"}


def context_text(c: pd.DataFrame) -> pd.Series:
    """Headline enriched with the GDELT-coded event (who did what to whom). This is our
    'longer text' variant, because GDELT provides no article body."""
    verb = c["root_code"].astype(str).map(CAMEO_ROOT).fillna("acts on")
    a1 = c["actor1"].fillna("").str.lower()
    a2 = c["actor2"].fillna("").str.lower()
    return c["headline"] + ". " + (a1 + " " + verb + " " + a2).str.strip()


def main(verbose=True):
    base = build_base_table()
    aligned = pd.read_parquet(config.DATA_INTERIM / "headlines_aligned.parquet")
    corpus = pd.read_parquet(config.CORPUS_FILE)

    # ---- Model A + B tabular features, warm-up removal, split -----------------------
    df = add_lexicon_features(base, aligned)
    need = market_feature_cols() + lexicon_feature_cols() + ["target_ret_next"]
    df = df.dropna(subset=need).reset_index(drop=True)
    df = chronological_split(df)
    df = add_volatility_target(df)
    td = df["date"]
    train_end = df.loc[df["split"] == "train", "date"].max()
    is_train_day = (df["split"] == "train").to_numpy()
    is_tr_row = (corpus["date"] <= train_end).to_numpy()
    if verbose:
        print(f"days {len(df)}  train_end {train_end.date()}  corpus rows {len(corpus):,}")

    groups = {"market": market_feature_cols(), "lexicon": lexicon_feature_cols()}

    # ---- Model B: VADER + TF-IDF/LSA ------------------------------------------------
    bt, lsa_terms, vader = basic_text_daily(corpus, td, train_end)
    df = df.merge(bt, on="date", how="left")
    groups["basic_text"] = basic_text_cols()

    # ---- Headline embeddings (cached) -----------------------------------------------
    texts = corpus["headline"].tolist()
    anchors = list(FA.ANCHORS.values())
    reps = {}
    tr_texts = aligned.loc[aligned["date"] <= train_end, "headline"]
    tr_texts = tr_texts.sample(min(300_000, len(tr_texts)), random_state=config.RANDOM_STATE)
    if (config.CACHE / "w2v.npy").exists() and (config.CACHE / "w2v_anchors.npy").exists():
        reps["w2v"] = (np.load(config.CACHE / "w2v.npy"), np.load(config.CACHE / "w2v_anchors.npy"))
    else:
        print("  training Word2Vec on", len(tr_texts), "train headlines")
        Xw, (w2i, vec, freq, pc) = E.w2v_headline_vectors(texts, tr_texts.tolist())
        Aw = E.sif_encode(anchors, w2i, vec, freq, pc=pc)
        np.save(config.CACHE / "w2v.npy", Xw)
        np.save(config.CACHE / "w2v_anchors.npy", Aw)
        reps["w2v"] = (Xw, Aw)
    enc = {
        "glove": lambda t: E.encode_sentence_transformer(t, config.GLOVE_MODEL, normalize=False),
        "minilm": lambda t: E.encode_sentence_transformer(t, config.GENERAL_ST_MODEL),
        "finbert": lambda t: E.encode_finbert(t),
    }
    for name, fn in enc.items():
        X = E.cached(name, fn, texts)
        A = E.cached(f"{name}_anchors", fn, anchors)
        reps[name] = (X, A)
    Xctx = E.cached("minilm_ctx", enc["minilm"], context_text(corpus).tolist())
    reps["minilm_ctx"] = (Xctx, reps["minilm"][1])

    # FinBERT: split embedding / sentiment probabilities
    fb_X, fb_A = reps["finbert"]
    fb_probs = fb_X[:, -3:]
    reps["finbert"] = (fb_X[:, :-3], fb_A[:, :-3])

    for name, (X, A) in reps.items():
        f, _ = FA.embedding_daily_features(name, corpus, X, td, is_train_day, A)
        df = df.merge(f, on="date", how="left")
        groups[name] = [c for c in f.columns if c != "date"]
        if verbose:
            print(f"  {name:11s} dim {X.shape[1]:4d} -> {len(groups[name])} daily features")
    fbs = FA.finbert_sentiment_daily(corpus, fb_probs, td)
    df = df.merge(fbs, on="date", how="left")
    groups["finbert"] = groups["finbert"] + ["fb_pos", "fb_neg", "fb_net", "fb_neg_share"]

    # ---- Topics: LDA and BERTopic-style embedding clusters --------------------------
    T, lda_terms = FA.lda_topics(corpus, is_tr_row)
    df = df.merge(FA.topic_share_daily(corpus, T, "lda", td), on="date", how="left")
    Ok, clu_lab, clu_terms = FA.embedding_clusters(corpus, reps["minilm"][0], is_tr_row)
    df = df.merge(FA.topic_share_daily(corpus, Ok, "clu", td), on="date", how="left")
    groups["topics"] = [f"lda_{i}" for i in range(T.shape[1])] + \
                       [f"clu_{i}" for i in range(Ok.shape[1])]
    np.save(config.CACHE / "corpus_lda.npy", T.astype(np.float32))
    np.save(config.CACHE / "corpus_cluster.npy", clu_lab)

    # ---- Entities: gazetteer over headline + GDELT actors ----------------------------
    ent = FA.entity_daily_features(aligned, td)
    df = df.merge(ent, on="date", how="left")
    Ec = FA.entity_matrix(corpus)
    rel = Ec.drop(columns=["ent_economy"]).any(axis=1).to_numpy()   # nations / central banks
    f, _ = FA.embedding_daily_features("entmini", corpus[rel].reset_index(drop=True),
                                       reps["minilm"][0][rel], td, is_train_day, None)
    df = df.merge(f, on="date", how="left")
    groups["entity"] = [c for c in ent.columns if c != "date"] + \
                       [c for c in f.columns if c != "date"]

    text_cols = [c for g, cols in groups.items() if g != "market" for c in cols]
    df[text_cols] = df[text_cols].fillna(0.0)

    # ---- Save -------------------------------------------------------------------------
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PROCESSED / "full_features.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(config.DATA_PROCESSED / f"{s}.csv", index=False)
    meta = {"groups": groups, "train_end": str(train_end.date()),
            "highvol_threshold": float(df.loc[df["split"] == "train", "target_ret_next"]
                                       .abs().median()),
            "lsa_terms": lsa_terms, "lda_terms": lda_terms, "cluster_terms": clu_terms,
            "anchors": FA.ANCHORS, "entity_relevant_share": float(rel.mean())}
    with open(config.DATA_PROCESSED / "feature_groups.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    # per-headline annotations used later by the analysis notebook
    ann = corpus[["date", "headline", "source", "max_sources", "rank_in_day"]].copy()
    ann["vader"] = vader
    ann["fb_pos"], ann["fb_neg"] = fb_probs[:, 0], fb_probs[:, 1]
    ann["cluster"] = clu_lab
    ann["lda_topic"] = T.argmax(1)
    ann.to_parquet(config.DATA_INTERIM / "headline_annotations.parquet", index=False)
    if verbose:
        print({g: len(c) for g, c in groups.items()})
    return df, meta


if __name__ == "__main__":
    main()
