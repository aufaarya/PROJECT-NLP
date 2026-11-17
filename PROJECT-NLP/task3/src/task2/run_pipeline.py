"""
run_pipeline.py
---------------
End-to-end Task 2 run:  python -m src.run_pipeline   (from the repo root)

  1. build the trading-day table from Task 1 outputs          (data_prep)
  2. extract market + NLP features                            (features)
  3. strict chronological train / val / test split            -> data/processed/
  4. baseline models (history only) and combined models       (models)
  5. evaluate on VAL (model selection) and TEST (held out)    -> results/
"""
import json

import numpy as np
import pandas as pd

from . import config
from .data_prep import build_base_table, chronological_split
from .evaluate import classification_metrics, regression_metrics
from .features import (add_market_features, add_news_volume_features, build_text_features,
                       market_feature_cols, news_volume_feature_cols, text_feature_cols)
from .models import arima_walk_forward, select_arima_order, tune_and_fit_classifier


def build_dataset(verbose=True):
    df = build_base_table()
    df = add_market_features(df)
    df = add_news_volume_features(df)

    feat_market = market_feature_cols()
    feat_news = news_volume_feature_cols()
    needed = feat_market + feat_news + ["target_ret_next", "target_up_next"]
    df = df.dropna(subset=needed).reset_index(drop=True)   # drop warm-up rows + last day
    df = chronological_split(df)

    feature_sets = {"market": feat_market, "market+news": feat_market + feat_news}

    # Optional: article-level text features (only if the Task 1 article file is present)
    text_info = None
    if config.ARTICLES_FILE.exists():
        arts = pd.read_csv(config.ARTICLES_FILE)
        train_end = df.loc[df["split"] == "train", "date"].max()
        text_df, top_terms = build_text_features(arts, df["date"], train_end)
        df = df.merge(text_df, on="date", how="left")
        feature_sets["market+news+text"] = feat_market + feat_news + text_feature_cols()
        text_info = top_terms
        if verbose:
            print("Text features added from", config.ARTICLES_FILE.name)
    elif verbose:
        print(f"[info] {config.ARTICLES_FILE.name} not found -> text features skipped; "
              "using lexicon category-count features only.")
    return df, feature_sets, text_info


def save_splits(df):
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(config.DATA_PROCESSED / f"{s}.csv", index=False)
    df.to_csv(config.DATA_PROCESSED / "full_features.csv", index=False)


def run(verbose=True):
    df, feature_sets, text_info = build_dataset(verbose)
    save_splits(df)
    train, val, test = (df[df["split"] == s].reset_index(drop=True)
                        for s in ("train", "val", "test"))
    m_tr = (df["split"] == "train").to_numpy()
    m_trva = df["split"].isin(["train", "val"]).to_numpy()
    idx_val = np.where(df["split"] == "val")[0]
    idx_te = np.where(df["split"] == "test")[0]

    split_info = {s: {"rows": int((df["split"] == s).sum()),
                      "start": str(df.loc[df["split"] == s, "date"].min().date()),
                      "end": str(df.loc[df["split"] == s, "date"].max().date()),
                      "up_share": float(df.loc[df["split"] == s, "target_up_next"].mean())}
                  for s in ("train", "val", "test")}
    if verbose:
        print(json.dumps(split_info, indent=2))

    rows_val, rows_test = [], []
    preds = pd.DataFrame({"date": test["date"], "y_ret": test["target_ret_next"],
                          "y_up": test["target_up_next"]})

    def add(track, model, feats, split, metrics):
        (rows_val if split == "val" else rows_test).append(
            {"track": track, "model": model, "features": feats, **metrics})

    # ---------------- naive benchmarks ----------------
    for split, d in (("val", val), ("test", test)):
        add("regression", "RandomWalk (0)", "none", split,
            regression_metrics(d["target_ret_next"], np.zeros(len(d))))
        fit_rows = train if split == "val" else pd.concat([train, val])
        maj = int(fit_rows["target_up_next"].mean() >= 0.5)
        add("classification", "Majority", "none", split,
            classification_metrics(d["target_up_next"], np.full(len(d), maj)))
        add("classification", "Persistence", "market", split,
            classification_metrics(d["target_up_next"], (d["ret"] > 0).astype(int)))

    # ---------------- ARIMA baseline ----------------
    order, aic_table = select_arima_order(train)
    aic_table.to_csv(config.RESULTS / "arima_order_search.csv", index=False)
    if verbose:
        print("ARIMA order (AIC on train):", order)
    p_val, _ = arima_walk_forward(df, m_tr, order)
    p_te, res_arima = arima_walk_forward(df, m_trva, order)
    for split, p, d in (("val", p_val[idx_val], val), ("test", p_te[idx_te], test)):
        add("regression", f"ARIMA{order}", "market", split,
            regression_metrics(d["target_ret_next"], p))
        add("classification", f"ARIMA{order}-sign", "market", split,
            classification_metrics(d["target_up_next"], (p > 0).astype(int)))
    preds["ARIMA"] = p_te[idx_te]

    # ---------------- ARIMAX (combined) ----------------
    cats = config.CATEGORIES + ["total_articles"]
    exog_candidates = {
        "news shock z": [f"shock_{c}" for c in cats],
        "news log-volume": [f"log_{c}" for c in cats],
        "news all": news_volume_feature_cols(),
    }
    best_ex = None
    for name, cols in exog_candidates.items():
        p, _ = arima_walk_forward(df, m_tr, order, exog_cols=cols)
        rmse = regression_metrics(val["target_ret_next"], p[idx_val])["RMSE_bp"]
        if best_ex is None or rmse < best_ex[1]:
            best_ex = (name, rmse, p)
    ex_name, _, p_val_x = best_ex
    p_te_x, res_arimax = arima_walk_forward(df, m_trva, order,
                                            exog_cols=exog_candidates[ex_name])
    for split, p, d in (("val", p_val_x[idx_val], val), ("test", p_te_x[idx_te], test)):
        add("regression", f"ARIMAX{order}", ex_name, split,
            regression_metrics(d["target_ret_next"], p))
        add("classification", f"ARIMAX{order}-sign", ex_name, split,
            classification_metrics(d["target_up_next"], (p > 0).astype(int)))
    preds["ARIMAX"] = p_te_x[idx_te]
    params = pd.Series(res_arimax.params, index=res_arimax.param_names)
    pvals = pd.Series(res_arimax.pvalues, index=res_arimax.param_names)
    arimax_coefs = pd.DataFrame({"coef": params, "p_value": pvals})
    arimax_coefs.to_csv(config.RESULTS / "arimax_coefficients.csv")

    # ---------------- ML classifiers: baseline vs combined ----------------
    chosen = {}
    for model_name in ("LogReg", "GBM"):
        for fs_name, cols in feature_sets.items():
            out = tune_and_fit_classifier(model_name, cols, train, val, test)
            chosen[f"{model_name}|{fs_name}"] = out["params"]
            for split, proba, d in (("val", out["val_proba"], val),
                                    ("test", out["test_proba"], test)):
                add("classification", model_name, fs_name, split,
                    classification_metrics(d["target_up_next"], (proba >= 0.5).astype(int),
                                           proba))
            preds[f"{model_name}|{fs_name}"] = out["test_proba"]

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    mv = pd.DataFrame(rows_val)
    mt = pd.DataFrame(rows_test)
    mv.to_csv(config.RESULTS / "metrics_val.csv", index=False)
    mt.to_csv(config.RESULTS / "metrics_test.csv", index=False)
    preds.to_csv(config.RESULTS / "predictions_test.csv", index=False)

    summary = {"splits": split_info, "arima_order": list(order),
               "arimax_exog_set": ex_name, "classifier_params": chosen,
               "feature_sets": feature_sets, "text_topics": text_info}
    with open(config.RESULTS / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        pd.set_option("display.width", 200)
        print("\n=== VALIDATION ===\n", mv.round(4).to_string(index=False))
        print("\n=== TEST ===\n", mt.round(4).to_string(index=False))
    return df, mv, mt, summary


if __name__ == "__main__":
    run()
