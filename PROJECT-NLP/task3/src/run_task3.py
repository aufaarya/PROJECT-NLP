"""
run_task3.py  -  ablation study, fusion experiments and tuning
------------------------------------------------------------
    python -m src.run_task3            (after src.news_corpus and src.build_features)

Ablation design (Task 3 req. 3b) - every model sees the SAME days, split and target:
  Model A   time series only            market features (Task 2 baseline), ARIMA
  Model B   A + basic NLP               Task 2 recipe: lexicon volume/mix/shock + VADER + LSA
  Model C   A + advanced text           one block per representation, so representations are
                                         compared on equal footing:
              C-w2v        static, in-domain (Word2Vec, ours)
              C-glove      static, general  (GloVe 6B)
              C-minilm     contextual, general  (all-MiniLM-L6-v2)        headline only
              C-minilm_ctx contextual, general                            headline + event
              C-finbert    contextual, domain   (FinBERT) + FinBERT sentiment
              C-topics     LDA + embedding-cluster topic shares
              C-entity     entity-group volume/shock + entity-only embedding
              C-full       B + best representation + topics + entity
            fusion variants for the best representation:
              early fusion (above), late fusion, attention fusion (neural)
For every C block the exponential-decay half-life h in {0,1,3,5} is tuned on validation.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config
from .evaluate import classification_metrics, regression_metrics
from .fusion_models import (GRIDS, BUILD, attention_fusion, headline_tensor, late_fusion,
                            tune_fit)
from .features_advanced import apply_decay

warnings.filterwarnings("ignore")
REPS = ["w2v", "glove", "minilm", "minilm_ctx", "finbert"]
Y = "target_up_next"


def load():
    df = pd.read_csv(config.DATA_PROCESSED / "full_features.csv", parse_dates=["date"])
    meta = json.load(open(config.DATA_PROCESSED / "feature_groups.json"))
    return df, meta


def splits(df):
    d = df.dropna(subset=["target_ret_next"])
    return [d[d["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")]


def run_block(df, cols_time, cols_text, model, halflifes=(0,) + config.DECAY_HALFLIFES,
              target=Y):
    """Early fusion of time + text block; tune decay half-life and model grid on VAL."""
    best = None
    for h in halflifes if cols_text else (0,):
        d = apply_decay(df, cols_text, h) if cols_text else df
        tr, va, te = splits(d)
        out = tune_fit(model, cols_time + cols_text, tr, va, te, target)
        if best is None or out["val_score"] > best[1]["val_score"] + 1e-12:
            best = (h, out)
    h, out = best
    out["halflife"] = h
    return out


def mcnemar(y, pa, pb):
    """Exact McNemar test: do models A and B make different numbers of errors?"""
    a, b = (pa == y), (pb == y)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return binomtest(n10, n01 + n10, 0.5).pvalue, n01, n10


def rolling_origin(df, cols_time, cols_text, model, params, halflife, n_folds=6, target=Y):
    """Expanding-window evaluation over the val+test period (~355 days, 6 folds of ~59 days).
    Unsupervised text models were fit on TRAIN text only, i.e. strictly before every fold."""
    d = apply_decay(df, cols_text, halflife) if cols_text else df
    d = d.dropna(subset=["target_ret_next"]).reset_index(drop=True)
    start = int((d["split"] == "train").sum())
    edges = np.linspace(start, len(d), n_folds + 1).astype(int)
    cols = cols_time + cols_text
    rows = []
    for k in range(n_folds):
        fit, ev = d.iloc[:edges[k]], d.iloc[edges[k]:edges[k + 1]]
        m = BUILD[model](params).fit(fit[cols].to_numpy(float), fit[target].to_numpy())
        p = m.predict_proba(ev[cols].to_numpy(float))[:, 1]
        met = classification_metrics(ev[target], (p >= 0.5).astype(int), p)
        rows.append({"fold": k + 1, "start": str(ev["date"].min().date()),
                     "end": str(ev["date"].max().date()), **met})
    return pd.DataFrame(rows)


def volatility_rolling(df=None, meta=None, summary=None):
    """Rolling-origin check for the volatility track (A, B and every C block with LGBM and
    LogReg), using the hyper-parameters / half-life chosen on validation in main()."""
    if df is None:
        df, meta = load()
    if summary is None:
        summary = json.load(open(config.RESULTS / "run_summary.json"))
    G, best_rep = meta["groups"], summary["best_representation_by_val"]
    B = G["lexicon"] + G["basic_text"]
    cols_of = {"A": [], "B": B, f"C-{best_rep}": G[best_rep], "C-entity": G["entity"],
               "C-full": B + G[best_rep] + G["topics"] + G["entity"]}
    out = []
    for key, info in summary["chosen"].items():
        track, mid = key.split("|", 1)
        if track != "volatility":
            continue
        bid, model = mid.split("|")
        r = rolling_origin(df, G["market"], cols_of[bid], model, info["params"],
                           info["halflife"], target="target_highvol_next")
        r.insert(0, "model_id", mid)
        out.append(r)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(config.RESULTS / "rolling_origin_volatility.csv", index=False)
    return out


def main(verbose=True, run_attention=True):
    df, meta = load()
    G = meta["groups"]
    tr, va, te = splits(df)
    mk = G["market"]
    B = G["lexicon"] + G["basic_text"]
    rows, preds, chosen = [], pd.DataFrame({"date": te["date"], "y_up": te[Y],
                                            "y_ret": te["target_ret_next"]}), {}
    val_preds = pd.DataFrame({"date": va["date"], "y_up": va[Y]})

    def record(model_id, family, features, fusion, out, track="direction", target=Y):
        for split, d, p in (("val", va, out["val_proba"]), ("test", te, out["test_proba"])):
            m = classification_metrics(d[target], (p >= 0.5).astype(int), p)
            rows.append({"track": track, "split": split, "model_id": model_id,
                         "family": family, "features": features, "fusion": fusion,
                         "halflife": out.get("halflife", 0), **m})
        if track == "direction":
            preds[model_id] = out["test_proba"]
            val_preds[model_id] = out["val_proba"]
        chosen[f"{track}|{model_id}"] = {k: out[k] for k in ("params", "halflife", "w_time",
                                                              "epochs") if k in out}

    # ------------------------------------------------------------ naive references
    maj = int(pd.concat([tr, va])[Y].mean() >= 0.5)
    for split, d in (("val", va), ("test", te)):
        for name, p in (("Majority", np.full(len(d), maj)), ("Persistence", (d["ret"] > 0).astype(int))):
            rows.append({"track": "direction", "split": split, "model_id": name, "family": "naive",
                         "features": "none", "fusion": "-", "halflife": 0,
                         **classification_metrics(d[Y], p)})

    # ------------------------------------------------------------ A / B / C (early fusion)
    blocks = {"A": ("A", "market", [])}
    blocks["B"] = ("B", "market + lexicon + VADER + LSA", B)
    for r in REPS:
        blocks[f"C-{r}"] = ("C", f"market + {r}", G[r])
    blocks["C-topics"] = ("C", "market + LDA/cluster topics", G["topics"])
    blocks["C-entity"] = ("C", "market + entity filter", G["entity"])

    results = {}
    for model in ("LogReg", "LGBM"):
        for bid, (fam, desc, cols) in blocks.items():
            out = run_block(df, mk, cols, model)
            results[(bid, model)] = out
            record(f"{bid}|{model}", fam, desc, "early" if cols else "-", out)
            if verbose:
                print(f"  {bid:13s} {model:6s} h={out['halflife']}  val F1 {out['val_score']:.3f}")

    # best representation (by validation macro-F1, either model)
    rep_scores = {r: max(results[(f"C-{r}", m)]["val_score"] for m in ("LogReg", "LGBM"))
                  for r in REPS}
    best_rep = max(rep_scores, key=rep_scores.get)
    full_cols = B + G[best_rep] + G["topics"] + G["entity"]
    for model in ("LogReg", "LGBM"):
        out = run_block(df, mk, full_cols, model)
        results[("C-full", model)] = out
        record(f"C-full|{model}", "C", f"market + B + {best_rep} + topics + entity", "early", out)

    # ------------------------------------------------------------ late fusion
    # time model = tuned Model A LogReg; text model = LogReg on the best representation only
    a_out = results[("A", "LogReg")]
    txt = run_block(df, [], G[best_rep], "LogReg")
    lf = late_fusion(a_out["val_proba"], txt["val_proba"], va[Y].to_numpy(),
                     a_out["test_proba"], txt["test_proba"])
    lf["halflife"] = txt["halflife"]
    record(f"C-late-{best_rep}|LogReg", "C", f"p(market) + p({best_rep})", "late", lf)
    record(f"T-only-{best_rep}|LogReg", "text-only", f"{best_rep} only", "-", txt)

    # ------------------------------------------------------------ attention fusion
    att_info = None
    if run_attention:
        corpus = pd.read_parquet(config.CORPUS_FILE)
        for rep in ("minilm", "finbert"):
            X = np.load(config.CACHE / f"{rep}.npy")
            if rep == "finbert":
                X = X[:, :-3]
            d = df.dropna(subset=["target_ret_next"]).reset_index(drop=True)
            H, M, rid = headline_tensor(corpus, X, d["date"])
            idx = {s: np.where(d["split"] == s)[0] for s in ("train", "val", "test")}
            out = attention_fusion(H, M, d[mk].to_numpy(float), d[Y].to_numpy(float),
                                   idx["train"], idx["val"], idx["test"])
            record(f"C-attn-{rep}|NN", "C", f"market + {rep} headline stream", "attention", out)
            if verbose:
                print(f"  attention fusion {rep}: epochs {out['epochs']}")
            if rep == "minilm":
                att = out["test_attention"]
                rid_te = rid[idx["test"]]
                top = []
                for i, day in enumerate(d.loc[idx["test"], "date"]):
                    j = np.argsort(att[i])[::-1][:3]
                    for rnk, jj in enumerate(j):
                        if rid_te[i, jj] >= 0:
                            top.append({"date": day.date(), "rank": rnk + 1,
                                        "attention": float(att[i, jj]),
                                        "headline": corpus.loc[rid_te[i, jj], "headline"]})
                pd.DataFrame(top).to_csv(config.RESULTS / "attention_top_headlines.csv",
                                         index=False)
                att_info = {"val_curve": out["val_curve"]}

    # ------------------------------------------------------------ volatility track
    for model in ("LogReg", "LGBM"):
        for bid, cols in (("A", []), ("B", B), (f"C-{best_rep}", G[best_rep]),
                          ("C-entity", G["entity"]), ("C-full", full_cols)):
            out = run_block(df, mk, cols, model, target="target_highvol_next")
            fam = bid.split("-")[0]
            record(f"{bid}|{model}", fam, bid, "early" if cols else "-", out,
                   track="volatility", target="target_highvol_next")

    # ------------------------------------------------------------ regression track (ARIMA)
    from .task2.models import arima_walk_forward, select_arima_order
    d = df.dropna(subset=["target_ret_next"]).reset_index(drop=True)
    m_tr = (d["split"] == "train").to_numpy()
    m_trva = d["split"].isin(["train", "val"]).to_numpy()
    iv, it = np.where(d["split"] == "val")[0], np.where(d["split"] == "test")[0]
    order, aic = select_arima_order(d[m_tr])
    aic.to_csv(config.RESULTS / "arima_order_search.csv", index=False)
    reg_rows = []
    for s, dd in (("val", d.iloc[iv]), ("test", d.iloc[it])):
        reg_rows.append({"split": s, "model": "RandomWalk (0)", "family": "naive",
                         **regression_metrics(dd["target_ret_next"], np.zeros(len(dd)))})
    shock_cols = [c for c in G["lexicon"] if c.startswith("shock_")]
    sem_cols = [c for c in G[best_rep] if "anchor" in c or "novelty" in c]
    for name, fam, ex in ((f"ARIMA{order}", "A", None),
                          (f"ARIMAX{order} lexicon shocks", "B", shock_cols),
                          (f"ARIMAX{order} {best_rep} anchors+novelty", "C", sem_cols)):
        pv, _ = arima_walk_forward(d, m_tr, order, exog_cols=ex)
        pt, _ = arima_walk_forward(d, m_trva, order, exog_cols=ex)
        for s, p, idx in (("val", pv, iv), ("test", pt, it)):
            reg_rows.append({"split": s, "model": name, "family": fam,
                             **regression_metrics(d.loc[idx, "target_ret_next"], p[idx])})
        preds[f"reg|{fam}"] = pt[it]
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(config.RESULTS / "regression_metrics.csv", index=False)

    # ------------------------------------------------------------ tables, tests, rolling
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "ablation_all.csv", index=False)
    dir_te = res[(res.track == "direction") & (res.split == "test")]
    dir_va = res[(res.track == "direction") & (res.split == "val")]
    # model selection for the headline comparison uses VALIDATION only
    cand = dir_va[dir_va.family == "C"].sort_values("MacroF1", ascending=False)
    best_c = cand.iloc[0]["model_id"]
    best_a = dir_va[dir_va.family == "A"].sort_values("MacroF1", ascending=False).iloc[0]["model_id"]
    best_b = dir_va[dir_va.family == "B"].sort_values("MacroF1", ascending=False).iloc[0]["model_id"]
    yte = te[Y].to_numpy().astype(int)
    tests = {}
    for a, b in ((best_a, best_c), (best_b, best_c), (best_a, best_b)):
        p, n01, n10 = mcnemar(yte, (preds[a] >= 0.5).astype(int).to_numpy(),
                              (preds[b] >= 0.5).astype(int).to_numpy())
        tests[f"{a} vs {b}"] = {"p_mcnemar": p, "only_first_right": n01, "only_second_right": n10}

    roll = []
    for mid in dict.fromkeys([best_a, best_b, best_c, "A|LogReg", "B|LogReg",
                              f"C-{best_rep}|LogReg", f"C-{best_rep}|LGBM", "A|LGBM", "B|LGBM"]):
        bid, model = mid.split("|")
        if model == "NN" or bid.startswith("C-late"):
            continue
        cols = {"A": [], "B": B, "C-full": full_cols}.get(bid)
        if cols is None:
            key = bid[2:]
            cols = G[key] if key in G else G.get(key, [])
        info = chosen[f"direction|{mid}"]
        r = rolling_origin(df, mk, cols, model, info["params"], info["halflife"])
        r.insert(0, "model_id", mid)
        roll.append(r)
    roll = pd.concat(roll, ignore_index=True)
    roll.to_csv(config.RESULTS / "rolling_origin.csv", index=False)

    preds.to_csv(config.RESULTS / "predictions_test.csv", index=False)
    val_preds.to_csv(config.RESULTS / "predictions_val.csv", index=False)
    summary = {"best_representation_by_val": best_rep, "rep_val_macroF1": rep_scores,
               "best_A": best_a, "best_B": best_b, "best_C": best_c,
               "mcnemar_test": tests, "arima_order": list(order), "chosen": chosen,
               "attention": att_info,
               "rolling_mean": roll.groupby("model_id")[["Accuracy", "MacroF1", "MCC"]]
               .mean().round(4).to_dict(orient="index")}
    with open(config.RESULTS / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    vroll = volatility_rolling(df, meta, json.loads(json.dumps(summary, default=str)))
    if verbose:
        print(vroll.groupby("model_id")[["Accuracy", "MacroF1", "MCC", "AUC"]].mean().round(3))
        pd.set_option("display.width", 220)
        show = ["model_id", "fusion", "halflife", "Accuracy", "MacroF1", "MCC", "AUC",
                "p_binom(acc>0.5)"]
        print("\n=== DIRECTION / VAL ===\n", dir_va[show].round(3).to_string(index=False))
        print("\n=== DIRECTION / TEST ===\n", dir_te[show].round(3).to_string(index=False))
        print("\n=== VOLATILITY / TEST ===\n",
              res[(res.track == "volatility") & (res.split == "test")][show].round(3)
              .to_string(index=False))
        print("\n=== REGRESSION ===\n", reg.round(3).to_string(index=False))
        print("\nbest:", best_a, best_b, best_c, "\n", json.dumps(tests, indent=1))
        print(roll.groupby("model_id")[["Accuracy", "MacroF1", "MCC"]].mean().round(3))
    return res, summary


if __name__ == "__main__":
    main()
