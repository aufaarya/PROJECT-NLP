"""
analysis.py  -  qualitative / quantitative feature analysis (Task 3 req. 3c)
--------------------------------------------------------------------------
    python -m src.analysis            (after src.run_task3)

  1. ablation chart                 reports/figures/ablation.png
  2. embedding-space visualisation  t-SNE of headlines + UMAP of daily vectors,
                                    high- vs low-volatility days   -> embedding_space.png
  3. feature importance             grouped permutation importance of the best C model,
                                    topic / anchor correlations with next-day moves
  4. error analysis                 days where text fixed the time-series model vs days
                                    where it added noise, with the headlines behind them
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .features_advanced import _unit, apply_decay, daily_pool  # noqa: E402
from .fusion_models import BUILD  # noqa: E402

warnings.filterwarnings("ignore")
FIG = config.FIGURES
GREY, BLUE, GREEN, ORANGE, RED = "#9e9e9e", "#3b75af", "#4c9a6a", "#e08a2c", "#c0392b"


def _load():
    df = pd.read_csv(config.DATA_PROCESSED / "full_features.csv", parse_dates=["date"])
    meta = json.load(open(config.DATA_PROCESSED / "feature_groups.json"))
    res = pd.read_csv(config.RESULTS / "ablation_all.csv")
    summ = json.load(open(config.RESULTS / "run_summary.json"))
    preds = pd.read_csv(config.RESULTS / "predictions_test.csv", parse_dates=["date"])
    return df, meta, res, summ, preds


def _cols_for(bid, G, best_rep):
    B = G["lexicon"] + G["basic_text"]
    if bid == "A":
        return []
    if bid == "B":
        return B
    if bid == "C-full":
        return B + G[best_rep] + G["topics"] + G["entity"]
    return G[bid[2:]]


# ----------------------------------------------------------------------------
# 1. ablation chart
# ----------------------------------------------------------------------------
def fig_ablation(res):
    d = res[(res.track == "direction")]
    order = [m for m in d[d.split == "test"].model_id.unique()]
    te = d[d.split == "test"].set_index("model_id").loc[order]
    va = d[d.split == "val"].set_index("model_id").loc[order]
    fam_col = {"naive": GREY, "A": BLUE, "B": GREEN, "C": ORANGE, "text-only": "#b07cc6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    y = np.arange(len(order))
    for ax, met in zip(axes, ("Accuracy", "MCC")):
        ax.barh(y, te[met], color=[fam_col[f] for f in te.family], alpha=0.9, label="test")
        ax.scatter(va[met], y, marker="|", color="k", s=90, label="validation", zorder=3)
        ax.axvline(0.5 if met == "Accuracy" else 0, color="k", lw=0.7, ls="--")
        ax.set_title(f"{met} (bars = test, ticks = validation)", fontsize=9)
        ax.tick_params(labelsize=7)
        if met == "Accuracy":
            ax.set_xlim(0.35, 0.68)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=7)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_col.values()]
    axes[1].legend(handles, list(fam_col), fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. embedding space
# ----------------------------------------------------------------------------
def fig_embedding_space(df, rep="minilm", n_per_class=2000, seed=0):
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    corpus = pd.read_parquet(config.CORPUS_FILE)
    X = np.load(config.CACHE / f"{rep}.npy")
    if rep == "finbert":
        X = X[:, :-3]
    d = df.dropna(subset=["target_ret_next"])
    absr = d.set_index("date")["target_ret_next"].abs()
    hi_thr, lo_thr = absr.quantile(0.8), absr.quantile(0.2)
    lab = corpus["date"].map(lambda t: 1 if absr.get(t, np.nan) >= hi_thr else
                             (0 if absr.get(t, np.nan) <= lo_thr else -1))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(lab == k)[0], n_per_class, replace=False)
                          for k in (0, 1)])
    Z = TSNE(n_components=2, init="pca", perplexity=40, random_state=seed).fit_transform(X[idx])
    y = lab.to_numpy()[idx]
    knn = cross_val_score(KNeighborsClassifier(15), X[idx], y, cv=5).mean()

    # daily mean vectors -> UMAP (fallback t-SNE), coloured by next-day |r| quintile
    td = d["date"]
    w = np.log1p(corpus["max_sources"].fillna(1).to_numpy().astype(float))
    M, has = daily_pool(corpus["date"], _unit(X), np.where(w > 0, w, 0.5), td)
    M = M[has]
    q = pd.qcut(d["target_ret_next"].abs().to_numpy()[has], 5, labels=False)
    try:
        import umap
        U = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=seed).fit_transform(M)
        uname = "UMAP"
    except Exception:
        U = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(M)
        uname = "t-SNE"
    years = d["date"].dt.year.to_numpy()[has]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, c, nm in ((0, BLUE, "low-vol day (bottom 20% |r|)"), (1, RED, "high-vol day (top 20% |r|)")):
        m = y == k
        axes[0].scatter(Z[m, 0], Z[m, 1], s=3, c=c, alpha=0.45, label=nm)
    axes[0].set_title(f"t-SNE of {rep} headline embeddings\n15-NN high/low-vol accuracy = {knn:.3f}",
                      fontsize=9)
    axes[0].legend(fontsize=7, markerscale=4)
    sc = axes[1].scatter(U[:, 0], U[:, 1], c=q, cmap="coolwarm", s=8)
    axes[1].set_title(f"{uname} of daily mean vectors\ncolour = next-day |r| quintile", fontsize=9)
    plt.colorbar(sc, ax=axes[1], fraction=0.04)
    sc2 = axes[2].scatter(U[:, 0], U[:, 1], c=years, cmap="viridis", s=8)
    axes[2].set_title(f"same {uname}, colour = year\n(news regimes drift over time)", fontsize=9)
    plt.colorbar(sc2, ax=axes[2], fraction=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "embedding_space.png", dpi=170)
    plt.close(fig)
    return {"knn15_highlow_acc": float(knn), "rep": rep, "map": uname}


# ----------------------------------------------------------------------------
# 3. feature importance
# ----------------------------------------------------------------------------
def feature_importance(df, meta, summ, n_repeats=30):
    """Grouped permutation importance (test ROC-AUC drop) of the best early-fusion C model,
    refit on TRAIN+VAL with its tuned hyper-parameters and decay half-life."""
    from sklearn.metrics import roc_auc_score
    G, best_rep = meta["groups"], summ["best_representation_by_val"]
    cands = [m for m in (summ["best_C"], f"C-full|LGBM", f"C-{best_rep}|LGBM")
             if not any(t in m for t in ("late", "attn"))]
    mid = cands[0]
    bid, model = mid.split("|")
    info = summ["chosen"][f"direction|{mid}"]
    text_cols = _cols_for(bid, G, best_rep)
    cols = G["market"] + text_cols
    d = apply_decay(df, text_cols, info["halflife"]).dropna(subset=["target_ret_next"])
    fit, te = d[d.split != "test"], d[d.split == "test"]
    m = BUILD[model](info["params"]).fit(fit[cols].to_numpy(float), fit["target_up_next"])
    Xte, yte = te[cols].to_numpy(float), te["target_up_next"].to_numpy()
    base = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    fam = {}
    for g, gc in G.items():
        for c in gc:
            fam.setdefault(c, g)
    groups = pd.Series({c: fam.get(c, "other") for c in cols})
    rng = np.random.default_rng(0)
    rows = []
    for g in groups.unique():
        j = [cols.index(c) for c in groups[groups == g].index]
        drops = []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp))][:, j]     # permute the whole group jointly
            drops.append(base - roc_auc_score(yte, m.predict_proba(Xp)[:, 1]))
        rows.append({"group": g, "n_features": len(j), "auc_drop_mean": np.mean(drops),
                     "auc_drop_std": np.std(drops)})
    grp = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)
    single = []
    for c in cols:
        j = cols.index(c)
        drops = []
        for _ in range(10):
            Xp = Xte.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - roc_auc_score(yte, m.predict_proba(Xp)[:, 1]))
        single.append({"feature": c, "group": groups[c], "auc_drop": np.mean(drops)})
    single = pd.DataFrame(single).sort_values("auc_drop", ascending=False)
    grp.to_csv(config.RESULTS / "importance_groups.csv", index=False)
    single.to_csv(config.RESULTS / "importance_features.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(grp.group, grp.auc_drop_mean, xerr=grp.auc_drop_std, color=ORANGE)
    axes[0].axvline(0, color="k", lw=0.7)
    axes[0].invert_yaxis()
    axes[0].set_title(f"{mid}: test AUC drop when a feature group is shuffled\n(base AUC {base:.3f})",
                      fontsize=9)
    top = single.head(15)
    axes[1].barh(top.feature, top.auc_drop, color=[BLUE if g == "market" else ORANGE for g in top.group])
    axes[1].invert_yaxis()
    axes[1].set_title("top-15 single features (blue = market, orange = text)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "importance.png", dpi=170)
    plt.close(fig)
    return {"model": mid, "base_auc": base}


def topic_effects(df, meta):
    """Spearman correlation of each topic share / anchor / sentiment feature with next-day
    return and |return| on TRAIN+VAL (test untouched)."""
    from scipy.stats import spearmanr
    G = meta["groups"]
    d = df[(df.split != "test")].dropna(subset=["target_ret_next"])
    rows = []
    feats = G["topics"] + [c for g in ("minilm", "finbert") for c in G[g] if "anchor" in c] + \
        ["fb_net", "fb_neg_share", "vader_mean", "minilm_novelty", "finbert_novelty"] + \
        [c for c in G["entity"] if c.endswith("_shock")]
    for c in dict.fromkeys(feats):
        r1, p1 = spearmanr(d[c], d["target_ret_next"])
        r2, p2 = spearmanr(d[c], d["target_ret_next"].abs())
        rows.append({"feature": c, "rho_ret": r1, "p_ret": p1, "rho_absret": r2, "p_absret": p2})
    t = pd.DataFrame(rows)
    labels = {**meta["lda_terms"], **meta["cluster_terms"]}
    t["terms"] = t.feature.map(lambda f: " ".join(labels.get(f, [])[:6]))
    t = t.sort_values("p_absret")
    t.to_csv(config.RESULTS / "topic_feature_correlations.csv", index=False)
    return t


# ----------------------------------------------------------------------------
# 4. error analysis
# ----------------------------------------------------------------------------
def error_analysis(df, summ, preds):
    ann = pd.read_parquet(config.DATA_INTERIM / "headline_annotations.parquet")
    a_id, c_id = summ["best_A"], summ["best_C"]
    p = preds.copy()
    p["a_hat"], p["c_hat"] = (p[a_id] >= 0.5).astype(int), (p[c_id] >= 0.5).astype(int)
    p["a_ok"], p["c_ok"] = p.a_hat == p.y_up, p.c_hat == p.y_up
    p["shift"] = p[c_id] - p[a_id]
    p["case"] = np.select([p.c_ok & ~p.a_ok, ~p.c_ok & p.a_ok, p.c_ok & p.a_ok],
                          ["text helped", "text hurt", "both right"], "both wrong")
    # attach news context
    feats = df.set_index("date")
    nov_col = summ["best_representation_by_val"] + "_novelty"
    p["news_novelty"] = p.date.map(feats[nov_col]) if nov_col in feats else np.nan
    p["shock_total"] = p.date.map(feats["shock_total_articles"])
    p["abs_ret_bp"] = p.y_ret.abs() * 1e4
    p.to_csv(config.RESULTS / "error_analysis_days.csv", index=False)

    ex = []
    for case in ("text helped", "text hurt"):
        top = p[p.case == case].reindex(p[p.case == case]["shift"].abs()
                                        .sort_values(ascending=False).index).head(5)
        for _, r in top.iterrows():
            hs = ann[(ann.date == r.date)].sort_values("rank_in_day").headline.head(3).tolist()
            ex.append({"case": case, "date": r.date.date(), "y": "UP" if r.y_up else "DOWN",
                       "p_A": round(r[a_id], 3), "p_C": round(r[c_id], 3),
                       "abs_ret_bp": round(r.abs_ret_bp, 1), "headlines": " | ".join(hs)})
    ex = pd.DataFrame(ex)
    ex.to_csv(config.RESULTS / "error_examples.csv", index=False)

    # error rate by news-intensity tercile and by move-size tercile
    p["novelty_tercile"] = pd.qcut(p.news_novelty.rank(method="first"), 3,
                                   labels=["low", "mid", "high"])
    p["move_tercile"] = pd.qcut(p.abs_ret_bp.rank(method="first"), 3,
                                labels=["small", "mid", "large"])
    by_nov = p.groupby("novelty_tercile")[["a_ok", "c_ok"]].mean()
    by_move = p.groupby("move_tercile")[["a_ok", "c_ok"]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    roll = p.set_index("date")[["a_ok", "c_ok"]].rolling(40, min_periods=20).mean()
    axes[0].plot(roll.index, roll.a_ok, color=BLUE, label=f"A: {a_id}")
    axes[0].plot(roll.index, roll.c_ok, color=ORANGE, label=f"C: {c_id}")
    axes[0].axhline(0.5, color="k", lw=0.7, ls="--")
    axes[0].set_title("40-day rolling test accuracy", fontsize=9)
    axes[0].legend(fontsize=6.5)
    axes[0].tick_params(labelsize=7)
    import matplotlib.dates as mdates
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    axes[0].tick_params(axis="x", rotation=30)
    for ax, tab, ttl in ((axes[1], by_nov, "accuracy by semantic-novelty tercile"),
                         (axes[2], by_move, "accuracy by next-day |move| tercile")):
        x = np.arange(len(tab))
        ax.bar(x - 0.2, tab.a_ok, 0.4, color=BLUE, label="A (time only)")
        ax.bar(x + 0.2, tab.c_ok, 0.4, color=ORANGE, label="C (time + text)")
        ax.set_xticks(x)
        ax.set_xticklabels(tab.index.astype(str), fontsize=8)
        ax.axhline(0.5, color="k", lw=0.7, ls="--")
        ax.set_ylim(0.3, 0.75)
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "error_analysis.png", dpi=170)
    plt.close(fig)
    counts = p.case.value_counts().to_dict()
    return {"cases": counts, "by_novelty": by_nov.round(3).to_dict(),
            "by_move": by_move.round(3).to_dict()}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, meta, res, summ, preds = _load()
    out = {}
    fig_ablation(res)
    out["embedding_space"] = fig_embedding_space(df)
    out["importance"] = feature_importance(df, meta, summ)
    t = topic_effects(df, meta)
    out["top_topic_correlations"] = t.head(10)[["feature", "rho_absret", "p_absret", "terms"]] \
        .round(4).to_dict(orient="records")
    out["error_analysis"] = error_analysis(df, summ, preds)
    with open(config.RESULTS / "analysis_summary.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
