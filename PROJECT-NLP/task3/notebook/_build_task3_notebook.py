"""Generates notebook/task3_ablation_and_analysis.ipynb (kept so the notebook can be regenerated).
    python notebook/_build_task3_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebook/task3_ablation_and_analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Task 3 — Model Refinement & Multimodal Integration
**Hypothesis:** global geopolitical news helps predict next-day moves of the US dollar — here the
Bank Indonesia JISDOR **USD/IDR** rate (the course-mandated source, Task 1 output).

This notebook is the reproducible companion to `src/`:

1. Headline corpus recovered from GDELT (what text we actually feed the encoders)
2. Target, split and feature families
3. **Ablation study** — Model A (time series) vs B (Task 2 basic NLP) vs C (embeddings / topics / entities),
   early vs late vs attention fusion — re-computed live for the LogReg rows, full table loaded from `results/`
4. Rolling-origin robustness and significance tests
5. Embedding-space visualisation (t-SNE / UMAP, high- vs low-volatility days)
6. Feature importance & topic interpretation
7. Error analysis: where text helped and where it added noise

Pipeline order: `python -m src.news_corpus` → `python -m src.build_features` → `python -m src.run_task3` → `python -m src.analysis`.
Embeddings are cached in `data/interim/cache/`, so everything below runs in about a minute.""")

code("""import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from src import config
from src.run_task3 import run_block, splits, mcnemar
from src.evaluate import classification_metrics
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 120)
df = pd.read_csv(config.DATA_PROCESSED / "full_features.csv", parse_dates=["date"])
meta = json.load(open(config.DATA_PROCESSED / "feature_groups.json"))
G = meta["groups"]
print(df.shape, {g: len(c) for g, c in G.items()})""")

md("""## 1. The headline corpus
GDELT event records carry no article text — Task 1's `event_text` is a synthetic code string
(`actor1 X actor2 Y event_code 112`). We recover the **headline from the URL slug** of whitelisted
international outlets, align it to the BI trading day with the Task 1 rule (16:00 WIB cut-off,
weekend/holiday → next trading day), and keep the 60 most widely-covered stories per day (GDELT `NumSources`).""")

code("""print(json.dumps(json.load(open(config.RESULTS / "corpus_summary.json")), indent=1))
corpus = pd.read_parquet(config.CORPUS_FILE)
display(corpus.groupby("source").size().sort_values(ascending=False).head(12).rename("headlines"))
for d in ["2022-02-24", "2023-10-09", "2025-04-07"]:
    print(d); display(corpus[corpus.date == d][["source", "max_sources", "headline"]].head(6))""")

md("## 2. Target, split and feature families")
code("""d = df.dropna(subset=["target_ret_next"])
display(d.groupby("split").agg(start=("date","min"), end=("date","max"), days=("date","size"),
        UP_share=("target_up_next","mean"), highvol_share=("target_highvol_next","mean")).loc[["train","val","test"]])
fig, ax = plt.subplots(figsize=(10, 3))
for s, c in (("train", "#3b75af"), ("val", "#4c9a6a"), ("test", "#e08a2c")):
    x = d[d.split == s]; ax.plot(x.date, x.usd_idr, color=c, label=s, lw=1)
ax.set_title("BI JISDOR USD/IDR and the chronological 70/15/15 split"); ax.legend(); plt.show()
for g, cols in G.items(): print(f"{g:11s} {len(cols):3d}  e.g. {cols[:4]}")""")

md("""**Topic labels** (fit on training headlines only): LDA topics and BERTopic-style MiniLM clusters, each named by its top terms.""")
code("""display(pd.DataFrame({k: " ".join(v[:7]) for k, v in meta["lda_terms"].items()}, index=["LDA top terms"]).T)
display(pd.DataFrame({k: " ".join(v[:7]) for k, v in meta["cluster_terms"].items()}, index=["cluster c-TF-IDF terms"]).T)""")

md("""## 3. Ablation study
### 3a. Live re-computation of the main ablation rows (LogReg, early fusion)
Each block is tuned on validation (C grid × decay half-life h ∈ {0,1,3,5}), refit on train+val and scored once on test.""")
code("""tr, va, te = splits(df)
mk, B = G["market"], G["lexicon"] + G["basic_text"]
live = []
for bid, cols in [("A", []), ("B", B)] + [(f"C-{r}", G[r]) for r in ("w2v","glove","minilm","minilm_ctx","finbert","topics","entity")]:
    out = run_block(df, mk, cols, "LogReg")
    m = classification_metrics(te.target_up_next, (out["test_proba"] >= .5).astype(int), out["test_proba"])
    live.append({"block": bid, "h": out["halflife"], "C": out["params"]["C"], "val_F1": out["val_score"], **m})
display(pd.DataFrame(live).round(3))""")

md("### 3b. Full ablation matrix (all models, fusion strategies) — from `results/ablation_all.csv`")
code("""res = pd.read_csv(config.RESULTS / "ablation_all.csv")
cols = ["model_id", "family", "fusion", "halflife", "Accuracy", "MacroF1", "MCC", "AUC", "p_binom(acc>0.5)"]
for tr_, sp in (("direction", "val"), ("direction", "test"), ("volatility", "test")):
    print(f"=== {tr_} / {sp} ==="); display(res[(res.track == tr_) & (res.split == sp)][cols].round(3))
print("=== regression (next-day log return, bp) ===")
display(pd.read_csv(config.RESULTS / "regression_metrics.csv").round(3))
display(Image(str(config.FIGURES / "ablation.png")))""")

md("## 4. Robustness: significance tests and rolling-origin evaluation")
code("""summ = json.load(open(config.RESULTS / "run_summary.json"))
print("best representation (validation):", summ["best_representation_by_val"])
print("best A / B / C (validation):", summ["best_A"], "/", summ["best_B"], "/", summ["best_C"])
display(pd.DataFrame(summ["mcnemar_test"]).T)
roll = pd.read_csv(config.RESULTS / "rolling_origin.csv")
display(roll.pivot(index="model_id", columns="fold", values="Accuracy").round(3)
        .assign(mean=lambda x: x.mean(1), std=lambda x: x.std(1)).round(3))""")

md("## 5. Embedding space: high- vs low-volatility days")
code("""display(Image(str(config.FIGURES / "embedding_space.png")))
print(json.load(open(config.RESULTS / "analysis_summary.json"))["embedding_space"])""")

md("## 6. Feature importance & interpretability")
code("""display(Image(str(config.FIGURES / "importance.png")))
display(pd.read_csv(config.RESULTS / "importance_groups.csv").round(4))
display(pd.read_csv(config.RESULTS / "importance_features.csv").head(15).round(4))
print("Topic / anchor / sentiment features most associated with next-day |move| (train+val, Spearman):")
display(pd.read_csv(config.RESULTS / "topic_feature_correlations.csv").head(12).round(4))""")

md("""### Attention over the headline stream
The attention-fusion network learns a weight per headline; these are the headlines it attended to most on test days.""")
code("""att = pd.read_csv(config.RESULTS / "attention_top_headlines.csv")
display(att[att["rank"] == 1].sort_values("attention", ascending=False).head(15))""")

md("## 7. Error analysis: where did text help, where did it add noise?")
code("""display(Image(str(config.FIGURES / "error_analysis.png")))
ea = pd.read_csv(config.RESULTS / "error_analysis_days.csv")
display(ea.case.value_counts())
display(pd.read_csv(config.RESULTS / "error_examples.csv"))""")

md("""## 8. Take-aways
See `reports/Task3_Report.pdf` §4–5 for the discussion. In short: the conclusions are drawn from
validation-selected models scored once on test, cross-checked by rolling-origin folds and McNemar tests,
because ~178 test days give a ±7-point noise band on accuracy.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "task3_ablation_and_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
