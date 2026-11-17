# Global Geopolitical Event Prediction — Impact on USD Exchange Rates
## Task 3: Model Refinement & Multimodal Integration

**Team:** Rafaela Anabel Purba (24/547080/PA/23218) · Aufaa Azzahra Aryawan (24/546854/PA/23215) ·
Aurora Cindita Syachbudtari (24/532921/PA/22554)

Hypothesis: global geopolitical news helps predict next-day moves of the US dollar —
here the **Bank Indonesia JISDOR USD/IDR** rate (Task 1 output, 2021-09-01 … 2026-09-01).

Text: **71,520 headlines** recovered from the URLs of Task 1's GDELT events (60 most widely
covered stories per BI trading day, 84 whitelisted international outlets).

```
data/
  raw/              Task 1 outputs: bi_usd_cleaned.csv, daily_aligned_dataset.csv, GDELT sample
  processed/        train.csv, val.csv, test.csv (chronological 70/15/15), full_features.csv,
                    feature_groups.json, headline_corpus.parquet (+ csv sample)
  interim/          (git-ignored) headlines_all/aligned.parquet, embedding caches *.npy
  task2_fred_nyt/   Task 2 data (FRED DTWEXBGS + NYT counts), unchanged
src/
  config.py            paths, whitelist, lexicons, entity gazetteer, model ids, split
  news_corpus.py       GDELT events -> URL-slug headlines -> filter -> align (16:00 WIB) -> top-60/day
  data_prep.py         BI trading-day table, targets (direction, volatility), split
  features_basic.py    Model B = Task 2 recipe on real text: lexicon volume/mix/shock, VADER, TF-IDF/LSA
  embeddings.py        Word2Vec (SGNS in PyTorch, train-only), GloVe, MiniLM, FinBERT (+ sentiment)
  features_advanced.py daily pooling: salience-weighted mean -> PCA, concept anchors, semantic novelty,
                       dispersion, exp. decay; LDA + BERTopic-style clusters; entity filtering
  build_features.py    builds every feature family -> data/processed/
  fusion_models.py     early fusion (LogReg/LightGBM), late fusion, attention-fusion network
  run_task3.py         ablation A/B/C, decay + hyper-parameter tuning, volatility & ARIMA tracks,
                       McNemar tests, rolling-origin evaluation
  analysis.py          ablation chart, t-SNE/UMAP, permutation importance, topics, error analysis
  evaluate.py          metrics (from Task 2)
  task2/               Task 2 pipeline, unchanged (python -m src.task2.run_pipeline)
notebook/
  task2_eda_and_baselines.ipynb       Task 2 EDA + baselines
  task3_ablation_and_analysis.ipynb   Task 3 ablation (live re-run), robustness, analysis (executed)
results/           ablation_all.csv, rolling_origin*.csv, regression_metrics.csv, predictions_*.csv,
                   importance_*.csv, error_*.csv, attention_top_headlines.csv, run_summary.json, ...
  task2/           Task 2 results
reports/           Task3_Report.pdf, figures/, make_report.py, make_diagram.py; task2/ (Task 2 report)
```

### Run
```bash
pip install -r requirements.txt
python -m src.news_corpus        # needs Task 1's news_cleaned.parquet (path in src/config.py), ~15 min
python -m src.build_features     # encoders on CPU, ~30 min first time (cached afterwards)
python -m src.run_task3          # ablation + fusion + robustness, ~25 min
python -m src.analysis           # figures / interpretability
python reports/make_diagram.py && python reports/make_report.py
python notebook/_build_task3_notebook.py
python -m nbconvert --to notebook --execute --inplace notebook/task3_ablation_and_analysis.ipynb
```

### Ablation design
| Model | Features | Notes |
|---|---|---|
| A | 9 market features (Task 2) | ARIMA for regression |
| B | A + Task 2 basic NLP (lexicon volume/mix/shock, VADER, TF-IDF/LSA) | now run on real headlines |
| C | A + one advanced block: Word2Vec, GloVe, MiniLM, MiniLM+event context, FinBERT, topics, entities, or all | early, late and attention fusion |

Every block is tuned on validation (classifier grid × decay half-life), refit on train+val and
scored once on test; robustness is checked with 6 rolling-origin folds and McNemar tests.

### Main result (details in the report)
Text features improve every **validation** score but not **test** direction accuracy;
rolling-origin AUC is slightly higher with text, and entity features help more for a
**volatility** target than for direction.

### Note on the Task 2 → Task 3 change
Task 2 used FRED DTWEXBGS + NYT daily counts, and its text layer never ran because the NYT article
file was missing. Task 3 uses the course-mandated BI series and the GDELT data from Task 1, and
re-runs the Task 2 baseline (A) and basic NLP (B) on it so the ablation is like-for-like. The
original Task 2 experiment is kept in `src/task2`, `data/task2_fred_nyt`, `results/task2`.
