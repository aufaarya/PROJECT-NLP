"""Builds reports/Task2_Report.pdf from results/ + reports/figures/ (run after run_pipeline)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Task2_Report.pdf"

mv = pd.read_csv(RES / "metrics_val.csv")
mt = pd.read_csv(RES / "metrics_test.csv")
summ = json.load(open(RES / "run_summary.json"))
sp = summ["splits"]
order = tuple(summ["arima_order"])
t1 = pd.read_csv(ROOT / "data" / "raw" / "daily_features_usd_target.csv")
zero_share = (t1["usd_pct_change_next"] == 0).mean()
t1_up = t1["usd_direction_next"].mean()
n_articles = pd.read_csv(ROOT / "data" / "raw" / "nyt_collection_log.csv")["n_articles"].sum()

ss = getSampleStyleSheet()
INK = colors.HexColor("#0b0b0b"); INK2 = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6"); GRID = colors.HexColor("#d9d8d2")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6, textColor=INK, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=INK, keepWithNext=1)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.4, spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.2, textColor=INK2, alignment=TA_CENTER, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=8, leading=10, spaceAfter=0)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=18, leading=22, spaceAfter=4)
SUB = ParagraphStyle("S", parent=B, alignment=TA_CENTER, textColor=INK2, fontSize=10)


def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BL, bulletText="•") for i in items]


def table(data, widths, header_bg="#eef4fc", highlight_rows=()):
    rows = [[Paragraph(str(c), TC) for c in r] for r in data]
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
          ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in highlight_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdeee6")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def f(x, d=3):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def role(model, feats):
    if feats == "none" or model.startswith("Persistence") or model.startswith("Majority"):
        return "naive"
    return "combined" if "news" in feats else "baseline"


def clf_rows(m):
    c = m[m.track == "classification"]
    out = [["Model", "Features", "Role", "Acc.", "Macro-F1", "MCC", "AUC", "p (acc>0.5)"]]
    hl = []
    for i, (_, r) in enumerate(c.iterrows(), start=1):
        rl = role(r.model, r.features)
        if rl == "combined": hl.append(i)
        out.append([r.model, r.features, rl, f(r.Accuracy), f(r.MacroF1), f(r.MCC), f(r.AUC),
                    f(r["p_binom(acc>0.5)"], 3)])
    return out, hl


def reg_rows(m):
    r_ = m[m.track == "regression"]
    out = [["Model", "Features", "Role", "RMSE (bp)", "MAE (bp)", "Dir. Acc."]]
    hl = []
    for i, r in enumerate(r_.itertuples(), start=1):
        rl = "naive" if r.features == "none" else ("combined" if "news" in r.features else "baseline")
        if rl == "combined": hl.append(i)
        out.append([r.model, r.features, rl, f(r.RMSE_bp, 2), f(r.MAE_bp, 2), f(r.DirAcc)])
    return out, hl


def g(m, model, feats, col):
    return m[(m.model == model) & (m.features == feats)][col].iloc[0]


A = f"ARIMA{order}"
rw_te = g(mt, "RandomWalk (0)", "none", "RMSE_bp")
ar_te = g(mt, A, "market", "RMSE_bp")
arx_te = g(mt, f"ARIMAX{order}", summ["arimax_exog_set"], "RMSE_bp")
best_te = mt[mt.track == "classification"].sort_values("Accuracy", ascending=False).iloc[0]
min_p = mt["p_binom(acc>0.5)"].min()

story = []
story += [P("Global Geopolitical Event Prediction:<br/>Impact on USD Exchange Rates", TITLE),
          P("Task 2 — Pipeline Proposal &amp; Baseline Experimentation", SUB),
          P("Team: ____________________ &nbsp;&nbsp; Members: ____________________", SUB),
          Spacer(1, 10)]

# ------------------------------------------------------------------ summary
story += [P("Summary", H1), P(
    f"We predict the <b>next trading day's</b> move of the FRED Nominal Broad U.S. Dollar Index "
    f"(DTWEXBGS) from New York Times geopolitical news (2021–2025, {n_articles:,} archive "
    f"articles screened). The task is formulated both as <b>binary direction</b> (UP/DOWN, "
    f"primary) and as <b>regression</b> of the next-day log return (secondary). NLP features "
    f"are built with a <b>keyword-lexicon</b> classifier over headline + abstract + NYT keyword "
    f"tags, turned into per-category volume, mix and <i>news-shock</i> features; a VADER + "
    f"TF-IDF/LSA text layer is implemented and switches on when the article-level file is "
    f"supplied. Baselines use exchange-rate history only (random walk, persistence, "
    f"ARIMA{order}, LogReg/GBM on lagged returns); combined models add the NLP features "
    f"(ARIMAX, LogReg/GBM). On a strictly chronological held-out test set "
    f"({sp['test']['start']} to {sp['test']['end']}, {sp['test']['rows']} days) no model is "
    f"statistically better than chance (smallest binomial p = {min_p:.3f}) and none beats the "
    f"random-walk RMSE — the expected, honest starting point for daily FX that Task 3 must improve on.")]

# ------------------------------------------------------------------ 1 formulation
story += [P("1. Task formulation", H1), P("1.1 Correction carried over from Task 1", H2), P(
    f"The Task 1 table lives on a calendar-day grid with the USD index forward-filled across "
    f"weekends and holidays. As a result <b>{zero_share:.1%}</b> of rows have a next-day change "
    f"of exactly 0 and are labelled DOWN, so only {t1_up:.1%} of labels are UP and the "
    f"'always DOWN' rule scores {1 - t1_up:.1%} by construction. That is why the Task 1 "
    f"logistic regression (63.0%) appeared to lose to the naive baseline (66.0%). FX does not "
    f"trade on those days, so they are not valid targets. We rebuild the dataset on "
    f"<b>trading days only</b> and <b>sum weekend/holiday news onto the next trading day</b> "
    f"(Sat + Sun + Mon → Monday) so no news is discarded. On the rebuilt data the UP share is "
    f"{sp['train']['up_share']:.1%} (train), {sp['val']['up_share']:.1%} (val) and "
    f"{sp['test']['up_share']:.1%} (test) — close to balanced, as expected for FX.")]

story += [P("1.2 Target variable", H2), *bullets([
    "<b>Primary — binary classification:</b> UP(t+1) = 1 if r(t+1) &gt; 0, else 0, where "
    "r(t+1) = ln P(t+1) − ln P(t) is the next trading day's log return. Direction is what a "
    "trading or hedging decision needs, and it is robust to the fat-tailed return distribution.",
    "<b>Secondary — regression:</b> r(t+1) itself, so ARIMA/ARIMAX can be compared on magnitude "
    "as well (RMSE/MAE in basis points).",
    "<b>Timing:</b> row t holds news published in (t_prev, t] and prices up to P(t); every feature "
    "is available at the end of day t, strictly before the target period. A perturbation test "
    "(changing r(t+1) leaves the forecast for day t unchanged) confirms there is no look-ahead."])]

story += [P("1.3 Chronological split", H2), table([
    ["Split", "Period", "Trading days", "UP share", "Used for"],
    ["Train (70%)", f"{sp['train']['start']} → {sp['train']['end']}", sp['train']['rows'], f"{sp['train']['up_share']:.3f}", "fitting parameters, ARIMA order (AIC), scalers, TF-IDF vocabulary"],
    ["Validation (15%)", f"{sp['val']['start']} → {sp['val']['end']}", sp['val']['rows'], f"{sp['val']['up_share']:.3f}", "hyper-parameters, ARIMAX exogenous-set choice"],
    ["Test (15%)", f"{sp['test']['start']} → {sp['test']['end']}", sp['test']['rows'], f"{sp['test']['up_share']:.3f}", "evaluated once, after refitting on train + val"],
], [2.6 * cm, 4.6 * cm, 2.2 * cm, 1.7 * cm, 6.2 * cm]),
    P("No shuffling. The first 20 trading days of 2021 are consumed as warm-up for the rolling "
      "features, so training starts on " + sp['train']['start'] + ". Rolling statistics use only "
      "past windows (shift 1). Files: <font face='Courier'>data/processed/{train,val,test}.csv</font>.", CAP)]

story += [P("1.4 Evaluation metrics", H2), *bullets([
    "<b>Accuracy</b> (directional accuracy) — the headline number, directly interpretable against 50%.",
    "<b>Macro-F1</b> — penalises a model that only predicts one class; also the model-selection criterion on validation.",
    "<b>MCC</b> (Matthews correlation) — 0 for any constant or random predictor, so it cannot be gamed by class balance shifts between periods (validation is 57% UP, test 48%).",
    "<b>ROC-AUC</b> — quality of the predicted probabilities, independent of the 0.5 threshold.",
    "<b>One-sided binomial test</b> of accuracy &gt; 50% — with ~185 test days the 95% noise band is roughly ±7 points, so we report whether any edge is distinguishable from luck.",
    "<b>RMSE / MAE</b> (bp) and directional accuracy for the regression track, always compared with the random walk (forecast = 0)."])]

# ------------------------------------------------------------------ 2 NLP
story += [P("2. NLP feature extraction and justification", H1), P("2.1 Which text field?", H2), P(
    "The NYT Archive API returns metadata, not full body text: headline, abstract, lead paragraph "
    "and editor-assigned <b>keywords</b> (subjects, geographic locations, organizations, persons). "
    "We use <b>headline + abstract + keywords</b>. Headlines are short, dense and written to state "
    "the event; the abstract adds the actor and outcome; the keyword tags are human-curated "
    "<b>entities</b> (e.g. tags for a specific war, country or trade policy), which gives us an "
    "entity signal without needing an NER model. Full body text would add length and noise "
    "(background paragraphs, quotes) and is not available from this API without scraping, which "
    "would raise copyright and reproducibility issues.")]

story += [P("2.2 Current features: lexicon category counts → volume, mix and shock", H2), P(
    "Each article is matched against a transparent keyword lexicon for five categories — "
    "military_conflict, sanctions, diplomacy, trade_conflict, political_crisis — kept identical "
    "to Task 1 so definitions are consistent across tasks. Per trading day we compute:"),
    *bullets([
        "<b>log1p volume</b> per category and total (averaged per calendar day so a Monday bucket is comparable to a Tuesday); log damps election-day outliers of 800–900 articles.",
        "<b>category share</b> of the day's geopolitical articles — the <i>mix</i> of news, independent of how much the NYT publishes overall.",
        "<b>news-shock z-score</b>: today's log volume against its own mean and std over the previous 20 trading days. Markets price <i>surprises</i>, not the fact that ~28 war articles appear every day; the shock isolates unusual news days (e.g. 24 Feb 2022, Russia's invasion: military-conflict shock z = 3.3; 3 Apr 2025, the day after the 'Liberation Day' tariffs: trade-conflict shock z = 3.4).",
    ]),
    P("Why lexicon-based rather than sentiment first: FX responds to <i>risk</i> and <i>uncertainty</i> "
      "(safe-haven flows into the dollar), which is closer to 'how much alarming news of type X' "
      "than to positive/negative tone. The lexicon is fully auditable — every feature can be traced "
      "back to the keywords that produced it — which matters because the course requires every "
      "design choice to be justified.")]

story += [P("2.3 Text layer (implemented, activates with the article file)", H2), *bullets([
    "<b>VADER</b> compound score on title + abstract, aggregated per day as mean tone, share negative, share positive. VADER is a rule-based lexicon tuned for short, headline-like text.",
    "<b>TF-IDF</b> (uni- and bi-grams, English stop-words, sublinear TF, min_df = 5) on titles → <b>TruncatedSVD / LSA</b> with 10 components → daily mean topic loadings. Vocabulary and SVD are fit on <b>train-period articles only</b>, and the top terms per component are exported so each topic can be named in the report.",
    "Loughran–McDonald was considered; its dictionary is built for financial filings, so it is kept as a Task 3 option (its <i>uncertainty</i> word list is the most relevant part).",
    "Per course rules, <b>no pre-trained embeddings or transformer models</b> are used anywhere.",
]), P("This layer was unit-tested on synthetic articles but not run on real data in this report, "
      "because only the aggregated daily counts were available to us; placing "
      "<font face='Courier'>nyt_geopolitical_articles_clean.csv</font> in "
      "<font face='Courier'>data/raw/</font> adds a third feature set "
      "(market + news + text) automatically.", CAP)]

# ------------------------------------------------------------------ 3 architecture
story += [KeepTogether([P("3. Pipeline architecture", H1),
          img(FIG / "pipeline_diagram.png", 17.5),
          P("Figure 1. End-to-end pipeline. Dashed boxes = implemented path that activates when the "
            "article-level file is present. Blue/green = data &amp; history-only path, orange = NLP path.", CAP)]),
          table([
              ["Module", "Responsibility"],
              ["src/config.py", "paths, categories and lexicon, rolling windows, split ratios, seed"],
              ["src/data_prep.py", "trading-day calendar, weekend-news bucketing, targets, chronological split"],
              ["src/features.py", "market features, lexicon volume/mix/shock features, VADER + TF-IDF/LSA text features"],
              ["src/models.py", "ARIMA order search (AIC), frozen-parameter walk-forward ARIMA/ARIMAX, LogReg/GBM with validation grid search"],
              ["src/evaluate.py", "regression and classification metrics, binomial test"],
              ["src/run_pipeline.py", "runs everything, writes data/processed/ and results/"],
          ], [4.2 * cm, 13.1 * cm])]

# ------------------------------------------------------------------ 4 models & results
story += [P("4. Baseline and combined models", H1), *bullets([
    f"<b>Random walk</b> (predict r = 0), <b>Majority</b> class and <b>Persistence</b> (tomorrow = today's direction): the reference points any model must beat.",
    f"<b>ARIMA{order}</b> — baseline time-series model on log returns (ADF p ≈ 1e-14, so d = 0). "
    f"Order chosen by AIC over p, q ∈ {{0..3}} on train; parameters are then frozen and one-step "
    f"forecasts are produced by running the Kalman filter forward (walk-forward without look-ahead).",
    f"<b>ARIMAX{order}</b> — combined model: same order plus news features of day t as exogenous "
    f"regressors for r(t+1). Three exogenous sets were compared on validation; "
    f"'<b>{summ['arimax_exog_set']}</b>' (the six shock z-scores) won.",
    "<b>Logistic regression</b> (standardised, C tuned) and <b>gradient boosting</b> (depth, learning rate, iterations tuned) — each trained on market features only (baseline) and market + news (combined). Hyper-parameters are picked by validation macro-F1, then the model is refit on train + val and scored once on test.",
])]

rr, hl = reg_rows(mt)
story += [KeepTogether([P("4.1 Test results — regression (next-day log return)", H2),
                        table(rr, [3.6 * cm, 2.8 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2.3 * cm], highlight_rows=hl)]),
          P(f"Random walk RMSE {rw_te:.2f} bp vs ARIMA {ar_te:.2f} bp vs ARIMAX {arx_te:.2f} bp. "
            f"Adding news slightly reduces ARIMA's error but neither beats 'no change'.", CAP)]

cr, hl = clf_rows(mt)
story += [KeepTogether([P("4.2 Test results — direction (UP/DOWN)", H2),
                        table(cr, [3.4 * cm, 2.4 * cm, 1.7 * cm, 1.4 * cm, 1.7 * cm, 1.4 * cm, 1.4 * cm, 1.9 * cm], highlight_rows=hl)]),
          P("Orange rows = combined (history + NLP) models.", CAP),
          img(FIG / "test_classification.png", 17),
          P("Figure 2. Test accuracy and MCC by model; grey = naive, blue = history only, orange = history + NLP.", CAP)]

cr_v, hl_v = clf_rows(mv)
story += [KeepTogether([P("4.3 Validation results — direction", H2),
                        table(cr_v, [3.4 * cm, 2.4 * cm, 1.7 * cm, 1.4 * cm, 1.7 * cm, 1.4 * cm, 1.4 * cm, 1.9 * cm], highlight_rows=hl_v)])]

# ------------------------------------------------------------------ 5 discussion
lr_m = g(mt, "LogReg", "market", "Accuracy"); lr_n = g(mt, "LogReg", "market+news", "Accuracy")
gb_m = g(mt, "GBM", "market", "Accuracy"); gb_n = g(mt, "GBM", "market+news", "Accuracy")
story += [P("5. Interpretation, limitations and next steps", H1), *bullets([
    f"<b>No edge yet.</b> The best test accuracy is {best_te.Accuracy:.1%} ({best_te.model}, "
    f"{best_te.features}), with binomial p = {best_te['p_binom(acc>0.5)']:.3f}. Daily USD returns are "
    f"close to white noise (all train autocorrelations inside the 95% band), so this is the "
    f"realistic baseline, not a failure of the pipeline.",
    f"<b>Mixed effect of news.</b> Adding NLP features raises LogReg from {lr_m:.1%} to {lr_n:.1%} "
    f"and turns ARIMA's negative MCC positive, but lowers GBM from {gb_m:.1%} to {gb_n:.1%}; on "
    f"validation the ranking is different again. With ~185 days per split these gaps are inside "
    f"the noise band, so we do <b>not</b> claim support for the hypothesis at this stage — that "
    f"test is Task 4.",
    "<b>ARIMA order.</b> AIC picked (3,0,2) but its AR(2)/MA(2) terms nearly cancel "
    "(−0.90 vs +0.86) and the AIC gain over AR(2) is only ~2 points: a sign of over-fitting noise. "
    "A simpler order is a candidate for Task 3.",
    "<b>Feature limitations.</b> Count features know <i>how much</i> news there is, not <i>what</i> it "
    "says or <i>who</i> is involved; the first-match category rule also assigns multi-topic articles "
    "to one category only. NYT timestamps are UTC while the FRED rate is a New York noon fix, so a "
    "few evening articles land a day late (conservative, never early).",
    "<b>Task 3 plan:</b> (1) run the VADER + TF-IDF/LSA layer on the article file; (2) multi-label "
    "categories and an uncertainty lexicon; (3) weight or filter by NYT section/desk (World, Business "
    "vs Opinion); (4) add a volatility target (|r| or 5-day realised volatility), where geopolitical "
    "news is more likely to matter than for direction; (5) rolling-origin evaluation over several "
    "test windows to reduce the variance of the comparison.",
])]

story += [P("Appendix — repository layout and how to run", H2), P(
    "<font face='Courier' size=8>data/raw/ (Task 1 outputs) • data/processed/train.csv, val.csv, "
    "test.csv, full_features.csv • src/ (modules above) • notebook/task2_eda_and_baselines.ipynb • "
    "results/ (metrics_val.csv, metrics_test.csv, predictions_test.csv, arima_order_search.csv, "
    "arimax_coefficients.csv, run_summary.json) • reports/ (this PDF, figures)</font><br/>"
    "Run: <font face='Courier'>pip install -r requirements.txt; python -m src.run_pipeline; "
    "python reports/make_diagram.py; python reports/make_report.py</font>")]


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "NLP Project — Task 2: Pipeline Proposal & Baseline Experimentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                        topMargin=1.7 * cm, bottomMargin=1.8 * cm, title="Task 2 Report",
                        author="NLP Project Team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("saved", OUT)
