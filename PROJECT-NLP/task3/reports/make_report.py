"""Builds reports/Task3_Report.pdf (max 5 pages) from results/ and reports/figures/.
    python reports/make_report.py
Edit TEAM / MEMBERS below, then re-run.
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

TEAM = ""   # optional team name
MEMBERS = ("Rafaela Anabel Purba (24/547080/PA/23218) &nbsp;·&nbsp; "
           "Aufaa Azzahra Aryawan (24/546854/PA/23215) &nbsp;·&nbsp; "
           "Aurora Cindita Syachbudtari (24/532921/PA/22554)")

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, PROC = ROOT / "results", ROOT / "reports" / "figures", ROOT / "data" / "processed"
OUT = ROOT / "reports" / "Task3_Report.pdf"

res = pd.read_csv(RES / "ablation_all.csv")
reg = pd.read_csv(RES / "regression_metrics.csv")
roll = pd.read_csv(RES / "rolling_origin.csv")
vroll = pd.read_csv(RES / "rolling_origin_volatility.csv")
summ = json.load(open(RES / "run_summary.json"))
corp = json.load(open(RES / "corpus_summary.json"))
ana = json.load(open(RES / "analysis_summary.json"))
meta = json.load(open(PROC / "feature_groups.json"))
imp_g = pd.read_csv(RES / "importance_groups.csv")
imp_f = pd.read_csv(RES / "importance_features.csv")
full = pd.read_csv(PROC / "full_features.csv", parse_dates=["date"]).dropna(subset=["target_ret_next"])
sp = full.groupby("split").agg(s=("date", "min"), e=("date", "max"), n=("date", "size"),
                               up=("target_up_next", "mean"))

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=10.9,
                   alignment=TA_JUSTIFY, spaceAfter=3)
BL = ParagraphStyle("bl", parent=B, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
H1 = ParagraphStyle("h1", parent=B, fontName="Helvetica-Bold", fontSize=11, leading=13,
                    spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("h2", parent=B, fontName="Helvetica-Bold", fontSize=9.2, leading=11,
                    spaceBefore=3, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=B, fontSize=7.3, leading=8.8, textColor=colors.HexColor("#444444"),
                     alignment=0)
TC = ParagraphStyle("tc", parent=B, fontSize=7.2, leading=8.4, alignment=0, spaceAfter=0)


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def table(rows, widths, hl_rows=(), fs=7.2, head_bg="#dfe7f1"):
    data = [[Paragraph(str(c), TC) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_bg)),
          ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#555555")),
          ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
    for r, c in hl_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(c)))
    t.setStyle(TableStyle(st))
    return t


def img(name, width):
    from PIL import Image as PI
    p = FIG / name
    w, h = PI.open(p).size
    return Image(str(p), width=width, height=width * h / w)


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


# ---------------------------------------------------------------- numbers used in the text
def row(track, split, mid):
    r = res[(res.track == track) & (res.split == split) & (res.model_id == mid)]
    return r.iloc[0] if len(r) else None


best_rep = summ["best_representation_by_val"]
bA, bB, bC = summ["best_A"], summ["best_B"], summ["best_C"]
tA, tB, tC = row("direction", "test", bA), row("direction", "test", bB), row("direction", "test", bC)
vA, vB, vC = row("direction", "val", bA), row("direction", "val", bB), row("direction", "val", bC)
rmean = roll.groupby("model_id")[["Accuracy", "MacroF1", "MCC", "AUC"]].mean()
vmean = vroll.groupby("model_id")[["Accuracy", "MacroF1", "MCC", "AUC"]].mean()
mc = summ["mcnemar_test"][f"{bA} vs {bC}"]
emb = ana["embedding_space"]
ea = ana["error_analysis"]


def ablation_rows():
    """One row per block: the model (LogReg / LightGBM) chosen by VALIDATION macro-F1."""
    d = res[res.track == "direction"]
    va, te = d[d.split == "val"].set_index("model_id"), d[d.split == "test"].set_index("model_id")
    blocks = [("A", "A  time series only (Task 2 baseline)"),
              ("B", "B  + Task 2 basic NLP (lexicon, VADER, LSA)"),
              ("C-w2v", "C  + Word2Vec (static, in-domain)"),
              ("C-glove", "C  + GloVe 6B (static, general)"),
              ("C-minilm", "C  + MiniLM (contextual, general) – headline"),
              ("C-minilm_ctx", "C  + MiniLM – headline + GDELT event context"),
              ("C-finbert", "C  + FinBERT (contextual, finance) + sentiment"),
              ("C-topics", "C  + LDA / embedding-cluster topics"),
              ("C-entity", "C  + entity filter (nations, central banks)"),
              ("C-full", f"C  + B + {best_rep} + topics + entity")]
    out = [["Model", "chosen clf (h)", "Val F1", "Test Acc", "Test F1", "Test MCC", "Test AUC",
            "Roll-6 AUC"]]
    for bid, name in blocks:
        cands = [m for m in (f"{bid}|LogReg", f"{bid}|LGBM") if m in va.index]
        m = max(cands, key=lambda k: va.loc[k, "MacroF1"])
        r, v = te.loc[m], va.loc[m]
        ro = rmean.loc[m, "AUC"] if m in rmean.index else float("nan")
        out.append([name, f"{m.split('|')[1]} (h={int(v.halflife)})", f3(v.MacroF1), f3(r.Accuracy),
                    f3(r.MacroF1), f3(r.MCC), f3(r.AUC), f3(ro)])
    for mid, name in ((f"C-late-{best_rep}|LogReg", f"C  late fusion p(time), p({best_rep})"),
                      ("C-attn-minilm|NN", "C  attention fusion, MiniLM stream"),
                      ("C-attn-finbert|NN", "C  attention fusion, FinBERT stream")):
        r, v = te.loc[mid], va.loc[mid]
        out.append([name, "NN" if "NN" in mid else "LogReg", f3(v.MacroF1), f3(r.Accuracy),
                    f3(r.MacroF1), f3(r.MCC), f3(r.AUC), "–"])
    maj = te.loc["Majority"]
    out.append(["Naive: always UP (majority of train+val)", "–", "–", f3(maj.Accuracy),
                f3(maj.MacroF1), "0.000", "–", "–"])
    return out


def vol_rows():
    d = res[res.track == "volatility"]
    va, te = d[d.split == "val"].set_index("model_id"), d[d.split == "test"].set_index("model_id")
    out = [["Volatility target", "Val F1", "Test F1", "Test MCC", "Test AUC", "Roll-6 F1",
            "Roll-6 AUC"]]
    for mid in ("A|LGBM", "B|LGBM", f"C-{best_rep}|LGBM", "C-entity|LGBM", "C-full|LGBM"):
        out.append([mid, f3(va.loc[mid, "MacroF1"]), f3(te.loc[mid, "MacroF1"]), f3(te.loc[mid, "MCC"]),
                    f3(te.loc[mid, "AUC"]), f3(vmean.loc[mid, "MacroF1"]), f3(vmean.loc[mid, "AUC"])])
    return out


def reg_rows():
    t = reg[reg.split == "test"]
    out = [["Regression (test, bp)", "RMSE", "MAE", "Dir. acc."]]
    for _, r in t.iterrows():
        out.append([r.model, f"{r.RMSE_bp:.2f}", f"{r.MAE_bp:.2f}", f3(r.DirAcc)])
    return out


# ---------------------------------------------------------------- document
story = []
story += [P("<b>Global Geopolitical Event Prediction: Impact on USD Exchange Rates</b>",
            ParagraphStyle("t", parent=B, fontSize=14, leading=17, alignment=1)),
          P("Task 3 — Model Refinement &amp; Multimodal Integration",
            ParagraphStyle("st", parent=B, fontSize=10, alignment=1, textColor=colors.HexColor("#555555"))),
          P((f"Team: {TEAM} &nbsp;&nbsp; " if TEAM else "") + f"Members: {MEMBERS}",
            ParagraphStyle("m", parent=B, fontSize=8, alignment=1)), Spacer(1, 4)]

story += [P("<b>Summary.</b> We replace Task 2's count features with learned text representations and fuse them "
            "with exchange-rate history. Text is 71,520 headlines recovered from the GDELT URLs of Task 1 "
            f"(60 per trading day); the target is the next-day direction of Bank Indonesia's JISDOR USD/IDR rate "
            f"({sp.loc['train','s'].date()} – {sp.loc['test','e'].date()}, {int(sp.n.sum())} trading days). "
            "Five representations (in-domain Word2Vec, GloVe, MiniLM, MiniLM + event context, FinBERT), "
            "topic models, entity filtering, exponential-decay pooling and three fusion strategies (early, late, "
            "attention) are compared in an A/B/C ablation. Result: text features <b>improve every validation score</b> "
            f"(best C {vC.MacroF1:.3f} macro-F1 vs A {vA.MacroF1:.3f}) but <b>do not carry over to the held-out test "
            f"period</b> (C {tC.Accuracy:.3f} vs A {tA.Accuracy:.3f} accuracy, McNemar p = {mc['p_mcnemar']:.2f}). "
            "Over six rolling-origin folds the text model has a slightly better ranking quality (AUC "
            f"{rmean.loc[bC,'AUC']:.3f} vs {rmean.loc[bA,'AUC']:.3f}), and for a <b>volatility</b> target the entity "
            f"features help more clearly (rolling AUC {vmean.loc['C-entity|LGBM','AUC']:.3f} vs "
            f"{vmean.loc['A|LGBM','AUC']:.3f}). Geopolitical news looks more informative about <i>how much</i> the "
            "rupiah moves than about its <i>direction</i>.")]

# ---------------------------------------------------------------- 1
story.append(P("1. Data used in Task 3 and changes from Task 2", H1))
story += bullets([
    "<b>Target – Bank Indonesia JISDOR USD/IDR.</b> Task 2 used the FRED broad dollar index. Task 3 switches to the "
    "course-mandated Bank Indonesia series produced in Task 1 and re-runs the Task 2 baseline (A) and basic-NLP "
    "model (B) on it, so all three ablation arms share the same days, split and labels. UP = USD/IDR rises "
    "(the rupiah weakens). The trading-day conventions are those of Task 2: no forward-fill, and the target is "
    "r<sub>t+1</sub> = ln P<sub>t+1</sub>/P<sub>t</sub>. "
    f"Split 70/15/15: train {sp.loc['train','s'].date()}–{sp.loc['train','e'].date()} ({sp.loc['train','n']} d), "
    f"val –{sp.loc['val','e'].date()} ({sp.loc['val','n']} d), test –{sp.loc['test','e'].date()} "
    f"({sp.loc['test','n']} d). The UP share on test is {sp.loc['test','up']:.1%}, so 'always UP' scores "
    f"{sp.loc['test','up']:.1%}.",
    "<b>Text: headlines recovered from GDELT URLs.</b> Task 2's text layer never ran because the NYT article file was "
    "missing. Task 1's GDELT table has no article text either (its <i>event_text</i> is a string of CAMEO codes). "
    "Most news sites put the headline in the URL slug "
    "(<i>/russia-launches-massive-missile-strike-kyiv-2022-10-10/</i>), so we take the path segment with the most "
    f"alphabetic words and strip ids and dates. Filtering: {corp['articles_with_headline']:,} articles from 84 "
    "whitelisted international outlets (from 10.6 M events: 2.0 M whitelisted, 1.65 M with a usable 5–30-word "
    "slug), collapsed to one row per URL. Alignment follows the Task 1 rule (≤16:00 WIB → same BI day; later, "
    "weekend or holiday → next BI day). Syndicated duplicates are removed per day, and the 60 headlines with the "
    "most covering sources (GDELT NumSources) are kept per day → "
    f"{corp['corpus_rows']:,} headlines over {corp['days_covered']:,} days.",
])

# ---------------------------------------------------------------- 2
story.append(P("2. Feature extraction strategy", H1))
story.append(P("We use four representation families, chosen to test two contrasts that Task 3 asks about: "
               "static vs contextual, and general vs domain-specific. All of them run on CPU and none is "
               "fine-tuned on our labels, so the encoders cannot leak the target.", B))
story += bullets([
    "<b>Word2Vec (static, in-domain).</b> Skip-gram with negative sampling (100-d, window 4, 5 negatives, "
    "sub-sampling 10<sup>-4</sup>), implemented in PyTorch and trained on 300 k <i>training-period</i> headlines only. "
    "Tokenisation is whitespace over the lower-cased slug. The headline vector is the SIF-weighted mean "
    "(a/(a+p(w)), a = 10<sup>-3</sup>) with the first principal component removed (Arora et al., 2017), which "
    "stops frequent words such as 'says' or 'new' from dominating.",
    "<b>GloVe 6B-300d (static, general).</b> Wikipedia + Gigaword co-occurrence vectors; the headline is the mean of "
    "its word vectors. GloVe knows world entities (countries, leaders) that a 5-year headline corpus sees too rarely.",
    "<b>all-MiniLM-L6-v2 (contextual, general).</b> A 6-layer transformer distilled for sentence similarity. "
    "WordPiece tokens (max 64), attention-masked mean pooling, L2-normalised. We use the pooling the model was "
    "trained with, because CLS pooling is not trained in this model.",
    "<b>FinBERT (contextual, finance-domain).</b> BERT-base further pre-trained on financial text and fine-tuned on "
    "Financial PhraseBank. We use the masked mean of the last layer (768-d) plus the softmax of its sentiment head, "
    "P(pos/neg/neu). The question is whether market-tone language transfers to geopolitical headlines.",
    "<b>Headline vs longer text.</b> GDELT has no article body, so our 'longer text' variant appends the coded event "
    "(<i>'… . russia fights ukraine'</i>: actor 1 + CAMEO verb + actor 2) and encodes it with MiniLM (minilm_ctx).",
])
story.append(P("<b>From many headline vectors to one daily row.</b> With 826 training days, raw 384–768-d inputs "
               "would over-fit, so each representation is compressed into 18 interpretable daily features: (i) "
               "salience-weighted mean vector (weight log(1+NumSources)) → PCA(8) fit on training days; (ii) "
               "cosine similarity to 8 hand-written <i>concept anchors</i> (escalation, de-escalation, sanctions/trade, "
               "monetary, energy, EM stress, US politics, China–Taiwan); (iii) <i>semantic novelty</i> = "
               "1 − cos(today's centroid, previous-20-day centroid), the embedding analogue of Task 2's count shock; "
               "(iv) <i>dispersion</i> (one dominant story vs scattered news). <b>Temporal decay:</b> each block can be "
               "replaced by an exponentially weighted mean over past trading days with half-life h ∈ {0, 1, 3, 5}, "
               "chosen on validation. <b>Topics:</b> LDA (12 topics) on word counts, plus BERTopic-style k-means(12) on "
               "MiniLM vectors labelled by class-based TF-IDF, both giving daily topic shares. <b>Entities:</b> a "
               "gazetteer over the headline and GDELT's actor names (central banks/Fed, US, China, Russia–Ukraine, "
               "Middle East, Indonesia/ASEAN, economy) gives daily share and shock features, plus an embedding block "
               f"computed only on the {meta['entity_relevant_share']:.0%} of headlines that mention a nation or "
               "central bank. Every unsupervised fit (Word2Vec, TF-IDF/LSA, LDA, k-means, PCA) uses training-period "
               "text only.", B))

# ---------------------------------------------------------------- 3
story.append(KeepTogether([P("3. Multimodal architecture", H1), img("architecture.png", 17.6 * cm)]))
story.append(P("Figure 1. X<sub>text</sub> (orange) and X<sub>time</sub> (blue) are merged in three ways. "
               "<b>Early fusion</b> concatenates the 9 market features with a text block and feeds logistic regression "
               "(standardised, C tuned) or LightGBM (leaves, learning rate and trees tuned). <b>Late fusion</b> trains "
               "separate time-only and text-only models and averages their probabilities, with weight w tuned on "
               "validation. <b>Attention fusion</b> is a small network that reads the raw stream of 60 headline vectors "
               "per day: tanh projection to 32-d, a learned softmax attention over headlines, concatenated with a 16-d "
               "MLP encoding of X<sub>time</sub>, then a logistic head. It uses AdamW, dropout 0.3, epochs chosen on "
               "validation and a 5-seed ensemble. Every model is tuned on validation, refit on train+val and scored "
               "once on test.", CAP))

# ---------------------------------------------------------------- 4
story.append(P("4. Ablation matrix and discussion", H1))
story.append(table(ablation_rows(), [5.6 * cm, 2.1 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm,
                                     1.6 * cm],
                   hl_rows=[(1, "#e3edf8"), (2, "#e5f2e8")]))
story.append(P("Table 1. Next-day direction of USD/IDR. For each block, the classifier and decay half-life h with "
               "the best <i>validation</i> macro-F1 are shown; Test = scored once on "
               f"{sp.loc['test','n']} test days. Roll-6 AUC = mean over six expanding-window folds covering val+test. "
               "With 178 test days the 95 % noise band on accuracy is about ±7 points.", CAP))
story += bullets([
    f"<b>Validation favours text, test does not.</b> On validation all text blocks except FinBERT beat Model A "
    f"(macro-F1 {vA.MacroF1:.3f}). The best ({bC}) reaches {vC.MacroF1:.3f} and topics {row('direction','val','C-topics|LGBM').MacroF1:.3f}. "
    f"On test, {bC} scores {tC.Accuracy:.3f} accuracy against {tA.Accuracy:.3f} for A and {tB.Accuracy:.3f} for B. "
    f"McNemar: A-only correct on {mc['only_first_right']} days, C-only on {mc['only_second_right']} (p = {mc['p_mcnemar']:.2f}). "
    "So the validation gains reflect the 9-month validation regime rather than a stable signal. The test period "
    "(Dec 2025 – Aug 2026) is a trending, mostly-UP rupiah depreciation, where 'always UP' "
    f"({sp.loc['test','up']:.1%}) beats every model.",
    "<b>Representation comparison.</b> Static vectors are competitive with transformers. GloVe and in-domain "
    "Word2Vec win on validation, and C-minilm_ctx|LogReg has the highest test MCC of all models "
    f"({row('direction','test','C-minilm_ctx|LogReg').MCC:.3f}). FinBERT is the weakest on validation: its "
    "domain is corporate and market tone, and its sentiment head labels 53 % of headlines neutral and 43 % "
    "negative (conflict news reads as uniformly negative), so its tone barely varies from day to day. "
    "Domain-specific pre-training therefore did <i>not</i> transfer to geopolitics. Adding the coded event to "
    "the headline raised MiniLM's test MCC and AUC with both classifiers (LogReg AUC "
    f"{row('direction','test','C-minilm|LogReg').AUC:.3f} → {row('direction','test','C-minilm_ctx|LogReg').AUC:.3f}), "
    "a sign that 'who did what to whom' carries signal.",
    f"<b>Fusion.</b> Late fusion chose w<sub>time</sub> = {summ['chosen'][f'direction|C-late-{best_rep}|LogReg']['w_time']:.1f}, "
    "i.e. it trusted text alone on validation, and generalised worst. The attention networks were the most "
    "stable between validation and test but stayed near chance. Early fusion with a regularised LogReg is the "
    "safest choice at this sample size.",
    f"<b>Robustness.</b> Averaged over 6 rolling folds, {bC} has higher AUC ({rmean.loc[bC,'AUC']:.3f} vs "
    f"{rmean.loc[bA,'AUC']:.3f}) and MCC ({rmean.loc[bC,'MCC']:.3f} vs {rmean.loc[bA,'MCC']:.3f}) than A. That is "
    "a small, consistent ranking edge that the 0.5 threshold does not turn into accuracy.",
])
vt = Table([[table(vol_rows(), [2.7 * cm, 1.1 * cm, 1.1 * cm, 1.2 * cm, 1.2 * cm, 1.3 * cm, 1.3 * cm]),
             table(reg_rows(), [3.8 * cm, 1.25 * cm, 1.25 * cm, 1.2 * cm])]],
           colWidths=[10.1 * cm, 7.7 * cm])
vt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(KeepTogether([vt, P(
    f"Table 2 (left). Secondary target: high volatility, |r<sub>t+1</sub>| > train median "
    f"({meta['highvol_threshold'] * 1e4:.1f} bp); LightGBM, h tuned on validation. "
    "Table 3 (right). Regression of r<sub>t+1</sub>; ARIMA order by AIC on train, frozen-parameter walk-forward.",
    CAP)]))
story.append(P(
    f"<b>Volatility vs direction.</b> For volatility, entity features add the most: test MCC "
    f"{row('volatility','test','C-entity|LGBM').MCC:.3f} vs {row('volatility','test','A|LGBM').MCC:.3f}, and "
    f"rolling AUC {vmean.loc['C-entity|LGBM','AUC']:.3f} vs {vmean.loc['A|LGBM','AUC']:.3f}. On validation, "
    f"however, {best_rep} was preferred. This direction of effect matches the safe-haven story: shocks involving "
    "nations and central banks raise uncertainty, but whether the rupiah weakens depends on factors the "
    "headlines do not reveal. In the regression track nothing beats the random walk by more than 0.3 bp RMSE, "
    "as in Task 2.", B))

# ---------------------------------------------------------------- 5
story.append(P("5. Open exploratory analysis", H1))
story.append(img("embedding_space.png", 17.6 * cm))
story.append(P(f"Figure 2. Left: t-SNE of 4,000 MiniLM headline vectors from the 20 % highest- vs 20 % lowest-"
               f"|r<sub>t+1</sub>| days. A 15-NN classifier separates them at only {emb['knn15_highlow_acc']:.3f}. "
               f"Middle/right: {emb['map']} of daily mean vectors. Next-day volatility is scattered, while the year is "
               "strongly ordered: news topics drift over time, which explains why features tuned on one period "
               "fail on the next.", CAP))
top_txt = ", ".join(f"{r.feature} ({r.auc_drop:+.3f})" for r in imp_f.head(6).itertuples())
story.append(img("importance.png", 15.5 * cm))
story.append(P(f"Figure 3. Permutation importance of {ana['importance']['model']} on test (AUC drop when a feature "
               "or group is shuffled, 30 repeats).", CAP))
story += bullets([
    f"<b>What drives predictions.</b> Market features matter most (group AUC drop "
    f"{imp_g.set_index('group').loc['market','auc_drop_mean']:.3f}), followed by the {best_rep} block "
    f"({imp_g.set_index('group').loc[best_rep,'auc_drop_mean']:.3f}). The top single features are {top_txt}. "
    "The text features that matter are the interpretable ones: the de-escalation and escalation anchors and "
    "semantic novelty, not the PCA axes.",
    "<b>Topics.</b> The k-means clusters are clean themes (e.g. clu_0 <i>gaza israel hamas hezbollah</i>, clu_9 "
    "<i>ukraine russia putin</i>, clu_5 <i>china taiwan</i>, clu_11 <i>north korea missile</i>). On train+val, the "
    "Gaza/Israel cluster share has the strongest correlation with next-day |move| (Spearman ρ = 0.094, p = 0.003). "
    "Middle-East and Russia–Ukraine count shocks correlate <i>negatively</i> with the next day's move "
    "(ρ ≈ −0.07 to −0.08), consistent with the market having priced the event by the time coverage peaks.",
    f"<b>Attention.</b> The attention network concentrates on military headlines (e.g. 'a us navy aircraft carrier "
    "is circling africa to reach the middle east', 'uae says drones targeted barakah', 'saudi military downs drones "
    "targeting shaybah oil field'). It learned by itself to look at the conflict / energy-risk stories that an "
    "FX analyst would pick.",
    f"<b>Error analysis</b> (best A vs best C, test): text helped on {ea['cases'].get('text helped')} days and hurt "
    f"on {ea['cases'].get('text hurt')}. C is worse than A mainly on low-novelty days and on days with small "
    "moves, i.e. when nothing new happened, text adds noise. The 'text hurt' days are dominated by non-FX "
    "headlines that survive the salience filter (crime trials, celebrity stories, Indian state politics). This "
    "is the main limitation of a URL-slug corpus.",
])
story.append(P("6. Conclusion and plan for Task 4", H1))
story.append(P("Advanced representations do not measurably improve next-day <i>directional</i> forecasting of USD/IDR "
               "out of sample. They give a small rolling-origin AUC gain, and they help more for <i>volatility</i>, "
               "especially entity-filtered features. For Task 4 we will (1) test the hypothesis formally on the "
               "volatility target with Diebold–Mariano / DeLong tests over the rolling folds; (2) tighten the corpus "
               "with an FX-relevance classifier, since the error analysis shows that irrelevant headlines hurt; "
               "(3) restrict modelling to high-novelty days, the regime where news should matter; and (4) report "
               "the time-drift result as a central finding.", B))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                        title="Task 3 Report - Model Refinement & Multimodal Integration")


def footer(c, d):
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.6 * cm, 0.8 * cm, "NLP Project — Task 3: Model Refinement & Multimodal Integration")
    c.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, str(d.page))


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("saved", OUT)
