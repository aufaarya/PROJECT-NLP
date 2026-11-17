"""Draws the pipeline architecture diagram -> reports/figures/pipeline_diagram.png"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "pipeline_diagram.png"
INK, INK2 = "#0b0b0b", "#52514e"
FILL = {"data": "#eef4fc", "nlp": "#fdeee6", "mkt": "#e8f6f0", "model": "#f3f2ee", "eval": "#fdf5e1"}
EDGE = {"data": "#2a78d6", "nlp": "#eb6834", "mkt": "#1baf7a", "model": "#8a8983", "eval": "#c98500"}

fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(0, 130); ax.set_ylim(0, 74); ax.axis("off")


def box(x, y, w, h, title, body, kind, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=7.6, color=INK2,
            linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1,
                                 color=INK2, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.6, (p[1] + q[1]) / 2, text, fontsize=7, color=INK2,
                style="italic")


def section(y, label):
    ax.text(1, y, label, fontsize=8, fontweight="bold", color=INK2, rotation=90, va="center")


# ---- row 1: sources (Task 1) ----
section(67.5, "SOURCES\n(Task 1)")
box(6, 62, 36, 11, "NYT Archive API", "2021–2025, ~250k articles\nheadline + abstract + keywords\n→ keyword lexicon → 5 categories", "data")
box(48, 62, 30, 11, "FRED  DTWEXBGS", "Nominal Broad USD Index\ndaily, business days", "data")
box(84, 62, 42, 11, "Article-level file (optional)", "nyt_geopolitical_articles_clean.csv\ntitle, abstract, date, category\n(not in current run → dashed path)", "data", dashed=True)

# ---- row 2: alignment ----
section(52, "ALIGN")
box(6, 47, 120, 10, "Trading-day alignment  (src/data_prep.py)",
    "drop weekends/holidays  •  news in (t_prev, t] summed onto trading day t (Sat+Sun+Mon → Mon)\n"
    "target:  r(t+1) = ln P(t+1) / P(t)    and    UP(t+1) = 1[r(t+1) > 0]", "model")
arrow((24, 62), (24, 57.8)); arrow((63, 62), (63, 57.8)); arrow((108, 62), (108, 57.8))

# ---- row 3: features ----
section(36, "FEATURES\n(src/features.py)")
box(6, 29, 36, 13.5, "Market features", "returns r(t)…r(t−4)\nvolatility 5d, 20d\nmomentum 5d, 20d", "mkt")
box(48, 29, 36, 13.5, "News-volume features (lexicon)", "log1p count per category + total\ncategory share of total\nnews SHOCK z-score vs past 20d\n(past-only window, shift 1)", "nlp")
box(90, 29, 36, 13.5, "Text features", "VADER tone of title+abstract\nTF-IDF (1–2 gram) → LSA topics\nvocab + SVD fit on TRAIN only\n(auto-enabled if file present)", "nlp", dashed=True)
arrow((24, 47), (24, 43.3)); arrow((66, 47), (66, 43.3)); arrow((108, 47), (108, 43.3))

# ---- chronological split bar ----
ax.add_patch(FancyBboxPatch((6, 21), 120, 4.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                            fc="white", ec=INK2, lw=0.8))
ax.add_patch(plt.Rectangle((6, 21), 84, 4.5, fc="#f3f2ee", ec="none"))
ax.add_patch(plt.Rectangle((90, 21), 18, 4.5, fc="#e3eefb", ec="none"))
ax.add_patch(plt.Rectangle((108, 21), 18, 4.5, fc="#fde9df", ec="none"))
ax.text(48, 23.2, "TRAIN 70%  (Feb 2021 – Jul 2024)", ha="center", va="center", fontsize=8, fontweight="bold")
ax.text(99, 23.2, "VAL 15%", ha="center", va="center", fontsize=8, fontweight="bold")
ax.text(117, 23.2, "TEST 15%", ha="center", va="center", fontsize=8, fontweight="bold")
ax.text(66, 18.6, "strict chronological split — no shuffling; scalers, TF-IDF, ARIMA params fit on past data only",
        ha="center", fontsize=7.3, color=INK2, style="italic", bbox=dict(fc="white", ec="none", pad=1))
for x in (24, 66, 108):
    arrow((x, 29), (x, 25.8))

# ---- row 4: models ----
section(7.5, "MODELS\n& EVAL")
box(6, 1, 38, 13, "BASELINE  (history only)", "Random walk (0), Majority, Persistence\nARIMA(p,0,q), order by AIC on train\nLogReg / GBM on market features", "mkt")
box(50, 1, 38, 13, "COMBINED  (history + NLP)", "ARIMAX = ARIMA + lagged news exog\nLogReg / GBM on market + news\n(+ text features when available)", "nlp")
box(94, 1, 32, 13, "Evaluation  (src/evaluate.py)", "Regression: RMSE, MAE, DirAcc\nDirection: Acc, macro-F1, MCC,\nAUC, binomial test vs 50%\nVAL = tuning • TEST = once", "eval")
arrow((25, 21), (25, 14.8)); arrow((69, 21), (69, 14.8))
ax.add_patch(FancyArrowPatch((44.6, 7.5), (49.4, 7.5), arrowstyle="<|-|>", mutation_scale=10, lw=1.1, color=INK2))
ax.text(47, 9.2, "vs.", ha="center", fontsize=7.5, color=INK2, style="italic")
arrow((88.6, 7.5), (93.4, 7.5))
ax.text(91, 9.2, "both", ha="center", fontsize=7, color=INK2, style="italic")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("saved", OUT)
