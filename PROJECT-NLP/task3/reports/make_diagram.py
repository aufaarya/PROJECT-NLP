"""Draws reports/figures/architecture.png - the Task 3 multimodal architecture schematic.
    python reports/make_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "architecture.png"
C_TEXT, C_TIME, C_FUSE, C_OUT, C_DATA = "#F6D7B0", "#C9DDF2", "#D9EAD3", "#EADCF0", "#EEEEEE"


def box(ax, x, y, w, h, text, color, fs=7.2, bold=False, ls="-", ha="center", ec="#444444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=color, ec=ec, lw=0.8, ls=ls))
    tx = x + w / 2 if ha == "center" else x + 0.01
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.995, "Task 3 multimodal architecture: GDELT headlines (X_text) + BI USD/IDR "
            "history (X_time) -> next-day USD direction", fontsize=10.5, weight="bold", va="top")

    # ---------------- inputs
    box(ax, 0.00, 0.78, 0.15, 0.15, "GDELT 2.0 events\n(Task 1 output)\n10.6 M records\nCAMEO roots 10-20",
        C_DATA, bold=True, fs=7)
    box(ax, 0.00, 0.08, 0.15, 0.14, "Bank Indonesia\nJISDOR USD/IDR\n(Task 1 output)\n2021-09 .. 2026-09",
        C_DATA, bold=True, fs=7)

    # ---------------- text path, top row
    box(ax, 0.18, 0.78, 0.17, 0.15, "Headline recovery\nURL slug -> headline\nsource whitelist (84)\n"
        "5-30 words, 1 row / URL", C_TEXT)
    box(ax, 0.38, 0.78, 0.20, 0.15, "Temporal alignment\n<= 16:00 WIB -> same BI day\nlater / weekend / holiday\n"
        "-> next BI day; top-60 / day\nby NumSources", C_TEXT, fs=6.9)
    arrow(ax, 0.15, 0.855, 0.18, 0.855)
    arrow(ax, 0.35, 0.855, 0.38, 0.855)

    # ---------------- text representations (container)
    ax.add_patch(FancyBboxPatch((0.18, 0.30), 0.40, 0.41, boxstyle="round,pad=0.006,rounding_size=0.01",
                                fc="white", ec="#b07a2a", lw=1.0, ls="--"))
    ax.text(0.19, 0.695, "Headline representations",
            fontsize=7.4, weight="bold", color="#8a5a00", va="top")
    subs = [
        ("Model B (Task 2 recipe): lexicon category volume / mix / shock,\n"
         "VADER compound, TF-IDF (1-2 gram) -> LSA(10)", 0.575),
        ("Static: Word2Vec skip-gram (ours, train-only, SIF mean, 100-d);\n"
         "GloVe 6B (general, mean of word vectors, 300-d)", 0.485),
        ("Contextual: MiniLM-L6 (general, masked mean pool, 384-d);\n"
         "FinBERT (finance, masked mean 768-d + P(pos / neg / neu))", 0.395),
        ("Themes: LDA(12) on word counts; k-means(12) on MiniLM + c-TF-IDF;\n"
         "entity gazetteer: nations, central banks, economy", 0.315),
    ]
    for t, y in subs:
        box(ax, 0.19, y, 0.38, 0.075, t, C_TEXT, fs=6.6, ha="left")
    arrow(ax, 0.48, 0.78, 0.48, 0.71)

    # ---------------- daily pooling
    box(ax, 0.62, 0.42, 0.17, 0.29, "Daily pooling -> X_text\n\nsalience-weighted mean\n-> PCA(8, train-fit)\n"
        "8 concept-anchor cosines\nsemantic novelty (20 d)\ndispersion, topic shares\n"
        "exp. decay, half-life h\n(h tuned on validation)", C_TEXT, fs=6.9)
    arrow(ax, 0.58, 0.565, 0.62, 0.565)

    # ---------------- time path
    box(ax, 0.18, 0.08, 0.17, 0.14, "X_time (Model A)\nlog-return lags 0-4\nvolatility 5 / 20 d\nmomentum 5 / 20 d",
        C_TIME)
    arrow(ax, 0.15, 0.15, 0.18, 0.15)
    box(ax, 0.38, 0.03, 0.20, 0.09, "ARIMA(p,0,q) / ARIMAX\n(regression track, r_{t+1})", C_TIME, fs=6.9)
    arrow(ax, 0.35, 0.10, 0.38, 0.075)
    arrow(ax, 0.35, 0.185, 0.62, 0.185, color="#2f6db5", lw=1.3)
    ax.text(0.45, 0.195, "X_time", fontsize=7.5, color="#2f6db5", weight="bold")

    # ---------------- fusion
    box(ax, 0.62, 0.03, 0.37, 0.33,
        "FUSION  (all tuned on validation, scored once on test)\n\n"
        "1  Early fusion   [X_time || X_text] -> LogReg / LightGBM\n\n"
        "2  Late fusion    w * p(X_time) + (1 - w) * p(X_text)\n\n"
        "3  Attention fusion (neural)\n"
        "     raw headline stream [60 x d] (brown arrow) -> Linear + tanh\n"
        "     -> softmax attention pool -> 32-d day vector\n"
        "     || MLP(X_time) 16-d  -> logistic head  (5-seed ensemble)",
        C_FUSE, fs=6.9, ha="left")
    arrow(ax, 0.705, 0.42, 0.705, 0.36, lw=1.3, color="#8a5a00")
    ax.text(0.712, 0.385, "X_text", fontsize=7.5, color="#8a5a00", weight="bold")
    arrow(ax, 0.575, 0.43, 0.66, 0.36, color="#8a5a00")

    box(ax, 0.83, 0.47, 0.16, 0.24, "Targets (t+1)\n\nP(UP): USD/IDR rises\n(primary)\n\nP(high volatility)\n"
        "|r| > train median\n\nr_{t+1} (ARIMAX)", C_OUT, fs=6.9)
    arrow(ax, 0.91, 0.36, 0.91, 0.47, lw=1.3)

    ax.text(0.0, 0.0, "Leakage control: Word2Vec, TF-IDF/LSA, LDA, k-means and every PCA are fit on "
            "TRAIN-period text only; pre-trained encoders are frozen; rolling windows look backwards only.",
            fontsize=6.6, color="#444444")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
