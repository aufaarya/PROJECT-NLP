"""
config.py  (Task 3)
-------------------
Every path, constant and design decision used by the Task 3 pipeline lives here, so the
report, the notebook and the code cannot silently drift apart.

Data streams used in Task 3
---------------------------
* Target   : Bank Indonesia JISDOR USD/IDR daily rate (Task 1 output `bi_usd_cleaned.csv`),
             1 Sep 2021 - 1 Sep 2026. This is the course-mandated "ultimate source".
* Text     : GDELT 2.0 Event records (Task 1 output `news_cleaned.parquet`, already filtered
             to CAMEO root codes 10-20 = demand ... mass violence). GDELT gives no article body,
             but every record carries the SOURCEURL, and the URL slug of almost every news site
             is the article headline ("/us-imposes-new-sanctions-on-russian-banks-2023..."),
             so we recover one headline per article from the slug.
* Task 2   : the original FRED DTWEXBGS + NYT-count experiment is kept untouched in
             src/task2, data/task2_fred_nyt, results/task2 for reference.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"          # large intermediate files (git-ignored)
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
CACHE = ROOT / "data" / "interim" / "cache"       # embedding caches (.npy)

BI_FILE = DATA_RAW / "bi_usd_cleaned.csv"
TASK1_DAILY_FILE = DATA_RAW / "daily_aligned_dataset.csv"   # Task 1 daily event counts
# Full GDELT event file from Task 1 (~0.9 GB). Not committed to GitHub; point this to the
# Task 1 output. A 1,000-row sample is in data/raw/news_cleaned_sample.csv.
GDELT_EVENTS_FILE = ROOT.parents[1] / "PROJECT-NLP" / "data" / "cleaned" / "news_cleaned.parquet"

HEADLINES_ALL = DATA_INTERIM / "headlines_all.parquet"        # every usable headline (whitelist)
CORPUS_FILE = DATA_PROCESSED / "headline_corpus.parquet"      # top-K salient headlines / day

# ----------------------------------------------------------------------------
# Temporal alignment (identical to the Task 1 rule)
# ----------------------------------------------------------------------------
TIMEZONE = "Asia/Jakarta"     # BI publishes JISDOR in WIB
CUTOFF_HOUR = 16              # news after 16:00 WIB -> next BI trading day
# Weekend / holiday news -> next BI trading day (no news is discarded).

# ----------------------------------------------------------------------------
# Corpus construction (filtering strategy, see news_corpus.py)
# ----------------------------------------------------------------------------
MIN_HEADLINE_WORDS = 5
MAX_HEADLINE_WORDS = 30
TOP_K_PER_DAY = 60            # most salient headlines kept per trading day for embeddings

# Internationally read English-language outlets / wires / aggregators. Local outlets are
# dropped: a city crime story coded as CAMEO 'assault' is not a geopolitical FX driver.
SOURCE_WHITELIST = {
    # wires and finance (course-listed sources first)
    "reuters.com", "apnews.com", "cnbc.com", "tradingeconomics.com", "bloomberg.com",
    "ft.com", "wsj.com", "marketwatch.com", "marketscreener.com", "investing.com",
    "businessinsider.com", "forbes.com", "fxstreet.com", "barrons.com", "economist.com",
    "finance.yahoo.com", "economictimes.indiatimes.com", "business-standard.com",
    # US / UK general
    "nytimes.com", "washingtonpost.com", "cnn.com", "edition.cnn.com", "foxnews.com",
    "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org", "politico.com",
    "politico.eu", "axios.com", "thehill.com", "usatoday.com", "latimes.com",
    "newsweek.com", "time.com", "voanews.com", "bbc.com", "bbc.co.uk", "theguardian.com",
    "independent.co.uk", "telegraph.co.uk", "dailymail.co.uk", "sky.com", "news.sky.com",
    "yahoo.com", "news.yahoo.com", "msn.com",
    # Europe / Middle East / international
    "dw.com", "france24.com", "euronews.com", "rfi.fr", "aljazeera.com", "arabnews.com",
    "timesofisrael.com", "jpost.com", "kyivindependent.com", "kyivpost.com",
    "themoscowtimes.com", "rferl.org", "globalsecurity.org", "eurasiareview.com",
    # Asia-Pacific (incl. Indonesia, relevant for USD/IDR)
    "scmp.com", "japantimes.co.jp", "straitstimes.com", "channelnewsasia.com",
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com",
    "indianexpress.com", "dawn.com", "koreaherald.com", "koreatimes.co.kr",
    "thejakartapost.com", "en.antaranews.com", "jakartaglobe.id", "abc.net.au",
    "smh.com.au", "rnz.co.nz", "nikkei.com", "asia.nikkei.com", "globaltimes.cn",
    "xinhuanet.com", "english.news.cn", "chinadaily.com.cn",
}

# ----------------------------------------------------------------------------
# Task 2 "basic NLP" lexicon (kept identical so Model B = Task 2 recipe)
# ----------------------------------------------------------------------------
CATEGORIES = ["military_conflict", "sanctions", "diplomacy", "trade_conflict",
              "political_crisis"]
CATEGORY_KEYWORDS = {
    "military_conflict": ["war", "conflict", "military", "invasion", "attack", "nato"],
    "sanctions": ["sanction", "embargo", "trade restriction"],
    "diplomacy": ["diplomatic", "negotiation", "peace talk", "treaty"],
    "trade_conflict": ["trade war", "tariff", "trade dispute"],
    "political_crisis": ["political crisis", "coup", "election", "government crisis"],
}

# ----------------------------------------------------------------------------
# Entity gazetteer for NER-style filtering (nations, central banks, economic events).
# GDELT's event coder already extracts Actor1/Actor2 names; we match those plus the
# headline against this list to build an "FX-relevant entity" sub-stream.
# ----------------------------------------------------------------------------
ENTITY_GROUPS = {
    "central_bank": ["federal reserve", "fed", "fomc", "powell", "central bank", "ecb",
                     "lagarde", "bank of japan", "boj", "pboc", "people's bank",
                     "bank indonesia", "bank of england", "interest rate", "rate hike",
                     "rate cut"],
    "us": ["united states", "u.s.", "us", "washington", "biden", "trump", "white house",
           "pentagon", "congress", "american"],
    "china": ["china", "chinese", "beijing", "xi", "taiwan", "taipei"],
    "russia_ukraine": ["russia", "russian", "moscow", "putin", "kremlin", "ukraine",
                       "ukrainian", "kyiv", "zelensky"],
    "middle_east": ["israel", "israeli", "gaza", "hamas", "iran", "iranian", "tehran",
                    "hezbollah", "houthi", "yemen", "saudi", "syria", "lebanon", "red sea"],
    "indonesia_asean": ["indonesia", "indonesian", "jakarta", "prabowo", "jokowi",
                        "asean", "rupiah"],
    "economy": ["tariff", "trade", "sanction", "oil", "inflation", "debt", "default",
                "recession", "economy", "economic", "market", "dollar", "currency",
                "export", "import", "opec"],
}

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
SHOCK_WINDOW = 20        # trading days used as the "normal level" for shock z-scores
RETURN_LAGS = 5
VOL_WINDOWS = (5, 20)
N_LSA_TOPICS = 10        # Task 2 TF-IDF/LSA layer
N_LDA_TOPICS = 12        # Task 3 topic model
N_EMB_PCA = 8            # daily embedding mean vector -> PCA components (fit on train)
DECAY_HALFLIFES = (1, 3, 5)   # trading days; candidates for exponential-decay pooling

# ----------------------------------------------------------------------------
# Embedding models (Hugging Face ids). All run on CPU.
# ----------------------------------------------------------------------------
GENERAL_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # general-purpose contextual
FINBERT_MODEL = "ProsusAI/finbert"                             # domain-specific contextual
GLOVE_MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"  # static, general
W2V_DIM = 100            # in-domain Word2Vec (skip-gram, trained on TRAIN headlines only)

# ----------------------------------------------------------------------------
# Chronological split (fractions of usable trading days, in time order)
# ----------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
RANDOM_STATE = 42
