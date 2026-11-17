"""
news_corpus.py
--------------
GDELT events (Task 1)  ->  one clean headline per article  ->  aligned, salience-ranked
headline corpus used by every text representation in Task 3.

Why this step is needed
-----------------------
Task 2's text layer (VADER + TF-IDF/LSA) never ran because the NYT article file was not
available. Task 1's GDELT table has 10.6 M event records but no article text: its
`event_text` field is a synthetic string of CAMEO codes ("actor1 X actor2 Y event_code 112"),
which no language model can read meaningfully. However, every record has the SOURCEURL and
most news sites put the headline in the URL slug:

    https://www.reuters.com/world/europe/russia-launches-massive-missile-strike-kyiv-2022-10-10/
        -> "russia launches massive missile strike kyiv"

Pipeline (one pass over the 86 parquet row groups, memory-safe):
  1. keep whitelisted international outlets (config.SOURCE_WHITELIST)       - source filter
  2. slug -> headline: pick the URL path segment with the most alphabetic
     words, strip file extensions / numeric ids / date tokens                - text recovery
  3. quality filter: 5-30 words, mostly alphabetic                          - noise filter
  4. collapse events -> articles (one row per URL; keep event metadata:
     max NumSources / NumMentions, mean tone, min Goldstein, actors)        - de-duplication
  5. timestamp = earliest GDELT DATEADDED (UTC) of the URL -> WIB
  6. align to BI trading days with the Task 1 rule (16:00 WIB cut-off,
     weekend / holiday news -> next trading day)                            - alignment
  7. drop exact duplicate headlines inside a trading day (syndicated copies)
  8. rank by salience (NumSources, then NumMentions) and keep TOP_K per day  - salience filter

Run:  python -m src.news_corpus
"""
import re
import time
from collections import Counter
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

# ----------------------------------------------------------------------------
# 1-3. URL slug -> headline
# ----------------------------------------------------------------------------
_EXT = re.compile(r"\.(s?html?|php|aspx?|cms|ece|story|amp|jsp)$", re.I)
_SPLIT = re.compile(r"[-_+.~]|%20|\s")
_ALPHA = re.compile(r"^[a-z]+$")
_SHORT_ALNUM = re.compile(r"^(?=.*[a-z])[a-z0-9]{2,4}$")      # g7, g20, f16, 5g, covid19 -> keep short ones
_MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct",
           "nov", "dec"}
# Section words that sometimes win as the "longest segment" but are not headlines.
_NOT_HEADLINE = {"news", "world", "article", "articles", "story", "politics", "business",
                 "amp", "index", "html", "www", "com", "video", "live", "opinion"}


def slug_to_headline(url: str):
    """Return the lower-case headline hidden in a news URL, or None if there is none."""
    try:
        path = unquote(urlsplit(url).path).lower()
    except Exception:
        return None
    best, best_n = None, 0
    for seg in path.split("/"):
        if not seg:
            continue
        seg = _EXT.sub("", seg)
        toks = [t for t in _SPLIT.split(seg) if t]
        if len(toks) < 3:
            continue
        words = []
        for t in toks:
            if _ALPHA.match(t):
                words.append(t)
            elif _SHORT_ALNUM.match(t) and not t.isdigit():
                words.append(t)
            # long alphanumeric ids (a1b2c3d4) and pure numbers (dates, ids) are dropped
        n_alpha = sum(1 for w in words if _ALPHA.match(w))
        # a real slug is mostly words; an id-like segment ("p5n3xz-20251021") is not
        if n_alpha > best_n and n_alpha >= 0.6 * len(toks):
            best, best_n = words, n_alpha
    if best is None:
        return None
    # trailing month names / 'amp' from date-stamped slugs carry no meaning
    while best and (best[-1] in _MONTHS or best[-1] in _NOT_HEADLINE):
        best = best[:-1]
    if not (config.MIN_HEADLINE_WORDS <= len(best) <= config.MAX_HEADLINE_WORDS):
        return None
    if sum(w in _NOT_HEADLINE for w in best) > len(best) / 2:
        return None
    return " ".join(best)


def _norm_source(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(r"^(www\d?|m|amp|mobile)\.", "", regex=True)


# ----------------------------------------------------------------------------
# 4-5. events -> articles (streamed over row groups)
# ----------------------------------------------------------------------------
COLS = ["recorded_utc", "source", "url", "event_root_code", "actor1", "actor2", "country",
        "num_mentions", "num_sources", "avg_tone", "goldstein_scale"]


def build_headlines_all(events_file=config.GDELT_EVENTS_FILE, out=config.HEADLINES_ALL,
                        verbose=True):
    pf = pq.ParquetFile(events_file)
    parts, stats = [], Counter()
    t0 = time.time()
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=COLS).to_pandas()
        stats["events"] += len(df)
        df["source"] = _norm_source(df["source"])
        df = df[df["source"].isin(config.SOURCE_WHITELIST)]
        stats["events_whitelisted"] += len(df)
        urls = df["url"].unique()
        heads = {u: slug_to_headline(u) for u in urls}
        df = df.assign(headline=df["url"].map(heads)).dropna(subset=["headline"])
        stats["events_with_headline"] += len(df)
        # collapse to one row per URL inside the row group
        g = df.groupby("url", sort=False)
        part = g.agg(recorded_utc=("recorded_utc", "min"), source=("source", "first"),
                     headline=("headline", "first"), n_events=("headline", "size"),
                     max_sources=("num_sources", "max"), max_mentions=("num_mentions", "max"),
                     tone_mean=("avg_tone", "mean"), goldstein_min=("goldstein_scale", "min"),
                     goldstein_mean=("goldstein_scale", "mean"),
                     root_code=("event_root_code", lambda x: x.mode().iat[0]),
                     actor1=("actor1", "first"), actor2=("actor2", "first"),
                     country=("country", "first")).reset_index()
        parts.append(part)
        if verbose and (rg % 10 == 0 or rg == pf.num_row_groups - 1):
            print(f"  row group {rg + 1}/{pf.num_row_groups}  "
                  f"articles so far {sum(len(p) for p in parts):,}  ({time.time() - t0:.0f}s)")
    art = pd.concat(parts, ignore_index=True)
    # the same URL can appear in several row groups -> merge once more
    art = art.sort_values("recorded_utc")
    agg = {"recorded_utc": "min", "source": "first", "headline": "first", "n_events": "sum",
           "max_sources": "max", "max_mentions": "max", "tone_mean": "mean",
           "goldstein_min": "min", "goldstein_mean": "mean", "root_code": "first",
           "actor1": "first", "actor2": "first", "country": "first"}
    art = art.groupby("url", sort=False).agg(agg).reset_index()
    stats["articles"] = len(art)
    out.parent.mkdir(parents=True, exist_ok=True)
    art.to_parquet(out, index=False)
    if verbose:
        print({k: f"{v:,}" for k, v in stats.items()})
    return art, dict(stats)


# ----------------------------------------------------------------------------
# 6. temporal alignment (Task 1 rule)
# ----------------------------------------------------------------------------
def align_to_trading_days(ts_utc: pd.Series, trading_dates, tz=config.TIMEZONE,
                          cutoff_hour=config.CUTOFF_HOUR):
    """Map UTC timestamps to the trading day whose close the news can first affect.

    * convert to local market time (WIB);
    * a story published on a trading day up to `cutoff_hour` belongs to that day;
    * after the cut-off, or on a weekend / holiday, it belongs to the NEXT trading day.
    Returns (aligned_date, reason)."""
    td = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().values
    local = pd.to_datetime(ts_utc, utc=True).dt.tz_convert(tz)
    day = local.dt.tz_localize(None).dt.normalize().values
    minutes = (local.dt.hour * 60 + local.dt.minute).values
    same = np.searchsorted(td, day, side="left")
    nxt = np.searchsorted(td, day, side="right")
    is_td = (same < len(td)) & (td[np.minimum(same, len(td) - 1)] == day)
    late = minutes > cutoff_hour * 60
    idx = np.where(is_td & ~late, same, nxt)
    ok = idx < len(td)
    out = np.full(len(day), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = td[idx[ok]]
    reason = np.select([is_td & ~late, is_td & late], ["same_day_before_cutoff",
                                                       "after_cutoff"], "weekend_or_holiday")
    return pd.to_datetime(out), reason


# ----------------------------------------------------------------------------
# 7-8. de-duplicate + salience top-K per day
# ----------------------------------------------------------------------------
def build_corpus(art: pd.DataFrame, trading_dates, top_k=config.TOP_K_PER_DAY):
    art = art.copy()
    art["date"], art["align_reason"] = align_to_trading_days(art["recorded_utc"], trading_dates)
    art = art.dropna(subset=["date"])
    # exact duplicate headline on the same trading day = syndicated copy -> keep most salient
    art = art.sort_values(["date", "max_sources", "max_mentions"],
                          ascending=[True, False, False])
    art = art.drop_duplicates(["date", "headline"])
    art["rank_in_day"] = art.groupby("date").cumcount()
    art["n_day_articles"] = art.groupby("date")["headline"].transform("size")
    corpus = art[art["rank_in_day"] < top_k].reset_index(drop=True)
    return art, corpus


def main():
    bi = pd.read_csv(config.BI_FILE, parse_dates=["date"])
    if config.HEADLINES_ALL.exists():
        print("[info] reusing", config.HEADLINES_ALL.name)
        art = pd.read_parquet(config.HEADLINES_ALL)
    else:
        print("Building headline table from", config.GDELT_EVENTS_FILE)
        art, _ = build_headlines_all()
    aligned, corpus = build_corpus(art, bi["date"])
    aligned.to_parquet(config.DATA_INTERIM / "headlines_aligned.parquet", index=False)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(config.CORPUS_FILE, index=False)
    corpus.head(2000).to_csv(config.DATA_PROCESSED / "headline_corpus_sample.csv", index=False)
    summary = {
        "articles_with_headline": len(art),
        "aligned_unique_per_day": len(aligned),
        "corpus_rows": len(corpus),
        "days_covered": corpus["date"].nunique(),
        "median_per_day": float(corpus.groupby("date").size().median()),
        "align_reasons": aligned["align_reason"].value_counts().to_dict(),
    }
    pd.Series(summary, dtype=object).to_json(config.RESULTS / "corpus_summary.json", indent=2)
    print(summary)


if __name__ == "__main__":
    main()
