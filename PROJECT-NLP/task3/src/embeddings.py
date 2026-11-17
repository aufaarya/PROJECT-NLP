"""
embeddings.py  -  headline -> vector, four representation families (Task 3 req. 1a)
----------------------------------------------------------------------------------
Static / non-contextual
  * w2v     : in-domain Word2Vec (skip-gram, negative sampling, 100-d) trained by us in
              PyTorch on TRAIN-period headlines only (gensim has no Python 3.14 wheel).
              Headline vector = SIF-weighted mean of word vectors (Arora et al., 2017):
              weight a/(a+p(w)) down-weights frequent words; the first principal component
              (fit on train) is removed.
  * glove   : general-purpose GloVe 6B 300-d (Wikipedia + Gigaword), mean of word vectors,
              loaded through sentence-transformers' average_word_embeddings model.
Contextual / transformer
  * minilm  : sentence-transformers/all-MiniLM-L6-v2 (general-purpose, 384-d). WordPiece
              tokenisation, max 64 tokens, attention-masked mean pooling, L2-normalised -
              the pooling the model was trained with for sentence similarity.
  * finbert : ProsusAI/finbert (BERT-base further pre-trained on financial text and
              fine-tuned on Financial PhraseBank, 768-d). We take (i) the masked mean of the
              last hidden layer as the embedding and (ii) the softmax of its classification
              head = P(positive), P(negative), P(neutral) as a domain sentiment signal.

All encoders are deterministic and cached to data/interim/cache/<name>.npy (row order =
corpus order), so the expensive CPU pass runs once.
"""
import os
import time
from collections import Counter

import numpy as np

from . import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cache_path(name):
    config.CACHE.mkdir(parents=True, exist_ok=True)
    return config.CACHE / f"{name}.npy"


def cached(name, fn, *args, **kw):
    p = _cache_path(name)
    if p.exists():
        return np.load(p)
    t0 = time.time()
    X = fn(*args, **kw)
    np.save(p, X.astype(np.float32))
    print(f"  [{name}] encoded {X.shape} in {time.time() - t0:.0f}s")
    return X


# ----------------------------------------------------------------------------
# Contextual: sentence-transformers (MiniLM) and GloVe-average
# ----------------------------------------------------------------------------
def encode_sentence_transformer(texts, model_name, batch_size=256, normalize=True):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count() or 4)
    model = SentenceTransformer(model_name, device="cpu")
    if hasattr(model, "max_seq_length") and model.max_seq_length:
        model.max_seq_length = min(model.max_seq_length, 64)
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=normalize, convert_to_numpy=True)


# ----------------------------------------------------------------------------
# Contextual, domain-specific: FinBERT
# ----------------------------------------------------------------------------
def encode_finbert(texts, model_name=config.FINBERT_MODEL, batch_size=128, max_len=64):
    """Returns array [n, 768 + 3]: mean-pooled embedding followed by (pos, neg, neu) probs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [lab["positive"], lab["negative"], lab["neutral"]]
    texts = list(texts)
    out = np.zeros((len(texts), model.config.hidden_size + 3), dtype=np.float32)
    # sort by length so each batch pads to a similar length (2-3x faster on CPU)
    idx = np.argsort([len(t) for t in texts])
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            b = idx[s:s + batch_size]
            enc = tok([texts[i] for i in b], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            prob = torch.softmax(o.logits, -1)[:, order]
            out[b] = torch.cat([emb, prob], 1).numpy()
    return out


# ----------------------------------------------------------------------------
# Static, in-domain: Word2Vec skip-gram with negative sampling (PyTorch)
# ----------------------------------------------------------------------------
def train_word2vec(sentences, dim=config.W2V_DIM, window=4, min_count=5, neg=5, epochs=3,
                   batch_size=8192, lr=0.003, seed=config.RANDOM_STATE):
    """Minimal SGNS (Mikolov et al., 2013). Returns (vocab dict word->idx, vectors, counts)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cnt = Counter(w for s in sentences for w in s.split())
    vocab = [w for w, c in cnt.items() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    freq = np.array([cnt[w] for w in vocab], dtype=np.float64)
    # sub-sampling of frequent words (t = 1e-4) and unigram^0.75 noise distribution
    pf = freq / freq.sum()
    keep_p = np.minimum(1.0, np.sqrt(1e-4 / pf) + 1e-4 / pf)
    noise = torch.tensor(pf ** 0.75 / (pf ** 0.75).sum(), dtype=torch.float)
    pairs = []
    for s in sentences:
        ids = [w2i[w] for w in s.split() if w in w2i]
        ids = [i for i in ids if rng.random() < keep_p[i]]
        for p, c in enumerate(ids):
            for q in range(max(0, p - window), min(len(ids), p + window + 1)):
                if q != p:
                    pairs.append((c, ids[q]))
    pairs = torch.tensor(np.array(pairs, dtype=np.int64))
    V = len(vocab)
    emb_in = torch.nn.Embedding(V, dim)
    emb_out = torch.nn.Embedding(V, dim)
    torch.nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(pairs))
        tot = 0.0
        for s in range(0, len(pairs), batch_size):
            b = pairs[perm[s:s + batch_size]]
            c, o = emb_in(b[:, 0]), emb_out(b[:, 1])
            ng = emb_out(torch.multinomial(noise, len(b) * neg, replacement=True)
                         .view(len(b), neg))
            pos = torch.nn.functional.logsigmoid((c * o).sum(-1))
            negs = torch.nn.functional.logsigmoid(-(ng @ c.unsqueeze(-1)).squeeze(-1)).sum(-1)
            loss = -(pos + negs).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print(f"    w2v epoch {ep + 1}/{epochs}  pairs {len(pairs):,}  loss {tot / len(pairs):.4f}")
    return w2i, emb_in.weight.detach().numpy(), freq


def sif_encode(texts, w2i, vectors, freq, a=1e-3, pc=None):
    p = freq / freq.sum()
    w = a / (a + p)
    X = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for k, t in enumerate(texts):
        ids = [w2i[x] for x in t.split() if x in w2i]
        if ids:
            X[k] = (vectors[ids] * w[ids, None]).sum(0) / len(ids)
    if pc is not None:
        X = X - (X @ pc[:, None]) * pc[None, :]
    return X


def w2v_headline_vectors(corpus_texts, train_texts):
    """Train on TRAIN headlines, remove the first principal component (fit on TRAIN)."""
    w2i, vec, freq = train_word2vec(train_texts)
    Xtr = sif_encode(train_texts[:200000], w2i, vec, freq)
    _, _, vt = np.linalg.svd(Xtr - 0, full_matrices=False)
    pc = vt[0]
    X = sif_encode(corpus_texts, w2i, vec, freq, pc=pc)
    np.save(_cache_path("w2v_vocab_vectors"), vec)
    return X, (w2i, vec, freq, pc)
