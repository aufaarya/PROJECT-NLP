"""
fusion_models.py  -  multimodal fusion of X_text and X_time (Task 3 req. 2 + 3a)
-----------------------------------------------------------------------------
Three fusion strategies, all tuned on VALIDATION and scored once on TEST (after refitting
on TRAIN+VAL), exactly the protocol of Task 2:

1. EARLY FUSION      [X_time | X_text] concatenated -> Logistic Regression or LightGBM.
2. LATE FUSION       separate unimodal models  p_time = f(X_time),  p_text = g(X_text);
                     final probability  p = w * p_time + (1 - w) * p_text,
                     w chosen on validation (w = 1 means "ignore text").
3. ATTENTION FUSION  (neural, intermediate fusion) - see AttnFusionNet below:
                     the raw stream of the day's K headline embeddings is pooled with a
                     learned attention layer (which headline matters?), concatenated with an
                     MLP encoding of the market features, and fed to a logistic head.
                     The attention weights are exported for interpretability.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# ----------------------------------------------------------------------------
# Tabular models + validation grid search
# ----------------------------------------------------------------------------
def _logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def _lgbm(leaves, lr, n):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(num_leaves=leaves, learning_rate=lr, n_estimators=n,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
                          random_state=config.RANDOM_STATE, n_jobs=4)


GRIDS = {
    "LogReg": [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
    "LGBM": [{"leaves": l, "lr": lr, "n": n} for l in (4, 8) for lr in (0.02, 0.05)
             for n in (100, 300)],
}
BUILD = {"LogReg": lambda p: _logreg(**p), "LGBM": lambda p: _lgbm(**p)}


def _score(y, proba):
    return f1_score(y, (proba >= 0.5).astype(int), average="macro")


def tune_fit(name, cols, train, val, test, target="target_up_next"):
    """Grid on VAL (fit on TRAIN) -> refit best on TRAIN+VAL -> predict TEST once."""
    Xtr, ytr = train[cols].to_numpy(float), train[target].to_numpy()
    Xva, yva = val[cols].to_numpy(float), val[target].to_numpy()
    Xte = test[cols].to_numpy(float)
    best = None
    for p in GRIDS[name]:
        m = BUILD[name](p).fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:, 1]
        s = _score(yva, pv)
        if best is None or s > best[0] + 1e-12:
            best = (s, p, pv)
    s, p, pv = best
    m_final = BUILD[name](p).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    return {"params": p, "val_score": s, "val_proba": pv,
            "test_proba": m_final.predict_proba(Xte)[:, 1], "model": m_final}


def late_fusion(p_time_val, p_text_val, y_val, p_time_test, p_text_test):
    best = None
    for w in np.linspace(0, 1, 11):
        s = _score(y_val, w * p_time_val + (1 - w) * p_text_val)
        if best is None or s > best[0] + 1e-12:
            best = (s, w)
    s, w = best
    return {"w_time": float(w), "val_score": s,
            "val_proba": w * p_time_val + (1 - w) * p_text_val,
            "test_proba": w * p_time_test + (1 - w) * p_text_test}


# ----------------------------------------------------------------------------
# Neural attention fusion
# ----------------------------------------------------------------------------
def headline_tensor(corpus, X, trading_dates, k=config.TOP_K_PER_DAY):
    """[n_days, k, d] headline embeddings (salience order) + mask [n_days, k] + row index."""
    td = pd.DatetimeIndex(trading_dates)
    pos = td.get_indexer(pd.DatetimeIndex(corpus["date"]))
    rank = corpus["rank_in_day"].to_numpy()
    ok = (pos >= 0) & (rank < k)
    H = np.zeros((len(td), k, X.shape[1]), dtype=np.float32)
    mask = np.zeros((len(td), k), dtype=bool)
    rid = -np.ones((len(td), k), dtype=np.int64)
    H[pos[ok], rank[ok]] = X[ok]
    mask[pos[ok], rank[ok]] = True
    rid[pos[ok], rank[ok]] = np.where(ok)[0]
    return H, mask, rid


def _net(d_text, d_time, h_text=32, h_time=16, p_drop=0.3):
    import torch
    import torch.nn as nn

    class AttnFusionNet(nn.Module):
        """text stream  [B,K,d] -> Linear(d,h) -> tanh -> attention over K -> [B,h]
           time vector  [B,m]   -> Linear(m,h') -> ReLU                    -> [B,h']
           concat -> dropout -> Linear(h+h', 1) -> P(UP)"""
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_text, h_text), nn.Tanh())
            self.att = nn.Linear(h_text, 1)
            self.time = nn.Sequential(nn.Linear(d_time, h_time), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(h_text + h_time, 1))

        def forward(self, H, mask, Xt):
            z = self.proj(H)                                    # [B,K,h]
            a = self.att(z).squeeze(-1).masked_fill(~mask, -1e9)
            a = torch.softmax(a, -1) * mask.any(-1, keepdim=True)   # empty day -> zeros
            text = (a.unsqueeze(-1) * z).sum(1)
            return self.head(torch.cat([text, self.time(Xt)], -1)).squeeze(-1), a

    return AttnFusionNet()


def _train_net(H, M, Xt, y, epochs, seed, lr=1e-3, wd=1e-2, H_va=None, M_va=None,
               Xt_va=None, y_va=None):
    import torch
    torch.manual_seed(seed)
    net = _net(H.shape[-1], Xt.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    H, M, Xt, y = T(H), T(M, torch.bool), T(Xt), T(y)
    hist = []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=g)
        for s in range(0, len(y), 64):
            b = perm[s:s + 64]
            logit, _ = net(H[b], M[b], Xt[b])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if H_va is not None:
            p = _predict_net(net, H_va, M_va, Xt_va)[0]
            hist.append(_score(y_va, p))
    return net, hist


def _predict_net(net, H, M, Xt):
    import torch
    net.eval()
    with torch.no_grad():
        logit, a = net(torch.tensor(H), torch.tensor(M), torch.tensor(Xt, dtype=torch.float32))
    return torch.sigmoid(logit).numpy(), a.numpy()


def attention_fusion(H, M, Xtime, y, idx_tr, idx_va, idx_te, seeds=(0, 1, 2, 3, 4),
                     max_epochs=40):
    """Choose the number of epochs on VAL (mean curve over seeds), then refit on TRAIN+VAL
    with that many epochs; seed-ensemble the probabilities (reduces variance)."""
    sc = StandardScaler().fit(Xtime[idx_tr])
    Xs = sc.transform(Xtime).astype(np.float32)
    curves, val_probs = [], []
    for s in seeds:
        net, hist = _train_net(H[idx_tr], M[idx_tr], Xs[idx_tr], y[idx_tr], max_epochs, s,
                               H_va=H[idx_va], M_va=M[idx_va], Xt_va=Xs[idx_va], y_va=y[idx_va])
        curves.append(hist)
    curve = np.mean(curves, 0)
    # smooth the validation curve (3-epoch moving average) before taking the arg-max
    sm = pd.Series(curve).rolling(3, min_periods=1, center=True).mean().to_numpy()
    best_ep = int(np.argmax(sm)) + 1
    for s in seeds:
        net, _ = _train_net(H[idx_tr], M[idx_tr], Xs[idx_tr], y[idx_tr], best_ep, s)
        val_probs.append(_predict_net(net, H[idx_va], M[idx_va], Xs[idx_va])[0])
    # final: refit on TRAIN+VAL with a scaler fit on TRAIN+VAL
    idx_fit = np.concatenate([idx_tr, idx_va])
    sc2 = StandardScaler().fit(Xtime[idx_fit])
    Xs2 = sc2.transform(Xtime).astype(np.float32)
    te_probs, te_att = [], []
    for s in seeds:
        net, _ = _train_net(H[idx_fit], M[idx_fit], Xs2[idx_fit], y[idx_fit], best_ep, s)
        p, a = _predict_net(net, H[idx_te], M[idx_te], Xs2[idx_te])
        te_probs.append(p)
        te_att.append(a)
    return {"epochs": best_ep, "val_curve": curve.tolist(),
            "val_proba": np.mean(val_probs, 0), "test_proba": np.mean(te_probs, 0),
            "test_attention": np.mean(te_att, 0)}
