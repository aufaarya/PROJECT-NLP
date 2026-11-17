"""
models.py
---------
Baseline (exchange-rate history only) and combined (history + NLP features) models.

Regression track  (target: r_{t+1}, next trading day's log return)
    * RandomWalk   : predicts 0 (the classic FX benchmark: tomorrow = today)
    * ARIMA(p,0,q) : fit on returns, order chosen by AIC on TRAIN only        [BASELINE]
    * ARIMAX       : same ARIMA order + lagged news features as exogenous      [COMBINED]

Classification track (target: 1 if r_{t+1} > 0 else 0)
    * Majority     : always predicts the TRAIN majority class
    * Persistence  : predicts tomorrow's direction = today's direction
    * ARIMA-sign   : sign of the ARIMA return forecast                         [BASELINE]
    * LogReg / GBM on market features only                                    [BASELINE]
    * LogReg / GBM on market + news features                                  [COMBINED]

Walk-forward protocol: parameters are estimated on TRAIN (and TRAIN+VAL for the final
test run); every forecast for day t uses only information available at the end of day t.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from . import config

warnings.filterwarnings("ignore")  # statsmodels convergence chatter


# ----------------------------------------------------------------------------
# ARIMA / ARIMAX
# ----------------------------------------------------------------------------
def _arima_design(data: pd.DataFrame, exog_cols=None):
    """Align series so that endog[j] = r_{j+1} = target of row j and exog[j] = features of row j.
    Then the one-step prediction for endog[j] is exactly the forecast for row j's target,
    using information up to row j only."""
    endog = data["target_ret_next"].to_numpy()
    exog = data[exog_cols].to_numpy() if exog_cols else None
    return endog, exog


def select_arima_order(train: pd.DataFrame, max_p=3, max_q=3):
    """Grid-search (p, q) on TRAIN by AIC. d = 0 because log returns are stationary
    (verified with an ADF test in the notebook)."""
    endog, _ = _arima_design(train)
    best, table = None, []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            res = ARIMA(endog, order=(p, 0, q), trend="c").fit()
            table.append({"p": p, "q": q, "aic": res.aic})
            if best is None or res.aic < best[1]:
                best = ((p, 0, q), res.aic)
        except Exception:
            continue
    return best[0], pd.DataFrame(table).sort_values("aic").reset_index(drop=True)


def arima_walk_forward(data: pd.DataFrame, fit_mask: np.ndarray, order, exog_cols=None):
    """Fit ARIMA(X) on the rows in `fit_mask`, then produce one-step-ahead forecasts for
    EVERY row of `data` with parameters frozen (statsmodels `apply` re-runs the Kalman
    filter over new data without re-estimating). Forecast for row j uses rows <= j only
    — in the aligned design, the one-step prediction for endog[j] conditions on
    endog[:j] = targets of rows < j, which are all realised by the end of day j.
    """
    endog, exog = _arima_design(data, exog_cols)
    scaler = None
    if exog is not None:
        scaler = StandardScaler().fit(exog[fit_mask])     # scale with fit-period stats only
        exog = scaler.transform(exog)
    model = ARIMA(endog[fit_mask], exog=None if exog is None else exog[fit_mask],
                  order=order, trend="c")
    res = model.fit()
    res_all = res.apply(endog, exog=exog)
    pred = res_all.predict()        # one-step-ahead, in-sample (non-dynamic)
    return np.asarray(pred), res


# ----------------------------------------------------------------------------
# Classifiers with validation-based hyper-parameter search
# ----------------------------------------------------------------------------
def _logreg(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def _gbm(depth, lr, n_iter):
    return HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr,
                                          max_iter=n_iter, l2_regularization=1.0,
                                          min_samples_leaf=40,
                                          random_state=config.RANDOM_STATE)


MODEL_GRIDS = {
    "LogReg": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)],
    "GBM": [{"depth": d, "lr": lr, "n_iter": n}
            for d in (2, 3) for lr in (0.03, 0.1) for n in (100, 300)],
}
BUILDERS = {"LogReg": lambda p: _logreg(**p), "GBM": lambda p: _gbm(**p)}


def tune_and_fit_classifier(name, feature_cols, train, val, test):
    """1) pick hyper-parameters by macro-F1 on VAL (model fit on TRAIN);
       2) report VAL predictions from that TRAIN-only model;
       3) refit on TRAIN+VAL with the chosen params and predict TEST once."""
    Xtr, ytr = train[feature_cols].to_numpy(), train["target_up_next"].to_numpy()
    Xva, yva = val[feature_cols].to_numpy(), val["target_up_next"].to_numpy()
    Xte = test[feature_cols].to_numpy()

    best = None
    for params in MODEL_GRIDS[name]:
        m = BUILDERS[name](params).fit(Xtr, ytr)
        score = f1_score(yva, m.predict(Xva), average="macro")
        if best is None or score > best[0]:
            best = (score, params, m)
    _, params, m_train = best

    val_proba = m_train.predict_proba(Xva)[:, 1]
    m_final = BUILDERS[name](params).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    test_proba = m_final.predict_proba(Xte)[:, 1]
    return {"params": params, "val_proba": val_proba, "test_proba": test_proba,
            "model": m_final}
