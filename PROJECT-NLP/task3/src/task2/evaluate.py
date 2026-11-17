"""
evaluate.py
-----------
Metrics for both task formulations.

Regression (next-day log return)
    RMSE, MAE (reported in basis points, 1 bp = 0.01%) and Directional Accuracy
    (share of days where sign(forecast) == sign(actual)).
Classification (next-day UP / DOWN)
    Accuracy, macro-F1 (robust to mild class imbalance), MCC (single number that is 0 for
    any constant or random predictor), ROC-AUC (ranking quality of the probabilities),
    and a one-sided binomial test of accuracy > 50% (is the edge distinguishable from luck?).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, roc_auc_score)

BP = 1e4  # log-return -> basis points


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nz = y_pred != 0
    da = np.mean(np.sign(y_pred[nz]) == np.sign(y_true[nz])) if nz.any() else np.nan
    return {
        "RMSE_bp": np.sqrt(mean_squared_error(y_true, y_pred)) * BP,
        "MAE_bp": mean_absolute_error(y_true, y_pred) * BP,
        "DirAcc": da,
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    n, k = len(y_true), int((y_true == y_pred).sum())
    out = {
        "Accuracy": k / n,
        "MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "p_binom(acc>0.5)": binomtest(k, n, 0.5, alternative="greater").pvalue,
        "n": n,
    }
    return out


def results_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index(["track", "model", "features"])
