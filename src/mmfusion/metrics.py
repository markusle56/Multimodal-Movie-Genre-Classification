"""Multi-label evaluation: ranking metrics (threshold-free) and decision metrics (thresholded).

Threshold-free : macro mAP (headline; insensitive to the decision threshold), micro AP, macro ROC-AUC
Thresholded    : micro / macro / samples / weighted F1, Hamming loss, subset (exact-match) accuracy
Thresholds are tuned per genre on the validation split (maximising per-genre F1) and then frozen
for test; a global 0.5 threshold is reported alongside for comparison.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss, precision_recall_curve,
                             roc_auc_score)


def per_class_ap(Y, P):
    return np.array([average_precision_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])])


def tune_thresholds(Y, P):
    th = np.full(Y.shape[1], 0.5)
    for k in range(Y.shape[1]):
        p, r, t = precision_recall_curve(Y[:, k], P[:, k])
        f = 2 * p * r / np.clip(p + r, 1e-9, None)
        th[k] = t[np.argmax(f[:-1])] if len(t) else 0.5
    return th


def predict(P, th):
    pred = (P >= th).astype(int)
    empty = pred.sum(1) == 0                         # every movie has >= 1 genre: force the top-scoring one
    pred[empty, P[empty].argmax(1)] = 1
    return pred


def all_metrics(Y, P, th):
    pred = predict(P, th)
    return {
        "mAP": float(per_class_ap(Y, P).mean()),
        "micro_AP": float(average_precision_score(Y, P, average="micro")),
        "macro_AUC": float(roc_auc_score(Y, P, average="macro")),
        "micro_F1": float(f1_score(Y, pred, average="micro")),
        "macro_F1": float(f1_score(Y, pred, average="macro", zero_division=0)),
        "samples_F1": float(f1_score(Y, pred, average="samples", zero_division=0)),
        "weighted_F1": float(f1_score(Y, pred, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(Y, pred)),
        "subset_acc": float((pred == Y).all(1).mean()),
        "macro_F1@0.5": float(f1_score(Y, predict(P, np.full(Y.shape[1], .5)), average="macro", zero_division=0)),
    }






def paired_bootstrap_f1(Y, PA, thA, PB, thB, n=1000, seed=0, average="macro"):
    """One-sided paired bootstrap for F1 with each model's own (frozen, val-tuned) thresholds.

    Returns (mean difference A-B, 95% CI of the difference, p = P(F1_A <= F1_B))."""
    rng = np.random.default_rng(seed)
    dA, dB = predict(PA, thA), predict(PB, thB)
    diffs = []
    for _ in range(n):
        s = rng.integers(0, len(Y), len(Y))
        diffs.append(f1_score(Y[s], dA[s], average=average, zero_division=0)
                     - f1_score(Y[s], dB[s], average=average, zero_division=0))
    diffs = np.array(diffs)
    return float(diffs.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), float((diffs <= 0).mean())
