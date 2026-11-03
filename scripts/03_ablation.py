"""Step 3 - preprocessing ablation: choose text cleaning and poster preprocessing on VALIDATION.

Compared on validation macro-F1 (the proposal's primary metric), with the decision rule of 04_tuning.py:
  * 5 text-cleaning variants x {TF-IDF + logistic regression, DistilBERT [CLS] MLP, DistilBERT mean MLP}
  * CLIP poster preprocessing: proposal centre crop vs letterbox (the full four-way study is 04_tuning.py part A)
Test scores are written for transparency but are NOT used for any choice.

Outputs: outputs/results/ablation.json, outputs/splits/choices.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion import models as M  # noqa: E402
from mmfusion.config import FEAT, GENRES, RESULTS, SEEDS, SPLITS, TEXT_VARIANTS  # noqa: E402
from mmfusion.metrics import all_metrics, tune_thresholds  # noqa: E402
from mmfusion.train import Feats, device, fit, predict_proba  # noqa: E402

meta = pd.read_csv(SPLITS / "meta.csv")
Y = meta[GENRES].values.astype(np.int8)
tr, va, te = (np.flatnonzero(meta.split.values == s) for s in (0, 1, 2))
dev = device()
OUT = {"text_tfidf": {}, "text_distil": {}, "image_clip": {}}
KEYS = ("macro_F1", "micro_F1", "samples_F1", "mAP")


def scores(Pva, Pte):
    th = tune_thresholds(Y[va], Pva)
    v, t = all_metrics(Y[va], Pva, th), all_metrics(Y[te], Pte, th)
    return {"val": {k: v[k] for k in KEYS}, "test": {k: t[k] for k in KEYS}}


def mlp(key, X, d):
    feats = Feats({key: X}, Y)
    rs = []
    for sd in SEEDS:
        m, info = fit(M.Unimodal(d, key), feats, tr, va, seed=sd)
        rs.append(scores(predict_proba(m, feats, va, dev), predict_proba(m, feats, te, dev)))
    return {sp: {k: float(np.mean([r[sp][k] for r in rs])) for k in KEYS} |
            {f"{k}_sd": float(np.std([r[sp][k] for r in rs])) for k in KEYS} for sp in ("val", "test")}


# ------------------------------------------------------------------ TF-IDF per variant
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402

for v in TEXT_VARIANTS:
    texts = np.array(json.load(open(SPLITS / "texts" / f"{v}.json")), dtype=object)
    t0 = time.time()
    # token pattern keeps 1-character tokens so the number mask "#" / placeholder "N" is not silently dropped
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True,
                          token_pattern=r"[\w#][\w#'-]*")
    Xtr = vec.fit_transform(texts[tr])
    clf = OneVsRestClassifier(LogisticRegression(C=4, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(Xtr, Y[tr])
    OUT["text_tfidf"][v] = scores(clf.predict_proba(vec.transform(texts[va])), clf.predict_proba(vec.transform(texts[te])))
    OUT["text_tfidf"][v]["vocab"] = len(vec.vocabulary_)
    print(f"tfidf {v:24s} val macroF1 {OUT['text_tfidf'][v]['val']['macro_F1']:.4f} mAP {OUT['text_tfidf'][v]['val']['mAP']:.4f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ DistilBERT per variant x pooling
for v in TEXT_VARIANTS:
    for pool in ("cls", "mean"):
        f = FEAT / f"txt_distil_{v}_{pool}.npy"
        if not f.exists():
            print("missing", f.name)
            continue
        OUT["text_distil"][f"{v}|{pool}"] = mlp("txt", np.load(f), 768)
        r = OUT["text_distil"][f"{v}|{pool}"]["val"]
        print(f"distil {v:24s} {pool:4s} val macroF1 {r['macro_F1']:.4f} ± {r['macro_F1_sd']:.4f} mAP {r['mAP']:.4f}", flush=True)

# ------------------------------------------------------------------ CLIP image preprocessing
for prep in ("crop", "lbox"):
    X = np.load(FEAT / f"clip_img_{prep}.npy").astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    OUT["image_clip"][prep] = mlp("img", X.astype(np.float16), 512)
    r = OUT["image_clip"][prep]["val"]
    print(f"clip image {prep}: val macroF1 {r['macro_F1']:.4f} ± {r['macro_F1_sd']:.4f} mAP {r['mAP']:.4f}", flush=True)


def pick(d, ref):
    """Same rule as 04_tuning.py: keep the reference unless the best alternative beats it on validation macro-F1
    by more than max(0.005, 2 x seed s.d.); single-run scores (TF-IDF) have s.d. 0."""
    sd = lambda k: d[k]["val"].get("macro_F1_sd", 0.0)
    alt = max((k for k in d if k != ref), key=lambda k: d[k]["val"]["macro_F1"])
    margin = max(0.005, 2 * max(sd(alt), sd(ref)))
    return alt if d[alt]["val"]["macro_F1"] - d[ref]["val"]["macro_F1"] > margin else ref


# references: the planned cleaning (detokenise + mask numbers, mean pooling) and the proposal's centre crop
best_distil = pick(OUT["text_distil"], "detok_num|mean")
choices = {"tfidf_variant": pick(OUT["text_tfidf"], "raw"), "distil_variant": best_distil.split("|")[0],
           "distil_pool": best_distil.split("|")[1], "clip_img_prep": pick(OUT["image_clip"], "crop"),
           "rule": "keep the reference unless validation macro-F1 gain > max(0.005, 2 x seed s.d.)"}
json.dump(OUT, open(RESULTS / "ablation.json", "w"), indent=1)
json.dump(choices, open(SPLITS / "choices.json", "w"), indent=1)
print(json.dumps(choices, indent=1))
