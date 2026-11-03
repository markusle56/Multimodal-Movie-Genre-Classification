"""Step 6 - comparison, significance, overfitting, robustness, bias and failure analysis.

Reads outputs/results/{runs.json, probs_*.npz}; writes every analysis number to
outputs/results/analysis.json.
Headline metric: macro-F1 (proposal).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion.config import FEAT, GENRES, N_BOOTSTRAP, RESULTS, SPLITS  # noqa: E402
from mmfusion.metrics import paired_bootstrap_f1, per_class_ap, predict  # noqa: E402

RUNS = json.load(open(RESULTS / "runs.json"))
CH = RUNS.pop("_choices")
meta = pd.read_csv(SPLITS / "meta.csv")
Y = meta[GENRES].values.astype(np.int8)
te = np.flatnonzero(meta.split.values == 2)
Yte = Y[te]
texts = json.load(open(SPLITS / "texts_clean.json"))
A = {"choices": CH}

LABEL = {"img_clip": "Image only: CLIP", "txt_distil": "Text only: DistilBERT",
         "late_avg": "Late fusion (average)", "late_weighted": "Late fusion (per-genre weights)",
         "early_fusion": "Early fusion", "gmu": "GMU fusion", "xattn": "Cross-attention fusion",
         "tfidf_lr": "TF-IDF + LR (text)", "img_resnet50": "ResNet-50 (image)", "img_vit": "ImageNet ViT-B/16 (image)",
         "txt_bert": "BERT-base 512 (text)", "txt_clip": "CLIP text 77 (text)",
         "early_no_moddrop": "Early fusion − modality dropout"}
FAMILY = {"img_clip": "proposal baseline", "txt_distil": "proposal baseline", "late_avg": "proposal fusion",
          "late_weighted": "proposal fusion", "early_fusion": "proposal fusion", "gmu": "extension", "xattn": "extension",
          "tfidf_lr": "encoder comparison", "img_resnet50": "encoder comparison", "img_vit": "encoder comparison",
          "txt_bert": "encoder comparison", "txt_clip": "encoder comparison", "early_no_moddrop": "ablation"}
ORDER = [k for k in LABEL if k in RUNS]
probs = {k: np.load(RESULTS / f"probs_{k}.npz") for k in ORDER}      # seed-0 predictions
MAINF = ["late_avg", "late_weighted", "early_fusion", "gmu", "xattn"]
BASE = ["img_clip", "txt_distil"]


def mean(k, split="test"):
    r = RUNS[k]
    return r.get(f"{split}_mean", r[split])


def sd(k, split="test"):
    return RUNS[k].get(f"{split}_sd", {})


best = max([m for m in MAINF if m in RUNS], key=lambda k: (mean(k, "val")["macro_F1"], mean(k, "val")["mAP"]))
best_base = max(BASE, key=lambda k: mean(k, "val")["macro_F1"])
A["best_model"], A["best_baseline"] = best, best_base
print("best fusion (val macro-F1):", best, "| best baseline:", best_base)

# ------------------------------------------------------------------ 1. main table + significance
rows = []
for k in ORDER:
    m, s = mean(k), sd(k)
    rows.append({"model": k, "label": LABEL[k], "family": FAMILY[k], "n_seeds": RUNS[k].get("n_seeds", 1), **m,
                 **{f"{x}_sd": s.get(x, 0.0) for x in m}, "val_macro_F1": mean(k, "val")["macro_F1"],
                 "val_mAP": mean(k, "val")["mAP"], "params": RUNS[k].get("params"),
                 "train_seconds": RUNS[k].get("train_seconds"), "infer_ms_per_1k": RUNS[k].get("infer_ms_per_1k"),
                 "best_epoch": RUNS[k].get("best_epoch")})
pd.DataFrame(rows).to_csv(RESULTS / "results_table.csv", index=False)
A["table"] = rows


def pb(a, b):
    d, ci, p = paired_bootstrap_f1(Yte, probs[a]["test"], probs[a]["th"], probs[b]["test"], probs[b]["th"], n=N_BOOTSTRAP)
    return {"diff": d, "ci": ci, "p": p}


SIG = {}
for f in [m for m in MAINF if m in RUNS]:
    for b in BASE:
        SIG[f"{f}>{b}"] = pb(f, b)
SIG["early_fusion>late_weighted"] = pb("early_fusion", "late_weighted")
SIG["late_weighted>late_avg"] = pb("late_weighted", "late_avg")
for m in {"gmu", best} - {"early_fusion"}:
    SIG[f"{m}>early_fusion"] = pb(m, "early_fusion")
SIG["txt_distil>tfidf_lr"] = pb("txt_distil", "tfidf_lr")
SIG["txt_bert>txt_distil"] = pb("txt_bert", "txt_distil")
SIG["img_clip>img_vit"] = pb("img_clip", "img_vit")
A["significance_macroF1"] = SIG
print({k: round(v["p"], 3) for k, v in SIG.items()})

# ------------------------------------------------------------------ 2. per-genre F1 (proposal objective 2)
show = [s for s in ["img_clip", "txt_distil", "late_avg", "late_weighted", "early_fusion", "gmu", "xattn", "tfidf_lr"] if s in RUNS]
prev = Yte.mean(0)
order = np.argsort(-prev)
F1g = pd.DataFrame({LABEL[k]: [RUNS[k]["test_F1_mean"][g] for g in GENRES] for k in show}, index=GENRES).iloc[order]
APg = pd.DataFrame({LABEL[k]: per_class_ap(Yte, probs[k]["test"]) for k in show}, index=GENRES).iloc[order]
A["per_genre_F1"] = F1g.round(4).to_dict()
A["per_genre_AP"] = APg.round(4).to_dict()
gain = F1g[LABEL[best]] - F1g[[LABEL["img_clip"], LABEL["txt_distil"]]].max(axis=1)
A["fusion_gain_F1_vs_best_single_per_genre"] = gain.round(4).to_dict()
A["poster_beats_plot_F1"] = list(F1g.index[F1g[LABEL["img_clip"]] > F1g[LABEL["txt_distil"]]])

# ------------------------------------------------------------------ 3. overfitting: train vs val macro-F1
OV = {}
for k in ORDER:
    s = RUNS[k].get("macro_F1_by_split_mean") or RUNS[k].get("macro_F1_by_split")
    if s and s.get("train") is not None:
        OV[k] = {"train": s["train"], "val": s["val"], "test": s["test"], "gap_train_val": s["train"] - s["val"],
                 "gap_val_test": s["val"] - s["test"], "best_epoch": RUNS[k].get("best_epoch"),
                 "epochs_run": len(RUNS[k]["histories"][0]) if RUNS[k].get("histories") else None}
A["overfitting_macroF1"] = OV

# ------------------------------------------------------------------ 4. missing modality
A["missing_modality"] = {
    LABEL[k]: {"both": mean(k)["macro_F1"], "image missing (text only)": RUNS[k]["test_macroF1_missing_img_mean"],
               "text missing (image only)": RUNS[k]["test_macroF1_missing_txt_mean"], "mAP both": mean(k)["mAP"],
               "mAP image missing": RUNS[k]["test_mAP_missing_img_mean"], "mAP text missing": RUNS[k]["test_mAP_missing_txt_mean"]}
    for k in ["late_avg", "late_weighted", "early_fusion", "gmu", "xattn", "early_no_moddrop"]
    if k in RUNS and "test_macroF1_missing_img_mean" in RUNS[k]}

# ------------------------------------------------------------------ 5. modality reliance + efficiency
A["late_weights"] = pd.Series(RUNS["late_weighted"]["image_weight_per_genre"]).iloc[order].round(2).to_dict()
gf = RESULTS / "gmu_gate_gmu.npy"
if gf.exists():
    gate = np.load(gf)
    A["gmu_gate_per_genre"] = {GENRES[k]: round(float(gate[Yte[:, k] == 1].mean()), 4) for k in order}
A["encoder_timing"] = json.load(open(FEAT / "timing.json"))

# ------------------------------------------------------------------ 6. confusions + failure cases (best model)
Pb, thb = probs[best]["test"], probs[best]["th"]
predb = predict(Pb, thb)
conf = []
for a in range(23):
    for b in range(23):
        neg = (Yte[:, a] == 1) & (Yte[:, b] == 0)
        if a != b and neg.sum() > 20:
            conf.append((GENRES[a], GENRES[b], float(predb[neg, b].mean())))
A["top_confusions"] = sorted(conf, key=lambda x: -x[2])[:10]
mt = meta.iloc[te].reset_index(drop=True)
sf1 = np.array([f1_score(Yte[i], predb[i], zero_division=0) for i in range(len(te))])
worst = np.argsort(-(np.abs(Pb - Yte) * (predb != Yte)).sum(1))[:10]
A["failure_cases"] = [{"idx": int(te[j]), "true": [GENRES[k] for k in np.flatnonzero(Yte[j])],
                       "pred": [GENRES[k] for k in np.flatnonzero(predb[j])], "n_words": int(mt.n_words[j]),
                       "plot": texts[te[j]][:160]} for j in worst]
A["failure_stats"] = {"share_samples_F1_zero": float((sf1 == 0).mean()),
                      "mean_words_F1_zero": float(mt.n_words[sf1 == 0].mean()), "mean_words_all": float(mt.n_words.mean()),
                      "mean_true_labels_worst10": float(Yte[worst].sum(1).mean()), "mean_true_labels_all": float(Yte.sum(1).mean())}

# ------------------------------------------------------------------ 7. bias / subgroups
trm = meta.split.values == 0
gshare = np.array([meta.grayscale.values[trm & (Y[:, k] == 1)].mean() for k in range(len(GENRES))])
old = list(np.argsort(-gshare)[:5])
A["grayscale_prone_genres"] = {GENRES[k]: float(gshare[k]) for k in old}
groups = {"colour poster": ~mt.grayscale.values, "grayscale poster": mt.grayscale.values}
q = pd.qcut(mt.n_words, 4, labels=["plot Q1 (shortest)", "plot Q2", "plot Q3", "plot Q4 (longest)"])
for lab in q.cat.categories:
    groups[str(lab)] = (q == lab).values
groups["fragment (<10 words)"] = mt.fragment.values
groups["1 genre"] = Yte.sum(1) == 1
groups["4+ genres"] = Yte.sum(1) >= 4
SG = []
for g, msk in groups.items():
    for k in [best, "txt_distil", "img_clip"]:
        pr = predict(probs[k]["test"][msk], probs[k]["th"])
        SG.append({"group": g, "model": LABEL[k], "n": int(msk.sum()),
                   "samples_F1": round(float(f1_score(Yte[msk], pr, average="samples", zero_division=0)), 4),
                   "micro_F1": round(float(f1_score(Yte[msk], pr, average="micro", zero_division=0)), 4)})
A["subgroups"] = SG
rho, p = spearmanr(prev, [RUNS[best]["test_F1_mean"][g] for g in GENRES])
A["spearman_prevalence_F1"] = {"rho": float(rho), "p": float(p)}
gs = mt.grayscale.values
pri = predict(probs["img_clip"]["test"], probs["img_clip"]["th"])
A["grayscale_shortcut"] = {
    "image_pred_rate_grayscale": float(pri[gs][:, old].mean()), "image_pred_rate_colour": float(pri[~gs][:, old].mean()),
    "true_rate_grayscale": float(Yte[gs][:, old].mean()), "true_rate_colour": float(Yte[~gs][:, old].mean()),
    "fusion_FP_grayscale": float(((predb[gs][:, old] == 1) & (Yte[gs][:, old] == 0)).mean()),
    "fusion_FP_colour": float(((predb[~gs][:, old] == 1) & (Yte[~gs][:, old] == 0)).mean()),
    "image_FP_grayscale": float(((pri[gs][:, old] == 1) & (Yte[gs][:, old] == 0)).mean()),
    "image_FP_colour": float(((pri[~gs][:, old] == 1) & (Yte[~gs][:, old] == 0)).mean())}

# ------------------------------------------------------------------ 8. stratification vs random splits of the same sizes
sp = meta.split.values
prev_by_split = np.stack([Y[sp == s].mean(0) for s in (0, 1, 2)])
rng = np.random.default_rng(0)
kept, sizes, fn = np.flatnonzero(sp >= 0), [int((sp == s).sum()) for s in (0, 1, 2)], GENRES.index("Film-Noir")
rand_val_fn, rand_gap = [], []
for _ in range(200):
    p = rng.permutation(kept)
    parts = np.split(p, np.cumsum(sizes)[:2])
    rand_val_fn.append(int(Y[parts[1], fn].sum()))
    rand_gap.append(float(np.ptp(np.stack([Y[x].mean(0) for x in parts]), 0).max()))
A["stratification"] = {
    "sizes": dict(zip(("train", "val", "test"), sizes)),
    "max_genre_share_gap": float(np.ptp(prev_by_split, 0).max()),
    "film_noir_share": prev_by_split[:, fn].round(4).tolist(), "film_noir_val_count": int(Y[sp == 1, fn].sum()),
    "random_200": {"film_noir_val_count_p5_p95": [int(np.percentile(rand_val_fn, 5)), int(np.percentile(rand_val_fn, 95))],
                   "max_genre_share_gap_median": float(np.median(rand_gap))}}

json.dump(A, open(RESULTS / "analysis.json", "w"), indent=1, default=float)
print("analysis written:", RESULTS / "analysis.json")
