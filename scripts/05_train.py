"""Step 5 - train and evaluate every model on the fixed 60/10/30 split.

Proposal models (Group 18, Project 2.1) - encoders: CLIP ViT-B/32 (poster), DistilBERT (plot)
  img_clip            poster MLP 512 -> 256 -> 23                             image-only baseline
  txt_distil          text MLP 768 -> 256 -> 23                               text-only baseline
  late_avg            average of the two baseline probability vectors        late fusion (proposal Appendix D.2)
  late_weighted       per-genre mixing weight tuned on validation            late fusion, optional optimisation
  early_fusion        LayerNorm + concat + joint MLP, modality dropout       early fusion (proposal Appendix D.1)
Extensions beyond the proposal
  gmu                 gated multimodal unit (Arevalo et al., 2017)
  xattn               token-level co-attention over CLIP patches and DistilBERT chunks
Encoder comparisons (feature-extraction evaluation)
  tfidf_lr            TF-IDF 1-2 grams + logistic regression (classical text baseline)
  img_resnet50, img_vit          ImageNet CNN / ViT poster MLPs
  txt_bert            BERT-base, 512 tokens (longer context, larger model)
  txt_clip            CLIP text tower (77 tokens)
Ablation
  early_no_moddrop    early fusion without modality dropout (missing-modality robustness)

Every neural model is trained with 3 seeds. Thresholds are tuned per genre on validation.
Outputs: outputs/results/{runs.json, probs_<model>.npz}, outputs/models/<model>_s<seed>.pt
Usage: python scripts/05_train.py [--only gmu xattn ...] [--seeds 0 1 2] [--verbose]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion import models as M  # noqa: E402
from mmfusion.config import FEAT, GENRES, MODELS, RESULTS, SEEDS, SPLITS, TRAIN  # noqa: E402
from mmfusion.metrics import all_metrics, per_class_ap, predict, tune_thresholds  # noqa: E402
from mmfusion.train import Feats, device, fit, predict_proba  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*")
ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

meta = pd.read_csv(SPLITS / "meta.csv")
Y = meta[GENRES].values.astype(np.int8)
tr, va, te = (np.flatnonzero(meta.split.values == s) for s in (0, 1, 2))
CH = json.load(open(SPLITS / "choices.json"))
TV, POOL, PREP = CH["distil_variant"], CH["distil_pool"], CH["clip_img_prep"]
# validated choices from scripts/04_tuning.py (fall back to the proposal's settings)
HEAD = CH.get("head", {"hidden": TRAIN["hidden"], "dropout": TRAIN["dropout"], "lr": TRAIN["lr"]})
MDROP = CH.get("modality_dropout", TRAIN["modality_dropout"])
DLEN = CH.get("distil_len", 256)
XA = CH.get("xattn", {"layers": TRAIN["xattn_layers"], "lr": TRAIN["lr_xattn"], "d": TRAIN["xattn_dim"]})
CFG = dict(TRAIN, hidden=HEAD["hidden"], dropout=HEAD["dropout"], lr=HEAD["lr"])
# patch tokens for cross-attention: the same poster encoding as the vector models when available
TOK_PREP = PREP if (FEAT / f"clip_img_{PREP}_tok.npy").exists() else "crop"
dev = device()
runs_path = RESULTS / "runs.json"
RUNS = json.load(open(runs_path)) if runs_path.exists() else {}
RUNS["_choices"] = CH
print("choices:", CH, flush=True)


def l2(x):
    x = x.astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def load(name):
    return np.load(FEAT / f"{name}.npy")


F_IMG = {"clip": lambda: l2(load(f"clip_img_{PREP}")), "resnet": lambda: load("img_resnet50"), "vit": lambda: load("img_vit")}
F_TXT = {"distil": lambda: load(f"txt_distil{'' if DLEN == 256 else DLEN}_{TV}_{POOL}"), "bert": lambda: load(f"txt_bert_{TV}_mean"),
         "cliptxt": lambda: l2(load(f"clip_txt_{TV}"))}
DIM = {"clip": 512, "resnet": 2048, "vit": 768, "distil": 768, "bert": 768, "cliptxt": 512}


def save_runs():
    json.dump(RUNS, open(runs_path, "w"), indent=1)


def split_f1(P_tr, P_va, P_te, th, idx_tr):
    """Macro-F1 on train / val / test with the SAME validation-tuned thresholds (proposal objective 2)."""
    return {s: float(f1_score(Y[i], predict(P, th), average="macro", zero_division=0))
            for s, P, i in (("train", P_tr, idx_tr), ("val", P_va, va), ("test", P_te, te))}


def record(name, Pva, Pte, info, kind, Ptr=None, tr_idx=None, extra=None):
    th = tune_thresholds(Y[va], Pva)
    res = {"kind": kind, "val": all_metrics(Y[va], Pva, th), "test": all_metrics(Y[te], Pte, th),
           "test_AP": dict(zip(GENRES, per_class_ap(Y[te], Pte).round(4).tolist())),
           "test_F1": dict(zip(GENRES, f1_score(Y[te], predict(Pte, th), average=None, zero_division=0).round(4).tolist())),
           **info, **(extra or {})}
    if Ptr is not None:
        res["macro_F1_by_split"] = split_f1(Ptr, Pva, Pte, th, tr_idx)
    np.savez_compressed(RESULTS / f"probs_{name}.npz", val=Pva.astype(np.float32), test=Pte.astype(np.float32), th=th)
    return res


def aggregate(name, per_seed):
    agg = dict(per_seed[0])
    for split in ("val", "test"):
        agg[f"{split}_mean"] = {k: float(np.mean([r[split][k] for r in per_seed])) for k in per_seed[0][split]}
        agg[f"{split}_sd"] = {k: float(np.std([r[split][k] for r in per_seed])) for k in per_seed[0][split]}
    if "macro_F1_by_split" in per_seed[0]:
        agg["macro_F1_by_split_mean"] = {s: float(np.mean([r["macro_F1_by_split"][s] for r in per_seed]))
                                         for s in ("train", "val", "test")}
    for k in [k for k in per_seed[0] if k.startswith("test_mAP_missing") or k.startswith("test_macroF1_missing")]:
        agg[k + "_mean"] = float(np.mean([r[k] for r in per_seed]))
    agg["test_F1_mean"] = {g: float(np.mean([r["test_F1"][g] for r in per_seed])) for g in GENRES}
    agg["n_seeds"] = len(per_seed)
    agg["histories"] = [r.pop("history") for r in per_seed if "history" in r] or agg.get("histories")
    agg.pop("history", None)
    RUNS[name] = agg
    save_runs()


def neural(name, build, arrays, kind, lr=None, missing_test=False):
    feats = Feats(arrays, Y)
    tr_probe = np.random.default_rng(0).choice(tr, 4000, replace=False)
    per_seed = []
    for sd in args.seeds:
        model, info = fit(build(), feats, tr, va, seed=sd, lr=lr, cfg=CFG, verbose=args.verbose)
        torch.save(model.state_dict(), MODELS / f"{name}_s{sd}.pt")
        Pva, Pte = predict_proba(model, feats, va, dev), predict_proba(model, feats, te, dev)
        Ptr = predict_proba(model, feats, tr_probe, dev)
        t0 = time.time()
        predict_proba(model, feats, te, dev)
        info["infer_ms_per_1k"] = round((time.time() - t0) / len(te) * 1e6, 1)
        info["params"] = M.n_params(model)
        extra = {}
        if missing_test:
            th = tune_thresholds(Y[va], Pva)
            for miss in ("img", "txt"):
                Pm = predict_proba(model, feats, te, dev, missing=miss)
                extra[f"test_mAP_missing_{miss}"] = float(per_class_ap(Y[te], Pm).mean())
                extra[f"test_macroF1_missing_{miss}"] = float(f1_score(Y[te], predict(Pm, th), average="macro", zero_division=0))
        if isinstance(model, M.GMU) and sd == args.seeds[0]:
            gates = []
            model.eval()
            with torch.no_grad():
                for s in range(0, len(te), 1024):
                    b, _ = feats.batch(te[s:s + 1024], dev)
                    model(b)
                    gates.append(model.last_gate.mean(1).cpu().numpy())
            np.save(RESULTS / f"gmu_gate_{name}.npy", np.concatenate(gates))
        r = record(name if sd == args.seeds[0] else f"_{name}_s{sd}", Pva, Pte, info, kind, Ptr, tr_probe, extra)
        print(f"  {name} seed{sd}: val macroF1 {r['val']['macro_F1']:.4f} mAP {r['val']['mAP']:.4f} | "
              f"test macroF1 {r['test']['macro_F1']:.4f} microF1 {r['test']['micro_F1']:.4f} mAP {r['test']['mAP']:.4f} "
              f"| ep {info['best_epoch']} {info['train_seconds']}s", flush=True)
        per_seed.append(r)
    aggregate(name, per_seed)


def single(name, r):
    r["test_mean"], r["test_sd"] = r["test"], {k: 0.0 for k in r["test"]}
    r["val_mean"], r["val_sd"] = r["val"], {k: 0.0 for k in r["val"]}
    r["test_F1_mean"] = r["test_F1"]
    r["n_seeds"] = 1
    RUNS[name] = r
    save_runs()


def tfidf_baseline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    v = CH["tfidf_variant"]
    texts = np.array(json.load(open(SPLITS / "texts" / f"{v}.json")), dtype=object)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True,
                          token_pattern=r"[\w#][\w#'-]*")
    Xtr = vec.fit_transform(texts[tr])                    # fitted on train only
    clf = OneVsRestClassifier(LogisticRegression(C=4, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(Xtr, Y[tr])
    secs = time.time() - t0
    Pva, Pte = clf.predict_proba(vec.transform(texts[va])), clf.predict_proba(vec.transform(texts[te]))
    probe = np.random.default_rng(0).choice(np.arange(len(tr)), 4000, replace=False)
    Ptr = clf.predict_proba(Xtr[probe])
    info = {"train_seconds": round(secs, 1), "params": int(sum(e.coef_.size for e in clf.estimators_)), "device": "cpu",
            "text_variant": v, "train_mAP_final": float(per_class_ap(Y[tr[probe]], Ptr).mean())}
    r = record("tfidf_lr", Pva, Pte, info, "A classical", Ptr, tr[probe])
    single("tfidf_lr", r)
    print(f"  tfidf_lr ({v}): val macroF1 {r['val']['macro_F1']:.4f} | test macroF1 {r['test']['macro_F1']:.4f} "
          f"mAP {r['test']['mAP']:.4f}", flush=True)


def seed_probs(name, sd):
    return np.load(RESULTS / (f"probs_{name}.npz" if sd == args.seeds[0] else f"probs__{name}_s{sd}.npz"))


def late(name, weighted):
    # one late-fusion model per seed, built from the image and text baselines trained with that seed
    per_seed = []
    for sd in args.seeds:
        a, b = seed_probs("img_clip", sd), seed_probs("txt_distil", sd)
        if weighted:        # per-genre weight on the poster, chosen on validation AP (proposal Appendix D.2 optimisation)
            ws = np.linspace(0, 1, 21)
            w = np.array([ws[np.argmax([per_class_ap(Y[va][:, [k]], (wi * a["val"] + (1 - wi) * b["val"])[:, [k]])[0]
                                        for wi in ws])] for k in range(len(GENRES))])
        else:
            w = np.full(len(GENRES), 0.5)
        Pva, Pte = w * a["val"] + (1 - w) * b["val"], w * a["test"] + (1 - w) * b["test"]
        f1_alone = {k: float(f1_score(Y[te], predict(z["test"], tune_thresholds(Y[va], z["val"])), average="macro", zero_division=0))
                    for k, z in (("img", a), ("txt", b))}
        info = {"params": RUNS["img_clip"]["params"] + RUNS["txt_distil"]["params"] + (len(GENRES) if weighted else 0),
                "train_seconds": RUNS["img_clip"]["train_seconds"] + RUNS["txt_distil"]["train_seconds"],
                "image_weight_per_genre": dict(zip(GENRES, w.round(2).tolist())),
                "test_mAP_missing_img": float(per_class_ap(Y[te], b["test"]).mean()),
                "test_mAP_missing_txt": float(per_class_ap(Y[te], a["test"]).mean()),
                "test_macroF1_missing_img": f1_alone["txt"], "test_macroF1_missing_txt": f1_alone["img"]}
        r = record(name if sd == args.seeds[0] else f"_{name}_s{sd}", Pva, Pte, info, "C late fusion")
        print(f"  {name} seed{sd}: val macroF1 {r['val']['macro_F1']:.4f} | test macroF1 {r['test']['macro_F1']:.4f} "
              f"mAP {r['test']['mAP']:.4f}", flush=True)
        per_seed.append(r)
    aggregate(name, per_seed)
    # train-split macro-F1 is not defined for a post-hoc mix of two separately trained models
    gaps = [RUNS["img_clip"]["macro_F1_by_split_mean"]["train"], RUNS["txt_distil"]["macro_F1_by_split_mean"]["train"]]
    RUNS[name]["macro_F1_by_split"] = {"train": None, "val": RUNS[name]["val_mean"]["macro_F1"],
                                       "test": RUNS[name]["test_mean"]["macro_F1"],
                                       "note": f"train not defined for a post-hoc mix; unimodal train macro-F1 {gaps}"}
    save_runs()


def want(n):
    return not args.only or n in args.only


# ------------------------------------------------------------------------------------ run
if want("tfidf_lr"):
    print("== A classical text baseline", flush=True)
    tfidf_baseline()

print("== B unimodal baselines + encoder comparison", flush=True)
UNI = {"img_clip": ("img", "clip"), "txt_distil": ("txt", "distil"), "img_resnet50": ("img", "resnet"),
       "img_vit": ("img", "vit"), "txt_bert": ("txt", "bert"), "txt_clip": ("txt", "cliptxt")}
for name, (key, enc) in UNI.items():
    if want(name):
        X = (F_IMG if key == "img" else F_TXT)[enc]()
        neural(name, lambda d=DIM[enc], key=key: M.Unimodal(d, key, hidden=HEAD["hidden"], p=HEAD["dropout"]), {key: X}, "B unimodal")

pair = None
if any(want(n) for n in ("early_fusion", "gmu", "early_no_moddrop")):
    pair = {"img": F_IMG["clip"](), "txt": F_TXT["distil"]()}
if want("late_avg"):
    print("== C late fusion", flush=True)
    late("late_avg", weighted=False)
if want("late_weighted"):
    late("late_weighted", weighted=True)
if want("early_fusion"):
    print("== D early fusion (proposal)", flush=True)
    neural("early_fusion", lambda: M.EarlyFusion(512, 768, hidden=HEAD["hidden"], p=HEAD["dropout"], mdrop=MDROP), pair, "D early fusion", missing_test=True)
if want("gmu"):
    print("== E GMU (extension)", flush=True)
    neural("gmu", lambda: M.GMU(512, 768, hidden=HEAD["hidden"], p=HEAD["dropout"], mdrop=MDROP), pair, "E GMU", missing_test=True)
if want("xattn"):
    print("== F cross-attention (extension)", flush=True)
    # image tokens (147 x 768 per movie for multi-crop) are memory-mapped rather than loaded into RAM
    tok = {"img_tok": np.load(FEAT / f"clip_img_{TOK_PREP}_tok.npy", mmap_mode="r"),
           "txt_tok": np.load(FEAT / f"txt_distil_{TV}_tok.npy"),
           "txt_mask": np.load(FEAT / f"txt_distil_{TV}_tokmask.npy")}
    print(f"   cross-attention image tokens: {TOK_PREP} {tok['img_tok'].shape}", flush=True)
    neural("xattn", lambda: M.CrossAttentionFusion(n_img=tok["img_tok"].shape[1], n_txt=tok["txt_tok"].shape[1],
                                                   d=XA["d"], layers=XA["layers"]), tok,
           "F cross-attention", lr=XA["lr"], missing_test=True)
    del tok

print("== G ablation", flush=True)
if want("early_no_moddrop"):
    neural("early_no_moddrop", lambda: M.EarlyFusion(512, 768, hidden=HEAD["hidden"], p=HEAD["dropout"], mdrop=0), pair, "G ablation", missing_test=True)
save_runs()
print("done", flush=True)
