"""Step 4 - hyper-parameter and design study on the VALIDATION split (test is never used for a choice).

Part "search" (before final training):
  A  poster preprocessing   centre crop (proposal) | letterbox | multi-crop (top+centre+bottom) | native aspect 224x352
  B  text context           DistilBERT 256 tokens (proposal) | 512 tokens
  C  head grid              hidden {128,256,512} x dropout {0.1,0.3,0.5} x lr {3e-4,1e-3,3e-3} (early fusion)
  D  modality dropout       p in {0, 0.15, 0.3, 0.5}: full-input vs missing-modality validation macro-F1
  G  cross-attention check  layers {1,2} x lr {1e-4,3e-4} x d {128,256}
Part "decision" (after final training, from saved probabilities):
  E  decision thresholds    0.5 | one global tuned threshold | per-genre tuned thresholds
  F  late-fusion weights    equal | one global weight | per-genre weights

Decision rule (fixed before running): a proposal setting is replaced only if the alternative improves mean
validation macro-F1 by more than max(0.005, 2 x seed s.d.). Otherwise the proposal's setting is kept.
Outputs: outputs/results/tuning.json, outputs/splits/choices.json (updated)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion import models as M  # noqa: E402
from mmfusion.config import FEAT, GENRES, RESULTS, SEEDS, SPLITS, TRAIN  # noqa: E402
from mmfusion.metrics import per_class_ap, predict, tune_thresholds  # noqa: E402
from mmfusion.train import Feats, device, fit, predict_proba  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["search", "decision"], required=True)
args = ap.parse_args()

meta = pd.read_csv(SPLITS / "meta.csv")
Y = meta[GENRES].values.astype(np.int8)
tr, va, te = (np.flatnonzero(meta.split.values == s) for s in (0, 1, 2))
dev = device()
CH_PATH = SPLITS / "choices.json"
CH = json.load(open(CH_PATH))
TUNE_PATH = RESULTS / "tuning.json"
TUNE = json.load(open(TUNE_PATH)) if TUNE_PATH.exists() else {}


def l2(x):
    x = x.astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def vmf1(P, Yv=None):
    """Validation macro-F1 with per-genre thresholds tuned on the same validation predictions (model selection score)."""
    Yv = Y[va] if Yv is None else Yv
    return float(f1_score(Yv, predict(P, tune_thresholds(Yv, P)), average="macro", zero_division=0))


def run(build, arrays, seeds=SEEDS, cfg=None, lr=None, missing=False):
    feats = Feats(arrays, Y)
    out = []
    for sd in seeds:
        m, info = fit(build(), feats, tr, va, seed=sd, lr=lr, cfg=cfg or TRAIN)
        Pv = predict_proba(m, feats, va, dev)
        r = {"val_macro_F1": vmf1(Pv), "val_mAP": float(per_class_ap(Y[va], Pv).mean()), "epoch": info["best_epoch"]}
        if missing:
            th = tune_thresholds(Y[va], Pv)
            for k in ("img", "txt"):
                Pm = predict_proba(m, feats, va, dev, missing=k)
                r[f"val_macro_F1_missing_{k}"] = float(f1_score(Y[va], predict(Pm, th), average="macro", zero_division=0))
        out.append(r)
    agg = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    agg["val_macro_F1_sd"] = float(np.std([o["val_macro_F1"] for o in out]))
    agg["n_seeds"] = len(seeds)
    return agg


def better(alt, base):
    """Decision rule: replace the proposal setting only for a clear validation gain."""
    margin = max(0.005, 2 * max(alt["val_macro_F1_sd"], base["val_macro_F1_sd"]))
    return alt["val_macro_F1"] - base["val_macro_F1"] > margin, margin


def save():
    json.dump(TUNE, open(TUNE_PATH, "w"), indent=1)
    json.dump(CH, open(CH_PATH, "w"), indent=1)


TXT = np.load(FEAT / f"txt_distil_{CH['distil_variant']}_{CH['distil_pool']}.npy")

if args.part == "search":
    # ---------------------------------------------------------------- A poster preprocessing
    A = {}
    for prep in ["crop", "lbox", "multi", "native"]:
        img = l2(np.load(FEAT / f"clip_img_{prep}.npy"))
        A[prep] = {"image_only": run(lambda: M.Unimodal(512, "img"), {"img": img}),
                   "early_fusion": run(lambda: M.EarlyFusion(512, 768), {"img": img, "txt": TXT})}
        print(f"A {prep:7s} image-only {A[prep]['image_only']['val_macro_F1']:.4f} ± {A[prep]['image_only']['val_macro_F1_sd']:.4f} | "
              f"early {A[prep]['early_fusion']['val_macro_F1']:.4f}", flush=True)
    best_alt = max([p for p in A if p != "crop"], key=lambda p: A[p]["image_only"]["val_macro_F1"])
    switch, margin = better(A[best_alt]["image_only"], A["crop"]["image_only"])
    CH["clip_img_prep"] = best_alt if switch else "crop"
    TUNE["A_poster_preprocessing"] = {"results": A, "best_alternative": best_alt, "margin": margin, "chosen": CH["clip_img_prep"]}
    save()
    IMG = l2(np.load(FEAT / f"clip_img_{CH['clip_img_prep']}.npy"))

    # ---------------------------------------------------------------- B text context
    B = {}
    for L, name in [(256, f"txt_distil_{CH['distil_variant']}_{CH['distil_pool']}"),
                    (512, f"txt_distil512_{CH['distil_variant']}_{CH['distil_pool']}")]:
        X = np.load(FEAT / f"{name}.npy")
        B[L] = {"text_only": run(lambda: M.Unimodal(768, "txt"), {"txt": X}),
                "early_fusion": run(lambda: M.EarlyFusion(512, 768), {"img": IMG, "txt": X})}
        print(f"B {L} tokens text-only {B[L]['text_only']['val_macro_F1']:.4f} | early {B[L]['early_fusion']['val_macro_F1']:.4f}", flush=True)
    switch, margin = better(B[512]["text_only"], B[256]["text_only"])
    CH["distil_len"] = 512 if switch else 256
    TUNE["B_text_context"] = {"results": {str(k): v for k, v in B.items()}, "margin": margin, "chosen": CH["distil_len"]}
    save()
    if CH["distil_len"] == 512:
        TXT = np.load(FEAT / f"txt_distil512_{CH['distil_variant']}_{CH['distil_pool']}.npy")
    pair = {"img": IMG, "txt": TXT}

    # ---------------------------------------------------------------- C head grid (early fusion)
    C = []
    for h, p, lr in itertools.product([128, 256, 512], [0.1, 0.3, 0.5], [3e-4, 1e-3, 3e-3]):
        cfg = dict(TRAIN, hidden=h, dropout=p, lr=lr)
        r = run(lambda h=h, p=p: M.EarlyFusion(512, 768, hidden=h, p=p), pair, seeds=[0], cfg=cfg, lr=lr)
        C.append({"hidden": h, "dropout": p, "lr": lr, **r})
        print(f"C h={h:3d} p={p:.1f} lr={lr:.0e}  val macroF1 {r['val_macro_F1']:.4f}  ep {r['epoch']}", flush=True)
    Cdf = pd.DataFrame(C)
    top = Cdf.sort_values("val_macro_F1", ascending=False).iloc[0]
    base_cfg = {"hidden": TRAIN["hidden"], "dropout": TRAIN["dropout"], "lr": TRAIN["lr"]}
    base3 = run(lambda: M.EarlyFusion(512, 768), pair)
    top_cfg = {"hidden": int(top.hidden), "dropout": float(top.dropout), "lr": float(top.lr)}
    if top_cfg != base_cfg:
        top3 = run(lambda: M.EarlyFusion(512, 768, hidden=top_cfg["hidden"], p=top_cfg["dropout"]), pair,
                   cfg=dict(TRAIN, **top_cfg), lr=top_cfg["lr"])
        switch, margin = better(top3, base3)
    else:
        top3, switch, margin = base3, False, 0.0
    CH["head"] = top_cfg if switch else base_cfg
    TUNE["C_head_grid"] = {"grid_seed0": C, "proposal_3seeds": base3, "grid_best": top_cfg, "grid_best_3seeds": top3,
                           "margin": margin, "chosen": CH["head"]}
    save()
    print("C chosen", CH["head"], flush=True)

    # ---------------------------------------------------------------- D modality dropout
    D = {}
    hc = CH["head"]
    for pm in [0.0, 0.15, 0.3, 0.5]:
        D[str(pm)] = run(lambda pm=pm: M.EarlyFusion(512, 768, hidden=hc["hidden"], p=hc["dropout"], mdrop=pm), pair,
                         cfg=dict(TRAIN, **hc), lr=hc["lr"], missing=True)
        r = D[str(pm)]
        print(f"D p={pm:.2f} full {r['val_macro_F1']:.4f}  no-poster {r['val_macro_F1_missing_img']:.4f}  "
              f"no-plot {r['val_macro_F1_missing_txt']:.4f}", flush=True)
    # rule: among settings whose full-input score is within the noise margin of the best, take the most robust
    best_full = max(v["val_macro_F1"] for v in D.values())
    ok = {k: v for k, v in D.items() if best_full - v["val_macro_F1"] <= max(0.005, 2 * v["val_macro_F1_sd"])}
    chosen = max(ok, key=lambda k: ok[k]["val_macro_F1_missing_img"] + ok[k]["val_macro_F1_missing_txt"])
    CH["modality_dropout"] = float(chosen)
    TUNE["D_modality_dropout"] = {"results": D, "chosen": float(chosen),
                                  "rule": "most robust among settings within noise of the best full-input score"}
    save()

    # ---------------------------------------------------------------- G cross-attention sanity check
    tokp = "crop"                              # patch tokens exist for the centre crop (49 tokens)
    tok = {"img_tok": np.load(FEAT / f"clip_img_{tokp}_tok.npy"), "txt_tok": np.load(FEAT / f"txt_distil_{CH['distil_variant']}_tok.npy"),
           "txt_mask": np.load(FEAT / f"txt_distil_{CH['distil_variant']}_tokmask.npy")}
    G = []
    for layers, lr, d in [(2, 3e-4, 256), (1, 3e-4, 256), (2, 1e-4, 256), (1, 1e-4, 128), (2, 3e-4, 128)]:
        r = run(lambda layers=layers, d=d: M.CrossAttentionFusion(n_txt=tok["txt_tok"].shape[1], d=d, layers=layers),
                tok, seeds=[0], lr=lr)
        G.append({"layers": layers, "lr": lr, "d": d, **r})
        print(f"G layers={layers} lr={lr:.0e} d={d}  val macroF1 {r['val_macro_F1']:.4f}  ep {r['epoch']}", flush=True)
    bestG = max(G, key=lambda g: g["val_macro_F1"])
    CH["xattn"] = {"layers": bestG["layers"], "lr": bestG["lr"], "d": bestG["d"]}
    TUNE["G_cross_attention"] = {"grid_seed0": G, "chosen": CH["xattn"]}
    save()
    print(json.dumps(CH, indent=1))

else:
    # ---------------------------------------------------------------- E thresholds
    # single-seed study (seed 0 predictions); the results tables report three-seed means
    E = {}
    for k in ["late_weighted", "early_fusion", "gmu", "xattn"]:
        z = np.load(RESULTS / f"probs_{k}.npz")
        Pv, Pt = z["val"], z["test"]
        grid = np.linspace(0.05, 0.95, 91)
        g = grid[np.argmax([f1_score(Y[va], predict(Pv, np.full(23, t)), average="macro", zero_division=0) for t in grid])]
        th = tune_thresholds(Y[va], Pv)
        E[k] = {}
        for name, t in [("fixed_0.5", np.full(23, .5)), ("global_tuned", np.full(23, g)), ("per_genre_tuned", th)]:
            E[k][name] = {"val_macro_F1": float(f1_score(Y[va], predict(Pv, t), average="macro", zero_division=0)),
                          "test_macro_F1": float(f1_score(Y[te], predict(Pt, t), average="macro", zero_division=0)),
                          "test_micro_F1": float(f1_score(Y[te], predict(Pt, t), average="micro", zero_division=0))}
        E[k]["global_threshold"] = float(g)
        print("E", k, {n: round(v["test_macro_F1"], 4) for n, v in E[k].items() if isinstance(v, dict)}, flush=True)
    TUNE["E_thresholds"] = E

    # ---------------------------------------------------------------- F late-fusion weights
    a, b = np.load(RESULTS / "probs_img_clip.npz"), np.load(RESULTS / "probs_txt_distil.npz")
    ws = np.linspace(0, 1, 21)

    def score(w):
        Pv, Pt = w * a["val"] + (1 - w) * b["val"], w * a["test"] + (1 - w) * b["test"]
        th = tune_thresholds(Y[va], Pv)
        return {"val_macro_F1": vmf1(Pv), "test_macro_F1": float(f1_score(Y[te], predict(Pt, th), average="macro", zero_division=0))}

    gw = ws[np.argmax([per_class_ap(Y[va], wi * a["val"] + (1 - wi) * b["val"]).mean() for wi in ws])]
    lw = json.load(open(RESULTS / "runs.json"))["late_weighted"]["image_weight_per_genre"]
    pw = np.array([lw[g] for g in GENRES])
    TUNE["F_late_weights"] = {"equal": score(np.full(23, .5)), "global": {**score(np.full(23, gw)), "w": float(gw)},
                              "per_genre": score(pw)}
    print("F", {k: round(v["test_macro_F1"], 4) for k, v in TUNE["F_late_weights"].items()}, flush=True)
    save()
