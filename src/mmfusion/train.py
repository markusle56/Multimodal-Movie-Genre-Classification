"""One training loop for every head: AdamW + cosine schedule, BCE loss, early stopping on val mAP.

Per epoch it logs train/validation loss, mAP and macro-F1 (train = fixed 4k-movie subset in eval mode;
macro-F1 uses per-genre thresholds tuned on validation at that epoch, i.e. the test-time decision rule).
These are the learning curves for the proposal's train-vs-validation macro-F1 overfitting check.
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .config import TRAIN
from .metrics import per_class_ap, predict, tune_thresholds


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Feats:
    """Holds the feature tensors (CPU, float16) and builds device batches by row index."""

    def __init__(self, arrays, Y):
        # memory-mapped arrays stay on disk; rows are read per batch
        self.a = {k: (v if isinstance(v, np.memmap) else torch.from_numpy(np.asarray(v))) for k, v in arrays.items()}
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def batch(self, idx, dev, missing=None):
        idx_t = torch.from_numpy(np.asarray(idx))
        b = {}
        for k, v in self.a.items():
            x = torch.from_numpy(np.ascontiguousarray(v[np.sort(idx)][np.argsort(np.argsort(idx))])) if isinstance(v, np.memmap) else v[idx_t]
            b[k] = x.to(dev) if x.dtype == torch.bool else x.to(dev).float()
        if missing:
            b["missing"] = missing
        return b, self.Y[idx_t].to(dev)


@torch.no_grad()
def predict_proba(model, feats, idx, dev, bs=1024, missing=None):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        b, _ = feats.batch(idx[s:s + bs], dev, missing)
        out.append(torch.sigmoid(model(b)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _loss(model, feats, idx, dev, bs=1024):
    model.eval()
    tot = 0.0
    for s in range(0, len(idx), bs):
        b, y = feats.batch(idx[s:s + bs], dev)
        tot += F.binary_cross_entropy_with_logits(model(b), y, reduction="sum").item()
    return tot / (len(idx) * feats.Y.shape[1])


def fit(model, feats, tr, va, seed=0, lr=None, cfg=TRAIN, verbose=False):
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(dev)
    lr = lr or cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(tr) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=cfg["max_epochs"] * steps_per_epoch,
                                                pct_start=0.05, anneal_strategy="cos")
    tr_probe = rng.choice(tr, min(4000, len(tr)), replace=False)
    Ytr, Yva = feats.Y[tr_probe].numpy(), feats.Y[va].numpy()
    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(tr)
        run = 0.0
        for s in range(0, len(perm), cfg["batch_size"]):
            b, y = feats.batch(perm[s:s + cfg["batch_size"]], dev)
            loss = F.binary_cross_entropy_with_logits(model(b), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(y)
        Pva = predict_proba(model, feats, va, dev)
        Ptr = predict_proba(model, feats, tr_probe, dev)
        th = tune_thresholds(Yva, Pva)              # same decision rule as at test time
        h = {"epoch": ep + 1, "train_loss": run / len(perm), "val_loss": _loss(model, feats, va, dev),
             "train_mAP": float(per_class_ap(Ytr, Ptr).mean()), "val_mAP": float(per_class_ap(Yva, Pva).mean()),
             "train_macro_F1": float(f1_score(Ytr, predict(Ptr, th), average="macro", zero_division=0)),
             "val_macro_F1": float(f1_score(Yva, predict(Pva, th), average="macro", zero_division=0))}
        hist.append(h)
        if verbose:
            print(f"    ep{ep + 1:3d} tl={h['train_loss']:.4f} vl={h['val_loss']:.4f} "
                  f"tmAP={h['train_mAP']:.3f} vmAP={h['val_mAP']:.3f}", flush=True)
        if h["val_mAP"] > best + 1e-4:
            best, best_state, bad = h["val_mAP"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_mAP": best, "best_epoch": int(np.argmax([h["val_mAP"] for h in hist]) + 1),
                   "train_seconds": round(time.time() - t0, 1), "device": dev}
