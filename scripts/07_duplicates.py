"""Step 7 - near-duplicate poster analysis: what dHash finds, why the threshold is 4, and what leakage it prevents.

Outputs: outputs/results/duplicates.json, outputs/figures/{dup_distance,dup_examples,dup_clip_audit}.pdf|png
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmfusion import preprocess as pp  # noqa: E402
from mmfusion.config import DHASH_MAX_HAMMING, FEAT, GENRES, RESULTS, SEED, SPLITS  # noqa: E402
from mmfusion.data import decode, images_memmap, load_table  # noqa: E402

FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"savefig.bbox": "tight", "savefig.dpi": 300})

_, Y, texts = load_table()
mm = images_memmap()
N = len(Y)
hashes, md5 = [], []
for s in range(0, N, 1024):
    raw = np.asarray(mm[s:s + 1024])
    md5 += [hashlib.md5(r.tobytes()).hexdigest() for r in raw]
    hashes.append(pp.dhash(decode(raw)))
hashes = np.concatenate(hashes)
md5 = np.array(md5)

# all poster pairs within Hamming distance 10
MAXD = 10
pairs = []
for s in range(0, N, 2048):
    d = np.bitwise_count(hashes[s:s + 2048, None] ^ hashes[None, :])
    r, c = np.nonzero(d <= MAXD)
    r = r + s
    k = c > r
    pairs.append(np.stack([r[k], c[k], d[r[k] - s, c[k]]], 1))
P = pd.DataFrame(np.concatenate(pairs), columns=["a", "b", "d"])

E = np.load(FEAT / "clip_img_multi.npy").astype(np.float32)          # whole-poster embedding, as in the final models
E /= np.linalg.norm(E, axis=1, keepdims=True)
P["clip_cos"] = (E[P.a] * E[P.b]).sum(1)
P["byte_identical"] = md5[P.a] == md5[P.b]
P["same_labels"] = (Y[P.a] == Y[P.b]).all(1)
inter, union = (Y[P.a] & Y[P.b]).sum(1), (Y[P.a] | Y[P.b]).sum(1)
P["label_jaccard"] = inter / np.maximum(union, 1)
P["same_plot"] = np.array([texts[a] == texts[b] for a, b in zip(P.a, P.b)])

band = pd.cut(P.d, [-1, 0, 2, 4, 6, 8, 10], labels=["0", "1-2", "3-4", "5-6", "7-8", "9-10"])
T = P.groupby(band, observed=False).agg(pairs=("d", "size"), byte_identical=("byte_identical", "mean"),
                                        clip_cos_median=("clip_cos", "median"), clip_cos_ge_095=("clip_cos", lambda x: (x >= .95).mean()),
                                        same_labels=("same_labels", "mean"), label_jaccard=("label_jaccard", "mean"),
                                        same_plot=("same_plot", "mean"))
print(T.round(3))

# random-pair reference for CLIP cosine and label agreement
rng = np.random.default_rng(0)
ra, rb = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
ok = ra != rb
ref = {"clip_cos_median": float(np.median((E[ra[ok]] * E[rb[ok]]).sum(1))),
       "same_labels": float((Y[ra[ok]] == Y[rb[ok]]).all(1).mean()),
       "label_jaccard": float(((Y[ra[ok]] & Y[rb[ok]]).sum(1) / np.maximum((Y[ra[ok]] | Y[rb[ok]]).sum(1), 1)).mean())}

# what the threshold adds beyond exact (byte) duplicates
near = P[P.d <= DHASH_MAX_HAMMING]
added = near[~near.byte_identical]
meta = pd.read_csv(SPLITS / "meta.csv")
gsize = meta.group.value_counts()
groups_multi = gsize[gsize > 1]
exact_only_rows = set()
for h, c in Counter(md5).items():
    if c > 1:
        exact_only_rows |= set(np.flatnonzero(md5 == h))
for t, c in Counter(texts).items():
    if c > 1:
        exact_only_rows |= set(i for i, x in enumerate(texts) if x == t)
rows_in_groups = set(meta.idx[meta.group.isin(groups_multi.index)])

# leakage simulation: naive random 60/10/30 split without grouping
perm = np.random.default_rng(SEED).permutation(N)
tr_n, te_n = perm[:int(.6 * N)], perm[int(.7 * N):]
in_tr = np.zeros(N, bool)
in_tr[tr_n] = True
partner = {}
for a, b in zip(near.a, near.b):
    partner.setdefault(a, []).append(b)
    partner.setdefault(b, []).append(a)
for t, c in Counter(texts).items():
    if c > 1:
        ids = [i for i, x in enumerate(texts) if x == t]
        for i in ids:
            partner.setdefault(i, []).extend(j for j in ids if j != i)
leaky = [i for i in te_n if any(in_tr[j] for j in partner.get(i, []))]
copy_pred = np.array([Y[[j for j in partner[i] if in_tr[j]][0]] for i in leaky]) if leaky else np.zeros((0, 23))
leak = {"naive_split_test_movies": int(len(te_n)), "test_movies_with_train_duplicate": int(len(leaky)),
        "share_of_test": float(len(leaky) / len(te_n)),
        "copy_train_duplicate_labels_samples_F1": float(f1_score(Y[leaky], copy_pred, average="samples", zero_division=0)) if leaky else None,
        "copy_exact_label_match": float((copy_pred == Y[leaky]).all(1).mean()) if leaky else None}

# ------------------------------------------------------------------ CLIP audit: what does dHash miss?
grp, spl = meta.group.values, meta.split.values
clip_pairs = []
for s_ in range(0, N, 2048):
    S_ = E[s_:s_ + 2048] @ E.T
    r, c = np.nonzero(S_ >= 0.90)
    r = r + s_
    k = c > r
    clip_pairs.append(np.stack([r[k], c[k], S_[r[k] - s_, c[k]]], 1))
C = pd.DataFrame(np.concatenate(clip_pairs), columns=["a", "b", "cos"])
C[["a", "b"]] = C[["a", "b"]].astype(int)
C["byte"] = md5[C.a] == md5[C.b]
C["grouped"] = grp[C.a] == grp[C.b]
C["cross"] = (spl[C.a] != spl[C.b]) & (spl[C.a] >= 0) & (spl[C.b] >= 0)
C["same_labels"] = (Y[C.a] == Y[C.b]).all(1)
audit = {}
for lo, hi in [(0.95, 1.01), (0.90, 0.95)]:
    q = C[(C.cos >= lo) & (C.cos < hi) & ~C.byte]
    audit[f"{lo:.2f}-{min(hi, 1):.2f}"] = {"non_identical_pairs": int(len(q)), "already_grouped": int(q.grouped.sum()),
                                          "ungrouped": int((~q.grouped).sum()), "crossing_split": int(q.cross.sum()),
                                          "same_labels": float(q.same_labels.mean()) if len(q) else None}
# effect of franchise siblings: test movies with a CLIP>=0.90 sibling in train vs the rest (GMU model)
sib = set()
for a_, b_ in zip(C.a[C.cross & ~C.byte], C.b[C.cross & ~C.byte]):
    if spl[a_] == 2 and spl[b_] == 0:
        sib.add(a_)
    if spl[b_] == 2 and spl[a_] == 0:
        sib.add(b_)
from mmfusion.metrics import predict  # noqa: E402
te = np.flatnonzero(spl == 2)
pz = np.load(RESULTS / "probs_gmu.npz")
pred = predict(pz["test"], pz["th"])
has = np.array([i in sib for i in te])
audit["franchise_siblings"] = {"test_movies_with_train_sibling": int(has.sum()),
                               "samples_F1_with_sibling": float(f1_score(Y[te][has], pred[has], average="samples", zero_division=0)) if has.any() else None,
                               "samples_F1_rest": float(f1_score(Y[te][~has], pred[~has], average="samples", zero_division=0))}
print(json.dumps(audit, indent=1))

OUT = {"clip_audit": audit, "pairs_by_band": T.round(4).reset_index().rename(columns={"d": "hamming_band"}).to_dict(orient="records"),
       "random_pair_reference": ref, "threshold": DHASH_MAX_HAMMING,
       "pairs_within_threshold": int(len(near)), "byte_identical_pairs_within_threshold": int(near.byte_identical.sum()),
       "pairs_added_beyond_byte_identical": int(len(added)),
       "added_pairs_clip_cos_median": float(added.clip_cos.median()) if len(added) else None,
       "added_pairs_same_labels": float(added.same_labels.mean()) if len(added) else None,
       "rows_in_duplicate_groups": int(len(rows_in_groups)), "rows_in_exact_duplicates": int(len(exact_only_rows)),
       "rows_added_by_near_duplicates": int(len(rows_in_groups - exact_only_rows)),
       "groups": int(len(groups_multi)), "largest_group": int(groups_multi.max()),
       "group_size_counts": {int(k): int(v) for k, v in groups_multi.value_counts().sort_index().items()},
       "leakage_naive_split": leak}
json.dump(OUT, open(RESULTS / "duplicates.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "pairs_by_band"}, indent=1, default=float))

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
cnt = P.d.value_counts().reindex(range(MAXD + 1), fill_value=0)
ax[0].bar(cnt.index, cnt.values, color=["#2E7D6B" if d <= DHASH_MAX_HAMMING else "#9A9A9A" for d in cnt.index])
ax[0].set_yscale("log")
ax[0].set_xlabel("dHash Hamming distance (bits of 64)")
ax[0].set_ylabel("poster pairs (log)")
ax[0].axvline(DHASH_MAX_HAMMING + .5, color="#B3402A", ls="--", lw=1)
ax[0].set_title("(a) Pairs by distance (green = grouped)")
g = P.groupby("d").agg(cos=("clip_cos", "median"), lab=("same_labels", "mean")).reindex(range(MAXD + 1))
ax[1].plot(g.index, g.cos, "o-", color="#3D5FA8", label="median CLIP cosine")
ax[1].plot(g.index, g.lab, "s-", color="#DD8452", label="share with identical labels")
ax[1].axhline(ref["clip_cos_median"], color="#3D5FA8", ls=":", lw=1)
ax[1].axhline(ref["same_labels"], color="#DD8452", ls=":", lw=1)
ax[1].axvline(DHASH_MAX_HAMMING + .5, color="#B3402A", ls="--", lw=1)
ax[1].set_xlabel("dHash Hamming distance")
ax[1].set_ylim(0, 1.05)
ax[1].legend(fontsize=8, loc="center right")
ax[1].set_title("(b) Visual and label agreement")
fig.tight_layout()
fig.savefig(FIG / "dup_distance.pdf")
fig.savefig(FIG / "dup_distance.png")
plt.close(fig)

rng = np.random.default_rng(3)
show = []
for lo, hi, lab in [(0, 0, "d = 0, not byte-identical"), (1, 2, "d = 1-2"), (3, 4, "d = 3-4"), (6, 8, "d = 6-8 (not grouped)")]:
    pool = P[(P.d >= lo) & (P.d <= hi)]
    if lo == 0:
        pool = pool[~pool.byte_identical]
    if len(pool):
        show.append((lab, pool.iloc[rng.integers(0, len(pool))]))
fig, axes = plt.subplots(2, len(show), figsize=(2.4 * len(show), 6))
for k, (lab, r) in enumerate(show):
    for row, i in enumerate((int(r.a), int(r.b))):
        axes[row, k].imshow(decode(np.asarray(mm[i])))
        axes[row, k].axis("off")
        axes[row, k].set_title(", ".join(GENRES[g][:6] for g in np.flatnonzero(Y[i]))[:28], fontsize=7)
    axes[0, k].text(0.5, 1.18, f"{lab}\nCLIP cos {r.clip_cos:.2f}", transform=axes[0, k].transAxes, ha="center",
                    fontsize=8.5, weight="bold")
fig.subplots_adjust(hspace=0.15, top=0.86)
fig.savefig(FIG / "dup_examples.pdf")
fig.savefig(FIG / "dup_examples.png")
plt.close(fig)
q = C[(C.cos >= 0.925) & ~C.byte & ~C.grouped].sort_values("cos", ascending=False).head(8)
fig, axes = plt.subplots(2, len(q), figsize=(2.0 * len(q), 5.6))
for k, (_, r) in enumerate(q.iterrows()):
    for row, i in enumerate((int(r.a), int(r.b))):
        axes[row, k].imshow(decode(np.asarray(mm[i])))
        axes[row, k].axis("off")
    axes[0, k].set_title(f"CLIP cos {r.cos:.3f}" + ("\ncrosses split" if r.cross else "\nsame split"), fontsize=8)
fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.9)
fig.savefig(FIG / "dup_clip_audit.pdf")
fig.savefig(FIG / "dup_clip_audit.png")
plt.close(fig)
print("figures written")
