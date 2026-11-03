"""Step 1 - data-quality checks, duplicate grouping, the fixed split and text-cleaning variants.

Outputs (outputs/splits/):
    meta.csv                 one row per movie: split, duplicate group, quality flags, image stats
    texts/<variant>.json     the five cleaning variants (see mmfusion.preprocess)
    text_variant_stats.json  length / vocabulary / token statistics per variant
    redundant_words.json     corpus-redundant words learned on the TRAINING split
    preprocess_summary.json
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion import preprocess as pp  # noqa: E402
from mmfusion.config import DATA, DISTIL_MAX_TOKENS, DISTIL_NAME, GENRES, SPLITS  # noqa: E402
from mmfusion.data import decode, images_memmap, load_table  # noqa: E402

t0 = time.time()
_, Y, texts = load_table()
mm = images_memmap()
N = len(Y)

# ------------------------------------------------------------------ image hashing + stats
hashes, md5, stats = [], [], {"brightness": [], "saturation": [], "grayscale": []}
for s in range(0, N, 1024):
    raw = np.asarray(mm[s:s + 1024])
    md5 += [hashlib.md5(r.tobytes()).hexdigest() for r in raw]
    rgb = decode(raw)
    hashes.append(pp.dhash(rgb))
    for k, v in pp.image_stats(rgb).items():
        stats[k].append(v)
hashes = np.concatenate(hashes)
stats = {k: np.concatenate(v) for k, v in stats.items()}
print(f"hashed {N} posters in {time.time() - t0:.0f}s")

exact_img = Counter(md5)
exact_txt = Counter(texts)


def dup_table(keys):
    """Proposal Table A.3 columns: groups, rows involved, redundant rows, groups with differing labels."""
    g = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    gs = [v for v in g.values() if len(v) > 1]
    return {"groups": len(gs), "rows": sum(map(len, gs)), "redundant_rows": sum(len(v) - 1 for v in gs),
            "groups_with_differing_labels": int(sum((Y[v] != Y[v[0]]).any() for v in gs))}
pairs = pp.near_duplicate_pairs(hashes)
groups = pp.duplicate_groups(N, pairs, texts)
flags = pp.quality_flags(Y, texts)
flags["duplicate_copy"] = pp.exact_copies(Y, texts, md5) & ~flags["label_conflict"]

SHARED = DATA / "shared_split.csv"                  # optional: the group's common split (columns idx, split)
if SHARED.exists():
    sh = pd.read_csv(SHARED).set_index("idx").split.reindex(range(N))
    code = {"train": 0, "val": 1, "validation": 1, "test": 2, 0: 0, 1: 1, 2: 2}
    split = sh.map(code).astype(int).values         # used exactly as shared, so test sets match across members
    split_source = f"shared file {SHARED.name}"
else:
    split = pp.grouped_stratified_split(Y, groups).astype(int)
    split[flags["label_conflict"]] = -1             # excluded: identical plot, contradictory labels
    split[flags["duplicate_copy"]] = -1             # excluded: second copy of the same movie (one copy kept)
    split_source = "regenerated (grouped iterative stratification, seed 42)"
tr = np.flatnonzero(split == 0)

# ------------------------------------------------------------------ text variants
variants, redundant = pp.make_variants(texts, Y, tr)
(SPLITS / "texts").mkdir(exist_ok=True)
for name, v in variants.items():
    json.dump(v, open(SPLITS / "texts" / f"{name}.json", "w"))
json.dump(variants["detok_num"], open(SPLITS / "texts_clean.json", "w"))      # display copy
json.dump(dict(sorted(redundant.items(), key=lambda x: x[1])), open(SPLITS / "redundant_words.json", "w"), indent=0)

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(DISTIL_NAME)
vstats = {}
for name, v in variants.items():
    ntok = np.array([len(x) for x in tok(v, add_special_tokens=True)["input_ids"]])
    words = [w for t in v for w in pp._WORD.findall(t) if w[0].isalnum() or w[0] == "#"]
    vstats[name] = {
        "mean_words": float(np.mean([len(t.split()) for t in v])),
        "vocab_size": len(set(words)),
        "mean_distilbert_tokens": float(ntok.mean()),
        "median_distilbert_tokens": float(np.median(ntok)),
        f"share_truncated_at_{DISTIL_MAX_TOKENS}": float((ntok > DISTIL_MAX_TOKENS).mean()),
        "share_truncated_at_512": float((ntok > 512).mean()),
        "share_with_N_or_mask": float(np.mean([(" N " in f" {t} ") or ("#" in t) for t in v])),
        "example": v[0][:220],
    }
json.dump(vstats, open(SPLITS / "text_variant_stats.json", "w"), indent=1)

meta = pd.DataFrame({"idx": np.arange(N), "split": split, "group": groups,
                     "n_words": flags["n_words"], "fragment": flags["fragment"],
                     "label_conflict": flags["label_conflict"], "duplicate_copy": flags["duplicate_copy"], **stats})
meta[GENRES] = Y
meta.to_csv(SPLITS / "meta.csv", index=False)

gsize = pd.Series(groups).value_counts()
names = {0: "train", 1: "val", 2: "test", -1: "excluded"}
summary = {
    "n_rows": N,
    "split_source": split_source,
    "duplicate_table_A3": {"plot_summary": dup_table(texts), "poster_image": dup_table(md5),
                           "plot_and_poster": dup_table([t + "|" + h for t, h in zip(texts, md5)])},
    "exact_duplicate_poster_groups": int(sum(c > 1 for c in exact_img.values())),
    "rows_in_exact_duplicate_posters": int(sum(c for c in exact_img.values() if c > 1)),
    "exact_duplicate_plot_groups": int(sum(c > 1 for c in exact_txt.values())),
    "rows_in_exact_duplicate_plots": int(sum(c for c in exact_txt.values() if c > 1)),
    "near_duplicate_image_pairs_dhash": len(pairs),
    "duplicate_groups_size_gt1": int((gsize > 1).sum()),
    "rows_in_duplicate_groups": int(gsize[gsize > 1].sum()),
    "fragments_lt_min_words": int(flags["fragment"].sum()),
    "label_conflict_rows_excluded": int(flags["label_conflict"].sum()),
    "duplicate_copy_rows_excluded": int(flags["duplicate_copy"].sum()),
    "grayscale_posters": int(stats["grayscale"].sum()),
    "split_sizes": {names[k]: int(v) for k, v in pd.Series(split).value_counts().sort_index().items()},
    "split_shares": {names[k]: round(float(v) / N, 4) for k, v in pd.Series(split).value_counts().sort_index().items()},
    "groups_crossing_splits": int((meta[meta.split >= 0].groupby("group").split.nunique() > 1).sum()),
    "genre_prevalence_by_split": {names[s]: Y[split == s].mean(0).round(4).tolist() for s in (0, 1, 2)},
    "min_positives_per_genre_by_split": {names[s]: int(Y[split == s].sum(0).min()) for s in (0, 1, 2)},
    "n_redundant_domain_words": len(redundant),
    "seconds": round(time.time() - t0, 1),
}
json.dump(summary, open(SPLITS / "preprocess_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "genre_prevalence_by_split"}, indent=1))
print(json.dumps({k: {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != "example"}
                  for k, v in vstats.items()}, indent=1))
print("redundant words:", list(redundant)[:60])
