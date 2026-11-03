"""Per-modality preprocessing, data-quality flags and the leakage-safe train/val/test split.

Text  : five cleaning variants (raw -> detok -> number masking -> stop words -> corpus-redundant
        words) are built here and compared on validation; see the text section below.
Image : decode the mean-subtracted BGR storage (data.decode), letterbox 256x160 -> 256x256 with
        white bars (so titles/credits at the poster edges are not cropped), resize to 224 and apply
        each backbone's own normalisation (done on the GPU in featurise.py).
Split : posters that are (near-)identical and plots that are identical are merged into one group
        with union-find; groups are then split 60/10/30 (proposal) with iterative multi-label stratification
        (Sechidis et al., 2011) so rare genres appear in every split and duplicates never cross it.
"""
import re

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .config import DHASH_MAX_HAMMING, FRAGMENT_MIN_WORDS, SEED, SPLIT_FRACTIONS

# ------------------------------------------------------------------------------------------ text
# The Kaggle plots are lower-cased, whitespace-tokenised ("ex - partner ' s"), and every digit run
# was already replaced by the placeholder "N" by the dataset authors. Cleaning is split into
# composable steps so each one can be ablated on the validation split (scripts/03_ablation.py):
#   raw            text as shipped
#   detok          re-join split punctuation / clitics / hyphens, normalise ellipses and quotes
#   detok_num      + number masking: drop "(N)" / "(N - N)" year asides, N -> "#" (one vocab token)
#   ..._stop       + remove English stop words (negations kept)        -> for bag-of-words models
#   ..._domain     + remove corpus-redundant words: frequent in TRAIN plots yet equally common
#                    across all genres (max |log2 lift| < 0.25), learned on the training split only
_DETOK = [
    (re.compile(r"(?:\s*\.){3,}"), " ..."),                     # ". . . ." -> "..."
    (re.compile(r"\s+'\s*(s|t|re|ve|ll|d|m)\b"), r"'\1"),       # partner ' s -> partner's ; don ' t -> don't
    (re.compile(r"(\w)\s+-\s+(?=\w)"), r"\1-"),                  # ex - partner -> ex-partner (also N - year - old)
    (re.compile(r"\s+([,.;:!?)%])"), r"\1"),                     # word , -> word,
    (re.compile(r"([($])\s+"), r"\1"),                           # ( word -> (word
    (re.compile(r'"\s+([^"]*?)\s+"'), r'"\1"'),                  # " rock opera " -> "rock opera"
    (re.compile(r"\s{2,}"), " "),
]
_NUM = [
    (re.compile(r"\s*\(\s*N(?:\s*-\s*N)?\s*\)"), ""),             # "(N)" / "(N - N)": masked release years, no content
    (re.compile(r"\bN\b"), "#"),                                 # remaining numbers -> single mask token
    (re.compile(r"#(?:[.,]#)+"), "#"),                            # "#,#" / "#.#" -> "#"
    (re.compile(r"#\s+s\b"), "#s"),                              # "the # s" (decades) -> "the #s"
    (re.compile(r"\s{2,}"), " "),
]
_WORD = re.compile(r"[\w#][\w#'-]*|[^\w\s]")
NEGATIONS = {"no", "not", "nor", "never", "nothing", "none", "nobody", "neither", "without", "cannot"}


def detok(t: str) -> str:
    t = t.strip()
    for pat, rep in _DETOK:
        t = pat.sub(rep, t)
    return t.strip()


def mask_numbers(t: str) -> str:
    for pat, rep in _NUM:
        t = pat.sub(rep, t)
    return t.strip()


def stopword_set():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return set(ENGLISH_STOP_WORDS) - NEGATIONS


def remove_words(t: str, drop: set) -> str:
    toks = [w for w in _WORD.findall(t) if w.lower().strip("'") not in drop]
    return detok(" ".join(toks))


def domain_redundant_words(texts, Y, train_idx, min_df=0.01, quantile=0.10):
    """Words that are frequent in training plots but carry almost no genre information.

    For every word with document frequency >= min_df in the TRAINING split we compute the mutual
    information between "word present" and each genre label, summed over the 23 genres (bits).
    The least informative `quantile` of these frequent words is treated as corpus-redundant.
    Returns (set of words, {word: summed MI in millibits})."""
    from sklearn.feature_extraction.text import CountVectorizer

    def h(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    cv = CountVectorizer(token_pattern=r"[\w#][\w#'-]*", binary=True, min_df=min_df)
    X = cv.fit_transform([texts[i] for i in train_idx])
    Yt = Y[train_idx].astype(float)
    n = len(train_idx)
    df = np.asarray(X.sum(0)).ravel()
    pw = df / n
    co = np.asarray(X.T @ Yt)                                    # (V, G) co-occurrence counts
    p_g_w = co / df[:, None]
    p_g_nw = (Yt.sum(0) - co) / (n - df)[:, None]
    mi = (h(Yt.mean(0))[None, :] - (pw[:, None] * h(p_g_w) + (1 - pw[:, None]) * h(p_g_nw))).sum(1)
    vocab = np.array(cv.get_feature_names_out())
    keep = (mi <= np.quantile(mi, quantile)) & (vocab != "#")
    return set(vocab[keep]), dict(zip(vocab[keep], (1000 * mi[keep]).round(2).tolist()))




def make_variants(texts, Y, train_idx):
    """All cleaning variants (dict name -> list[str]) plus the learned redundant-word list."""
    d = [detok(t) for t in texts]
    dn = [mask_numbers(t) for t in d]
    stop = stopword_set()
    dns = [remove_words(t, stop) for t in dn]
    dom, dom_df = domain_redundant_words(dns, Y, train_idx)
    dnsd = [remove_words(t, dom) for t in dns]
    return {"raw": list(texts), "detok": d, "detok_num": dn, "detok_num_stop": dns,
            "detok_num_stop_domain": dnsd}, dom_df


# ----------------------------------------------------------------------------------------- image
def letterbox(x, size=256, fill=1.0):
    """torch (B,3,H,W) float in [0,1] -> (B,3,size,size), padding the short side symmetrically."""
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    ph, pw = size - h, size - w
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=fill)


def dhash(rgb, hash_w=9, hash_h=8):
    """64-bit difference hash (Krawetz, 2013) of (N,H,W,3) uint8 posters -> uint64[N]."""
    g = rgb.astype(np.float32).mean(-1)
    n, h, w = g.shape
    ys = np.linspace(0, h, hash_h + 1).astype(int)
    xs = np.linspace(0, w, hash_w + 1).astype(int)
    small = np.stack([np.stack([g[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean((1, 2)) for j in range(hash_w)], 1)
                      for i in range(hash_h)], 1)                       # (N, 8, 9)
    bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(n, -1)
    return np.packbits(bits, axis=1).view(">u8").ravel().astype(np.uint64)


def near_duplicate_pairs(hashes, max_dist=DHASH_MAX_HAMMING, chunk=2048):
    pairs = []
    for s in range(0, len(hashes), chunk):
        d = np.bitwise_count(hashes[s:s + chunk, None] ^ hashes[None, :])
        r, c = np.nonzero(d <= max_dist)
        r = r + s
        keep = c > r
        pairs += list(zip(r[keep].tolist(), c[keep].tolist()))
    return pairs


def image_stats(rgb):
    """Cheap quality descriptors used for the bias / failure analysis."""
    x = rgb.astype(np.float32)
    sat = x.max(-1) - x.min(-1)
    return {"brightness": x.mean((1, 2, 3)), "saturation": sat.mean((1, 2)),
            "grayscale": (sat.mean((1, 2)) < 8)}


# ----------------------------------------------------------------------------------------- split
class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def groups(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def duplicate_groups(n, image_pairs, texts):
    uf = UnionFind(n)
    for a, b in image_pairs:
        uf.union(a, b)
    first = {}
    for i, t in enumerate(texts):
        if t in first:
            uf.union(first[t], i)
        else:
            first[t] = i
    return uf.groups()


def exact_copies(Y, texts, poster_hashes):
    """True copies: same poster bytes, same plot and same labels. Marks every copy except the first row."""
    first, copy = {}, np.zeros(len(Y), bool)
    for i, key in enumerate(zip(poster_hashes, texts, map(bytes, Y.astype(np.int8)))):
        if key in first:
            copy[i] = True
        else:
            first[key] = i
    return copy


def quality_flags(Y, texts):
    words = np.array([len(t.split()) for t in texts])
    by_text = {}
    for i, t in enumerate(texts):
        by_text.setdefault(t, []).append(i)
    conflict = np.zeros(len(Y), bool)
    for m in by_text.values():
        if len(m) > 1 and (Y[m] != Y[m[0]]).any():
            conflict[m] = True                 # identical plot text, different genre labels -> label noise
    return {"fragment": words < FRAGMENT_MIN_WORDS, "label_conflict": conflict, "n_words": words}


def grouped_stratified_split(Y, groups, fractions=SPLIT_FRACTIONS, seed=SEED):
    """Returns split codes (0=train, 1=val, 2=test) per row."""
    ug, inv = np.unique(groups, return_inverse=True)
    Yg = np.zeros((len(ug), Y.shape[1]), np.int8)
    np.maximum.at(Yg, inv, Y)                                  # group label = union of member labels
    tr_f, va_f, te_f = fractions
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=te_f, random_state=seed)
    rest, test = next(s1.split(np.zeros(len(ug)), Yg))
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=va_f / (tr_f + va_f), random_state=seed)
    tr, va = next(s2.split(np.zeros(len(rest)), Yg[rest]))
    code_g = np.zeros(len(ug), np.int8)
    code_g[rest[va]] = 1
    code_g[test] = 2
    return code_g[inv]
