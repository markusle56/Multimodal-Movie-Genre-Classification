"""Central configuration: paths, label names, preprocessing and training hyper-parameters.

Every script imports from here so that one edit changes the whole pipeline.
Override the data location with the MMIMDB_DATA environment variable.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MMIMDB_DATA", ROOT / "data"))
OUT = ROOT / "outputs"
FEAT = OUT / "features"
SPLITS = OUT / "splits"
MODELS = OUT / "models"
RESULTS = OUT / "results"
for _d in (FEAT, SPLITS, MODELS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Column order of the one-hot genre vector. Not shipped with the Kaggle files; recovered in the EDA
# by matching column sums to the genre frequency table of Arevalo et al. (2017) (all 23 match).
GENRES = ["Drama", "Comedy", "Romance", "Thriller", "Crime", "Action", "Adventure", "Horror",
          "Documentary", "Mystery", "Sci-Fi", "Fantasy", "Family", "Biography", "War", "History",
          "Music", "Animation", "Musical", "Western", "Sport", "Short", "Film-Noir"]
N_CLASSES = len(GENRES)

# ---------------------------------------------------------------- preprocessing
# Proposal (Group 18, Project 2.1): 60.02 / 10.00 / 29.98 split without cross-split duplicate groups.
SPLIT_FRACTIONS = (0.60, 0.10, 0.30)       # train / val / test, grouped + iteratively stratified
DHASH_MAX_HAMMING = 4                       # posters within this dHash distance are one duplicate group
FRAGMENT_MIN_WORDS = 10                     # plots shorter than this are flagged as fragments
IMG_SIZE = 224

# Text-cleaning variants compared on VALIDATION (scripts/03_ablation.py); see preprocess.py.
TEXT_VARIANTS = ["raw", "detok", "detok_num", "detok_num_stop", "detok_num_stop_domain"]

# DistilBERT (proposal encoder) and BERT-base (longer-context comparison)
DISTIL_NAME = "distilbert-base-uncased"
DISTIL_MAX_TOKENS = 256                     # proposal: 256-token limit
BERT_NAME = "bert-base-uncased"
BERT_MAX_TOKENS = 512
TEXT_TOKEN_CHUNK = 16                       # token states mean-pooled in 16-token chunks for cross-attention

# CLIP ViT-B/32 (proposal image encoder). Proposal preprocessing = resize + centre crop;
# letterboxing (no crop) is compared as an ablation. ViT-B/32 at 224 has a native 7x7 patch grid.
CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-32-quickgelu", "openai"   # OpenAI weights were trained with QuickGELU
VIT_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"   # ImageNet ViT comparison

# ---------------------------------------------------------------- training
# Heads follow proposal Appendix D: 256-unit hidden layer, ReLU, dropout 0.3.
TRAIN = dict(
    batch_size=256,
    lr=1e-3,
    lr_xattn=3e-4,
    weight_decay=1e-4,
    max_epochs=80,
    patience=10,            # early stopping on validation mAP (threshold-free, stable)
    hidden=256,
    dropout=0.3,
    modality_dropout=0.15,  # proposal §8: modality dropout as a bias / robustness mitigation
    xattn_dim=256,
    xattn_heads=4,
    xattn_layers=2,
)
SEEDS = [0, 1, 2]           # every neural model is trained with three seeds; mean ± sd reported
N_BOOTSTRAP = 1000
