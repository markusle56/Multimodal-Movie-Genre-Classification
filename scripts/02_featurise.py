"""Step 2 - extract frozen features from the pretrained encoders.

Proposal encoders (main analysis)
    clip_img_{prep}.npy             (N, 512)       CLIP ViT-B/32 image embedding, one file per poster encoding:
                                                    crop = proposal (resize short side + centre crop), lbox = letterbox,
                                                    multi = mean of top/centre/bottom 224x224 crops (chosen),
                                                    native = 224x352 with the position grid interpolated to 11x7
    clip_img_{prep}_tok.npy         (N, T, 768)    CLIP patch tokens for cross-attention (--tokens; T = 49 crop, 147 multi)
    txt_distil_{variant}_{cls,mean}.npy (N, 768)   DistilBERT, 256 tokens, one file per text-cleaning variant
    txt_distil512_{variant}_{cls,mean}.npy         DistilBERT with a 512-token window (--distil-len 512; context study)
    txt_distil_{variant}_tok.npy    (N, 16, 768)   16-token chunk means (cross-attention input; --tokens)
    txt_distil_{variant}_tokmask.npy (N, 16) bool
Comparison encoders
    img_resnet50.npy (N, 2048), img_vit.npy (N, 768)      ImageNet CNN / ViT (letterboxed)
    txt_bert_{variant}_{cls,mean}.npy (N, 768)              BERT-base, 512 tokens
    clip_txt_{variant}.npy (N, 512)                         CLIP text tower (77 tokens)
    timing.json                                             wall-clock seconds per encoder

The exact sequence of calls is in run_pipeline.sh.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmfusion.config import (BERT_MAX_TOKENS, BERT_NAME, CLIP_ARCH, CLIP_WEIGHTS, DISTIL_MAX_TOKENS,  # noqa: E402
                             DISTIL_NAME, FEAT, IMG_SIZE, SPLITS, TEXT_TOKEN_CHUNK,
                             TEXT_VARIANTS, VIT_NAME)
from mmfusion.data import decode, images_memmap  # noqa: E402
from mmfusion.preprocess import letterbox  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", default=["distil", "clip_img"],
                help="any of: distil bert clip_img clip_txt resnet vit")
ap.add_argument("--variants", nargs="*", default=TEXT_VARIANTS, help="text-cleaning variants")
ap.add_argument("--preps", nargs="*", default=["crop", "lbox"], help="CLIP image preprocessing: crop lbox multi native")
ap.add_argument("--tokens", action="store_true", help="also save token-level features")
ap.add_argument("--distil-len", type=int, default=DISTIL_MAX_TOKENS, help="DistilBERT window (256 = proposal)")
ap.add_argument("--limit", type=int, default=None, help="debug: only the first n rows")
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
mm = images_memmap()
N = args.limit or mm.shape[0]
timing_path = FEAT / "timing.json"
timing = json.load(open(timing_path)) if timing_path.exists() else {}
print("device:", dev, "rows:", N, flush=True)


def texts_for(variant):
    return json.load(open(SPLITS / "texts" / f"{variant}.json"))[:N]


def to_square(x, prep):
    """x: (B,3,256,160) in [0,1] -> (B,3,224,224); "multi" -> (B,3,358,224); "native" -> (B,3,352,224)."""
    if prep == "multi":                                    # keep the full poster at CLIP's scale (crops taken later)
        return F.interpolate(x, size=(358, IMG_SIZE), mode="bicubic", antialias=True, align_corners=False)
    if prep == "native":                                   # full poster, aspect kept (11x7 patch grid of 32 px)
        return F.interpolate(x, size=(352, IMG_SIZE), mode="bicubic", antialias=True, align_corners=False)
    if prep == "lbox":                                     # pad short side (white), keep the whole poster
        return F.interpolate(letterbox(x), size=IMG_SIZE, mode="bicubic", antialias=True, align_corners=False)
    h, w = x.shape[-2:]                                    # crop: resize short side to 224, centre-crop 224
    s = IMG_SIZE / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bicubic", antialias=True, align_corners=False)
    top, left = (x.shape[-2] - IMG_SIZE) // 2, (x.shape[-1] - IMG_SIZE) // 2
    return x[..., top:top + IMG_SIZE, left:left + IMG_SIZE]


def image_batches(mean, std, prep="lbox", bs=args.batch):
    mean = torch.tensor(mean, device=dev)[:, None, None]
    std = torch.tensor(std, device=dev)[:, None, None]
    for s in range(0, N, bs):
        rgb = decode(np.asarray(mm[s:min(s + bs, N)]))
        x = torch.from_numpy(rgb).to(dev).permute(0, 3, 1, 2).float() / 255
        yield s, (to_square(x, prep).clamp(0, 1) - mean) / std


def save(name, arr):
    np.save(FEAT / f"{name}.npy", arr)
    print(f"  saved {name} {arr.shape} {arr.dtype}", flush=True)


def record(name, t0, n_params):
    dt = time.time() - t0
    timing[name] = {"seconds": round(dt, 1), "items_per_s": round(N / dt, 1), "params_M": round(n_params / 1e6, 1),
                    "device": dev}
    json.dump(timing, open(timing_path, "w"), indent=1)
    print(f"  {name}: {dt:.0f}s ({N / dt:.0f}/s)", flush=True)


def sync():
    if dev == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def encode_text(model, tk, texts, max_len, prefix, tokens):
    """Length-sorted batches; saves CLS, masked mean and (optionally) 16-token chunk means."""
    nch = max_len // TEXT_TOKEN_CHUNK
    cls, mean = np.zeros((N, 768), np.float16), np.zeros((N, 768), np.float16)
    if tokens:
        tok = np.lib.format.open_memmap(FEAT / f"{prefix}_tok.npy", "w+", np.float16, (N, nch, 768))
        tmask = np.zeros((N, nch), bool)
    order = np.argsort([-len(t) for t in texts])
    t0, bs = time.time(), 64
    for b in range(0, N, bs):
        idx = np.sort(order[b:b + bs])
        enc = tk([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                 return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state
        am = enc["attention_mask"].unsqueeze(-1).float()
        cls[idx] = h[:, 0].float().cpu().numpy()
        mean[idx] = ((h * am).sum(1) / am.sum(1)).float().cpu().numpy()
        if tokens:
            pad = (-h.shape[1]) % TEXT_TOKEN_CHUNK
            hc = F.pad(h * am, (0, 0, 0, pad)).reshape(len(idx), -1, TEXT_TOKEN_CHUNK, 768)
            mc = F.pad(am, (0, 0, 0, pad)).reshape(len(idx), -1, TEXT_TOKEN_CHUNK, 1)
            c = (hc.sum(2) / mc.sum(2).clamp(min=1)).float().cpu().numpy()
            tok[idx, :c.shape[1]] = c
            tmask[idx, :c.shape[1]] = (mc.sum(2).squeeze(-1) > 0).cpu().numpy()
    sync()
    if tokens:
        tok.flush()
        np.save(FEAT / f"{prefix}_tokmask.npy", tmask)
    save(f"{prefix}_cls", cls)
    save(f"{prefix}_mean", mean)
    return t0


def run_hf_text(name, max_len, short):
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).to(dev).eval()
    for v in args.variants:
        prefix = f"txt_{short}_{v}"
        print(f"== {prefix}", flush=True)
        t0 = encode_text(m, tk, texts_for(v), max_len, prefix, args.tokens)
        record(prefix, t0, sum(p.numel() for p in m.parameters()))
    del m


def clip_model():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
    return m.to(dev).eval(), open_clip.get_tokenizer(CLIP_ARCH)


@torch.no_grad()
def run_clip_img():
    mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    for prep in args.preps:
        print(f"== clip_img_{prep}", flush=True)
        m, _ = clip_model()
        m.visual.output_tokens = True
        if prep == "native":                               # interpolate the 7x7 position grid to 11x7 (H x W)
            pe = m.visual.positional_embedding.data
            grid = pe[1:].reshape(1, 7, 7, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(11, 7), mode="bicubic", align_corners=False)
            m.visual.positional_embedding = torch.nn.Parameter(
                torch.cat([pe[:1], grid.permute(0, 2, 3, 1).reshape(77, -1)], 0))
        emb = np.zeros((N, 512), np.float16)
        if args.tokens:                                    # multi-crop keeps all 3 x 49 patch tokens (whole poster)
            ntok = 147 if prep == "multi" else 77 if prep == "native" else 49
            tok = np.lib.format.open_memmap(FEAT / f"clip_img_{prep}_tok.npy", "w+", np.float16, (N, ntok, 768))
        t0 = time.time()
        for s, x in image_batches(mean, std, prep):
            if prep == "multi":                            # top / centre / bottom crops cover the whole poster
                crops = [x[..., o:o + IMG_SIZE, :] for o in (0, 67, 134)]
                outs = [m.visual(c) for c in crops]
                pooled = torch.stack([F.normalize(o_[0], dim=-1) for o_ in outs]).mean(0)
                emb[s:s + len(x)] = pooled.float().cpu().numpy()
                if args.tokens:                            # top, centre, bottom tokens in reading order
                    tok[s:s + len(x)] = torch.cat([o_[1] for o_ in outs], 1).float().cpu().numpy()
                continue
            pooled, tokens = m.visual(x)
            emb[s:s + len(x)] = pooled.float().cpu().numpy()
            if args.tokens:
                tok[s:s + len(x)] = tokens.float().cpu().numpy()
        sync()
        if args.tokens:
            tok.flush()
        record(f"clip_img_{prep}", t0, sum(p.numel() for p in m.visual.parameters()))
        save(f"clip_img_{prep}", emb)


@torch.no_grad()
def run_clip_txt():
    m, tk = clip_model()
    for v in args.variants:
        texts = texts_for(v)
        out, t0 = np.zeros((N, 512), np.float16), time.time()
        for s in range(0, N, 256):
            out[s:s + 256] = m.encode_text(tk(texts[s:min(s + 256, N)]).to(dev)).float().cpu().numpy()
        sync()
        record(f"clip_txt_{v}", t0, sum(p.numel() for p in m.transformer.parameters()))
        save(f"clip_txt_{v}", out)


@torch.no_grad()
def run_resnet():
    import torchvision
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    m.fc = torch.nn.Identity()
    m = m.to(dev).eval()
    out, t0 = np.zeros((N, 2048), np.float16), time.time()
    for s, x in image_batches([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        out[s:s + len(x)] = m(x).float().cpu().numpy()
    record("img_resnet50", t0, sum(p.numel() for p in m.parameters()))
    save("img_resnet50", out)


@torch.no_grad()
def run_vit():
    import timm
    m = timm.create_model(VIT_NAME, pretrained=True, num_classes=0).to(dev).eval()
    cfg = m.pretrained_cfg
    cls, t0 = np.zeros((N, 768), np.float16), time.time()
    for s, x in image_batches(cfg["mean"], cfg["std"]):
        cls[s:s + len(x)] = m.forward_features(x)[:, 0].float().cpu().numpy()
    record("img_vit", t0, sum(p.numel() for p in m.parameters()))
    save("img_vit", cls)


RUN = {"distil": lambda: run_hf_text(DISTIL_NAME, args.distil_len,
                                     "distil" if args.distil_len == DISTIL_MAX_TOKENS else f"distil{args.distil_len}"),
       "bert": lambda: run_hf_text(BERT_NAME, BERT_MAX_TOKENS, "bert"),
       "clip_img": run_clip_img, "clip_txt": run_clip_txt, "resnet": run_resnet, "vit": run_vit}
for name in args.only:
    RUN[name]()
    if dev == "mps":
        torch.mps.empty_cache()
print("featurise done", flush=True)
