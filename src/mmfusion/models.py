"""Classification heads trained on frozen encoder features.

All models output 23 logits (multi-label, sigmoid + BCE). Inputs are dicts so every model shares
one training loop:
    {"img": (B, Di), "txt": (B, Dt)}                                  vector models
    {"img_tok": (B, 49, 768), "txt_tok": (B, 16, 768), "txt_mask": (B, 16)}  token models

Fusion families compared (course taxonomy; Baltrusaitis et al., 2019):
    Unimodal       MLP on one modality                              (baselines)
    Late fusion    weighted average of unimodal probabilities       (decision level, in train.py)
    Early fusion   LayerNorm + concatenate embeddings -> joint MLP  (feature level, proposal Appendix D.1)
    GMU            Gated Multimodal Unit, Arevalo et al. (2017)     (learned per-dimension gating)
    Cross-attn     bidirectional cross-attention over image patch and text chunk tokens
                   (intermediate fusion; ViLBERT-style co-attention, Lu et al., 2019)
"""
import torch
import torch.nn as nn

from .config import N_CLASSES, TRAIN


def _mlp(d_in, hidden, p):
    """Proposal Appendix D head: LayerNorm -> Linear(hidden) -> ReLU -> Dropout(p) -> Linear(23)."""
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(hidden, N_CLASSES))


class Unimodal(nn.Module):
    """Text MLP (768 -> 256 -> 23) or poster MLP (512 -> 256 -> 23), proposal Appendix D.2."""

    def __init__(self, d_in, key, hidden=TRAIN["hidden"], p=TRAIN["dropout"]):
        super().__init__()
        self.key = key
        self.net = _mlp(d_in, hidden, p)

    def forward(self, b):
        return self.net(b[self.key])


def _modality_dropout(hi, ht, p, training):
    """During training, blank out one whole modality for a random p-fraction of the batch."""
    if not training or p == 0:
        return hi, ht
    B = hi.shape[0]
    r = torch.rand(B, device=hi.device)
    drop_i = (r < p / 2).float().unsqueeze(-1)
    drop_t = ((r >= p / 2) & (r < p)).float().unsqueeze(-1)
    return hi * (1 - drop_i), ht * (1 - drop_t)


def _missing(hi, ht, b):
    """Test-time missing-modality simulation: the same blanking used by modality dropout."""
    m = b.get("missing")
    return (hi * 0 if m == "img" else hi), (ht * 0 if m == "txt" else ht)


class EarlyFusion(nn.Module):
    """Proposal Appendix D.1: LayerNorm each embedding, concatenate, joint MLP (-> 256 -> ReLU -> dropout -> 23).

    Modality dropout (proposal section 8) blanks one normalised embedding for a random share of training rows."""

    def __init__(self, d_img, d_txt, hidden=TRAIN["hidden"], p=TRAIN["dropout"], mdrop=TRAIN["modality_dropout"]):
        super().__init__()
        self.ni, self.nt = nn.LayerNorm(d_img), nn.LayerNorm(d_txt)
        self.head = nn.Sequential(nn.Linear(d_img + d_txt, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, N_CLASSES))
        self.mdrop = mdrop

    def forward(self, b):
        xi, xt = _modality_dropout(self.ni(b["img"]), self.nt(b["txt"]), self.mdrop, self.training)
        xi, xt = _missing(xi, xt, b)
        return self.head(torch.cat([xi, xt], -1))


class GMU(nn.Module):
    """h = z * tanh(W_v x_v) + (1 - z) * tanh(W_t x_t),  z = sigmoid(W_z [x_v; x_t]) (Arevalo et al., 2017)."""

    def __init__(self, d_img, d_txt, hidden=TRAIN["hidden"], p=TRAIN["dropout"], mdrop=TRAIN["modality_dropout"]):
        super().__init__()
        self.ni, self.nt = nn.LayerNorm(d_img), nn.LayerNorm(d_txt)
        self.wi, self.wt = nn.Linear(d_img, hidden), nn.Linear(d_txt, hidden)
        self.wz = nn.Linear(d_img + d_txt, hidden)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(hidden, N_CLASSES)
        self.mdrop = mdrop
        self.last_gate = None

    def forward(self, b):
        xi, xt = self.ni(b["img"]), self.nt(b["txt"])
        xi, xt = _modality_dropout(xi, xt, self.mdrop, self.training)
        xi, xt = _missing(xi, xt, b)
        hi, ht = torch.tanh(self.wi(self.drop(xi))), torch.tanh(self.wt(self.drop(xt)))
        z = torch.sigmoid(self.wz(torch.cat([xi, xt], -1)))
        self.last_gate = z.detach()                         # share of the image path, for interpretation
        return self.head(self.drop(z * hi + (1 - z) * ht))


class CoAttentionLayer(nn.Module):
    """Text tokens attend to image tokens and vice versa, each followed by self-attention + FFN."""

    def __init__(self, d, heads, p):
        super().__init__()
        self.t2i = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.i2t = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.self_t = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.self_i = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.nt, self.ni = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, t, i, t_pad):
        tn, inn = self.nt(t), self.ni(i)
        t = t + self.t2i(tn, inn, inn, need_weights=False)[0]
        i = i + self.i2t(inn, tn, tn, key_padding_mask=t_pad, need_weights=False)[0]
        return self.self_t(t, src_key_padding_mask=t_pad), self.self_i(i)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_img=768, d_txt=768, n_img=49, n_txt=16, d=TRAIN["xattn_dim"], heads=TRAIN["xattn_heads"],
                 layers=TRAIN["xattn_layers"], p=TRAIN["dropout"] / 2, mdrop=TRAIN["modality_dropout"]):
        super().__init__()
        self.pi = nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, d))
        self.pt = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d))
        self.pos_i = nn.Parameter(torch.zeros(1, n_img + 1, d))
        self.pos_t = nn.Parameter(torch.zeros(1, n_txt + 1, d))
        self.cls_i = nn.Parameter(torch.zeros(1, 1, d))
        self.cls_t = nn.Parameter(torch.zeros(1, 1, d))
        for prm in (self.pos_i, self.pos_t, self.cls_i, self.cls_t):
            nn.init.trunc_normal_(prm, std=0.02)
        self.layers = nn.ModuleList([CoAttentionLayer(d, heads, p) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(2 * p), nn.Linear(2 * d, N_CLASSES))
        self.mdrop = mdrop

    def forward(self, b):
        B = b["img_tok"].shape[0]
        i = torch.cat([self.cls_i.expand(B, -1, -1), self.pi(b["img_tok"])], 1) + self.pos_i
        t = torch.cat([self.cls_t.expand(B, -1, -1), self.pt(b["txt_tok"])], 1) + self.pos_t
        if self.training and self.mdrop > 0:          # blank all tokens of one modality for some samples
            r = torch.rand(B, device=i.device)
            i = i * (r >= self.mdrop / 2).float()[:, None, None]
            t = t * ((r < self.mdrop / 2) | (r >= self.mdrop)).float()[:, None, None]
        if b.get("missing") == "img":
            i = i * 0
        elif b.get("missing") == "txt":
            t = t * 0
        t_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=t.device), ~b["txt_mask"]], 1)
        for layer in self.layers:
            t, i = layer(t, i, t_pad)
        return self.head(torch.cat([i[:, 0], t[:, 0]], -1))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
