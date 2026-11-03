"""Loaders for the Simplified MM-IMDb Kaggle release (javierurea/simplified-mm-imdb).

Files expected in config.DATA:
    data.npy    object array (N, 3): [image_index, one_hot_genres(23), plot_text]
    images.npz  uncompressed zip holding images.npy, uint8 (N, 3, 256, 160)

IMPORTANT (found in the EDA): images.npy does NOT hold displayable RGB. Each poster was stored as
(BGR pixel - Caffe/VGG ImageNet channel mean) and then cast to uint8, so negative values wrapped
around. `decode()` undoes this exactly (up to +-1 grey level from the mean's rounding) and returns RGB.
"""
import struct
import zipfile

import numpy as np

from .config import DATA

BGR_MEAN = np.array([104, 117, 124], dtype=np.uint16)[:, None, None]


def load_table(path=None):
    """Returns (img_idx: int[N], Y: int8[N, 23], texts: list[str])."""
    raw = np.load(path or DATA / "data.npy", allow_pickle=True)
    Y = np.stack([np.asarray(r, dtype=np.int8) for r in raw[:, 1]])
    return raw[:, 0].astype(int), Y, raw[:, 2].astype(str).tolist()


def images_memmap(path=None):
    """Zero-copy (N, 3, 256, 160) uint8 view of the array stored uncompressed inside images.npz.

    Memory-mapping avoids loading 3.2 GB into RAM (needed on Colab and 16 GB laptops)."""
    path = path or DATA / "images.npz"
    zf = zipfile.ZipFile(path)
    info = zf.getinfo("images.npy")
    assert info.compress_type == zipfile.ZIP_STORED, "images.npz must be uncompressed to memory-map"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        n, m = struct.unpack("<HH", f.read(30)[26:30])        # local header: name + extra lengths
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        read_hdr = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_hdr(f)
        offset = f.tell()
    assert not fortran
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def decode(raw_chw):
    """(..., 3, H, W) stored uint8 -> (..., H, W, 3) displayable RGB uint8."""
    x = ((raw_chw.astype(np.uint16) + BGR_MEAN) % 256).astype(np.uint8)
    x = x[..., ::-1, :, :]                                      # BGR -> RGB
    return np.ascontiguousarray(np.moveaxis(x, -3, -1))

