"""Perceptual hashes (aHash, dHash, pHash) and near-duplicate search by Hamming distance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_LANCZOS = Image.Resampling.LANCZOS


class ImageHash:
    """A 64-bit (by default) perceptual hash. `a - b` is the Hamming distance."""

    def __init__(self, bits: np.ndarray):
        self.bits = np.asarray(bits, dtype=bool).flatten()

    def __sub__(self, other: "ImageHash") -> int:
        if self.bits.shape != other.bits.shape:
            raise ValueError("hashes have different sizes")
        return int(np.count_nonzero(self.bits != other.bits))

    def __eq__(self, other) -> bool:
        return isinstance(other, ImageHash) and np.array_equal(self.bits, other.bits)

    def __hash__(self):
        return hash(self.bits.tobytes())

    def __str__(self) -> str:
        width = (len(self.bits) + 3) // 4
        return format(int("".join("1" if b else "0" for b in self.bits), 2), f"0{width}x")

    __repr__ = __str__

    @classmethod
    def from_hex(cls, s: str, bits: int = 64) -> "ImageHash":
        return cls(np.array([c == "1" for c in format(int(s, 16), f"0{bits}b")]))


def _load(image) -> Image.Image:
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    return ImageOps.exif_transpose(img).convert("L")


def average_hash(image, size: int = 8) -> ImageHash:
    px = np.asarray(_load(image).resize((size, size), _LANCZOS), dtype=np.float64)
    return ImageHash(px > px.mean())


def difference_hash(image, size: int = 8) -> ImageHash:
    px = np.asarray(_load(image).resize((size + 1, size), _LANCZOS), dtype=np.float64)
    return ImageHash(px[:, 1:] > px[:, :-1])


def _dct_matrix(n: int) -> np.ndarray:
    k, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return 2 * np.cos(np.pi * k * (2 * i + 1) / (2 * n))  # unnormalized DCT-II


def phash(image, size: int = 8, highfreq_factor: int = 4) -> ImageHash:
    """DCT-based hash: keep the lowest size x size frequencies of a 32x32 image, threshold at their median."""
    n = size * highfreq_factor
    px = np.asarray(_load(image).resize((n, n), _LANCZOS), dtype=np.float64)
    d = _dct_matrix(n)
    low = (d @ px @ d.T)[:size, :size]
    return ImageHash(low > np.median(low))


HASHES = {"phash": phash, "dhash": difference_hash, "ahash": average_hash}


def similarity(distance: int, bits: int = 64) -> float:
    return 1 - distance / bits


@dataclass
class Match:
    a: Path
    b: Path
    distance: int

    @property
    def similarity(self) -> float:
        return similarity(self.distance)


def compare(a, b, method: str = "phash") -> int:
    fn = HASHES[method]
    return fn(a) - fn(b)


def find_duplicates(folder: Path, threshold: int = 10, method: str = "phash", recursive: bool = True) -> list[Match]:
    """All pairs of images in `folder` whose hashes differ in at most `threshold` bits, closest first."""
    files = sorted(p for p in (folder.rglob("*") if recursive else folder.iterdir())
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    hashes = []
    for f in files:
        try:
            hashes.append((f, HASHES[method](f)))
        except (OSError, ValueError):
            continue  # unreadable or not really an image
    matches = [Match(fa, fb, ha - hb) for i, (fa, ha) in enumerate(hashes) for fb, hb in hashes[i + 1:] if ha - hb <= threshold]
    return sorted(matches, key=lambda m: (m.distance, str(m.a), str(m.b)))
