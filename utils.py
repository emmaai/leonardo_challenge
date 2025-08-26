from __future__ import annotations

from pathlib import Path
import re

from PIL import Image as PILImage
import numpy as np
from numpy.typing import NDArray

Embedding = NDArray[np.floating]

# ================
# Math utilities
# ================

def l2_normalize(v: Embedding) -> Embedding:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def cosine(a: Embedding, b: Embedding) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def soft_f1(prec: float, rec: float, eps: float = 1e-8) -> float:
    s = prec + rec
    return 0.0 if s < eps else (2.0 * prec * rec / (s + eps))

# ===================
# Text/token utilities
# ===================
_token_pattern = re.compile(r"[A-Za-z0-9\-]+")

# Pattern: .../users/<HASH>/...
_LEONARDO_USERS_RE = re.compile(r"/users/([A-Za-z0-9\-]+)/")


def url_to_local_png(url_or_path: str) -> str:
    """Map a Leonardo CDN URL to a local file name.

    Example:
    https://cdn.leonardo.ai/users/85498bb1-.../generations/.../file.jpg
    → ./85498bb1-....png

    If the pattern isn't found, the input is returned unchanged.
    """
    m = _LEONARDO_USERS_RE.search(url_or_path)
    if not m:
        return url_or_path
    return f"{m.group(1)}.png"


def simple_tokenize(s: str) -> list[str]:
    return _token_pattern.findall(s.lower())

def string_similarity(a: str, b: str) -> float:
    A, B = set(simple_tokenize(a)), set(simple_tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def pairwise_best_match(targets: list[str], found: list[str], thresh: float) -> tuple[int, int]:
    matched = 0
    used: set[int] = set()
    for t in targets:
        best, best_j = -1.0, -1
        for j, f in enumerate(found):
            if j in used:
                continue
            sim = string_similarity(t, f)
            if sim > best:
                best, best_j = sim, j
        if best >= thresh and best_j >= 0:
            matched += 1
            used.add(best_j)
    return matched, len(targets)
