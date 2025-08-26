# text_image_matcher_utils.py
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image as PILImage
import numpy as np
from numpy.typing import NDArray

Embedding = NDArray

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

# ==============================
# Standalone VQA question maker
# ==============================

def generate_vqa_questions(prompt: str, limit: int = 12) -> list[str]:
    """Minimal, rule-based expansion of a prompt into yes/no checks.
    Replace with spaCy/LLM for better coverage.
    """
    text = prompt.strip()
    tokens = simple_tokenize(text)

    colors = {
        "red","blue","green","yellow","purple","orange","pink",
        "black","white","gray","grey","brown","gold","silver",
    }
    sizes = {"small","big","large","tiny","huge","massive","miniature"}

    # naïve object guess: non-attr tokens len>2
    objs: list[str] = []
    for t in tokens:
        if t in colors or t in sizes or len(t) <= 2:
            continue
        if t not in objs:
            objs.append(t)

    qs: list[str] = []

    # object presence
    for o in objs[:5]:
        qs.append(f"Is there a {o}?")

    # attribute association to nearest following token
    for i, t in enumerate(tokens):
        if t in colors or t in sizes:
            nearest: str | None = None
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j] not in colors and tokens[j] not in sizes and len(tokens[j]) > 2:
                    nearest = tokens[j]
                    break
            if nearest:
                qs.append(f"Is the {nearest} {t}?")

    # relation hints (very rough)
    toks = set(tokens)
    if "left" in toks and "right" in toks:
        qs.append("Is one object to the left of another object?")
    if "on" in toks:
        qs.append("Is one object on top of another object?")
    if "under" in toks:
        qs.append("Is one object under another object?")
    if "front" in toks and "behind" in toks:
        qs.append("Is one object in front of another object?")

    # unique & cap
    qs = list(dict.fromkeys(qs))
    return qs[:limit]

