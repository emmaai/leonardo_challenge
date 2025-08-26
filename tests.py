# Integration tests that exercise the *actual* default models (Hugging Face)
# These tests download real models and can be slow. They are skipped unless you set
#   RUN_HF_INTEGRATION=1
# and have the HF stack installed (transformers, sentence-transformers, torch, timm, etc.).

from __future__ import annotations

import os
from pathlib import Path
import importlib

import numpy as np
import pytest
from PIL import Image as PILImage
from matcher import TextImageMatcher

RUN_INTEGRATION = os.environ.get("RUN_HF_INTEGRATION", "1") == "1"


def _hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


skip_no_run = pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_HF_INTEGRATION=1 to run integration tests")
skip_no_hf = pytest.mark.skipif(not _hf_available(), reason="Hugging Face stack not installed")


def _make_test_image(path: Path, color=(255, 0, 0), size=(128, 128)) -> Path:
    img = PILImage.new("RGB", size, color)
    img.save(path)
    return path


@pytest.fixture(scope="session")
@skip_no_run
@skip_no_hf
def default_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Minimal YAML: omit `models:` so the class uses DEFAULT_MODEL_SPECS
    p = tmp_path_factory.mktemp("cfg") / "config.yaml"
    p.write_text(
        "weights: {alpha_embed: 0.4, beta_caption: 0.2, gamma_ground: 0.2, delta_vqa: 0.2}\n"
        "str_sim_threshold: 0.6\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="session")
@skip_no_run
@skip_no_hf
def matcher(default_yaml: Path) -> TextImageMatcher:
    try:
        return TextImageMatcher.from_yaml(default_yaml)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not instantiate default HF models: {e}")


@pytest.fixture(scope="session")
@skip_no_run
@skip_no_hf
def red_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_test_image(tmp_path_factory.mktemp("data") / "red.png", (255, 0, 0))


@skip_no_run
@skip_no_hf
def test_default_models_build(matcher: TextImageMatcher):
    # We only assert that the instances exist (actual accuracy is not the goal here)
    assert matcher.embed_model is not None
    assert matcher.captioner is not None
    assert matcher.sts_model is not None
    assert matcher.grounding_model is not None
    assert matcher.vqa_model is not None


@skip_no_run
@skip_no_hf
def test_embedding_interface(matcher: TextImageMatcher, red_image: Path):
    img = PILImage.open(red_image).convert("RGB")
    v_img = matcher.embed_model.encode_image(img)  # type: ignore[union-attr]
    v_txt = matcher.embed_model.encode_text("a red square")  # type: ignore[union-attr]
    assert isinstance(v_img, np.ndarray) and v_img.ndim == 1 and np.isfinite(v_img).all()
    assert isinstance(v_txt, np.ndarray) and v_txt.ndim == 1 and np.isfinite(v_txt).all()


@skip_no_run
@skip_no_hf
def test_captioner_interface(matcher: TextImageMatcher, red_image: Path):
    img = PILImage.open(red_image).convert("RGB")
    cap = matcher.captioner.caption(img)  # type: ignore[union-attr]
    assert isinstance(cap, str) and len(cap.strip()) > 0


@skip_no_run
@skip_no_hf
def test_sts_interface(matcher: TextImageMatcher):
    sim_same = matcher.sts_model.similarity("a red square", "a red square")  # type: ignore[union-attr]
    sim_diff = matcher.sts_model.similarity("a red square", "a blue circle")  # type: ignore[union-attr]
    # Basic sanity: identical vs different should not be inverted
    assert sim_same >= sim_diff


@skip_no_run
@skip_no_hf
def test_grounding_interface(matcher: TextImageMatcher, red_image: Path):
    img = PILImage.open(red_image).convert("RGB")
    dets = matcher.grounding_model.detect(img, targets=["square", "red object"])  # type: ignore[union-attr]
    assert isinstance(dets, dict)
    # do not assert specific keys; some models may return empty results on synthetic images


@skip_no_run
@skip_no_hf
def test_vqa_interface(matcher: TextImageMatcher, red_image: Path):
    img = PILImage.open(red_image).convert("RGB")
    p = matcher.vqa_model.yesno_prob(img, "Is it red?")  # type: ignore[union-attr]
    assert isinstance(p, float) and 0.0 <= p <= 1.0


@skip_no_run
@skip_no_hf
def test_score_all_smoke(matcher: TextImageMatcher, red_image: Path):
    scores = matcher.score_all(str(red_image), "a red square")
    assert set(scores.keys()) == {"embedding", "caption_compare", "grounded", "vqa", "aggregate"}
    for k, v in scores.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

