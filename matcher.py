from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable, TypedDict, Any, Mapping
import importlib
import csv as _csv
import json as _json

import yaml 

from utils import (
    PILImage, Embedding,
    cosine, sigmoid, soft_f1,
    simple_tokenize, string_similarity, pairwise_best_match,
    generate_vqa_questions,
)

# ==========================
# Protocols (pluggable APIs)
# ==========================

@runtime_checkable
class EmbeddingModel(Protocol):
    def encode_image(self, image: PILImage.Image) -> Embedding: ...
    def encode_text(self, text: str) -> Embedding: ...


@runtime_checkable
class Captioner(Protocol):
    def caption(self, image: PILImage.Image) -> str: ...


@runtime_checkable
class STSModel(Protocol):
    """Semantic textual similarity."""
    def similarity(self, a: str, b: str) -> float: ...


class Detection(TypedDict, total=False):
    score: float
    bbox: tuple[float, float, float, float]
    attributes: list[str]


@runtime_checkable
class GroundingModel(Protocol):
    def detect(self, image: PILImage.Image, targets: list[str]) -> dict[str, list[Detection]]: ...


@runtime_checkable
class VQAModel(Protocol):
    def yesno_prob(self, image: PILImage.Image, question: str) -> float: ...


# ==========================
# YAML-driven model builders + default adapters
# ==========================

def _load_dotted(path: str) -> Any:
    """Load an object from a dotted path like 'module.sub:Attr' or 'module.sub.Attr'."""
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        parts = path.rsplit(".", 1)
        mod_name, attr = (parts[0], parts[1]) if len(parts) == 2 else (path, None)
    module = importlib.import_module(mod_name)
    return getattr(module, attr) if attr else module


def _instantiate(target: Any, kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = dict(kwargs or {})
    if callable(target):
        return target(**kwargs)
    raise TypeError(f"Target {target!r} is not callable for instantiation")


def _has_methods(obj: Any, *methods: str) -> bool:
    return all(callable(getattr(obj, m, None)) for m in methods)


# ---- Lightweight default adapters (used if wrapper not provided and methods missing)
class _CLIPLikeWrapper:
    def __init__(self, model: Any, image_method: str = "encode_image", text_method: str = "encode_text") -> None:
        self.m = model
        self.imethod = image_method
        self.tmethod = text_method
    def encode_image(self, image: PILImage.Image):
        fn = getattr(self.m, self.imethod)
        return fn(image)
    def encode_text(self, text: str):
        fn = getattr(self.m, self.tmethod)
        return fn(text)

class _CaptionCallableWrapper:
    def __init__(self, model: Any, method: str = "__call__", key: str | None = None) -> None:
        self.m = model
        self.method = method
        self.key = key
    def caption(self, image: PILImage.Image) -> str:
        if self.method == "__call__":
            out = self.m(image)
        else:
            out = getattr(self.m, self.method)(image)
        if isinstance(out, str):
            return out
        if self.key is not None and isinstance(out, (list, dict)):
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict) and self.key in first:
                    return str(first[self.key])
            if isinstance(out, dict) and self.key in out:
                return str(out[self.key])
        return str(out)

class _SBERTLikeSTSWrapper:
    def __init__(self, model: Any, encode_method: str = "encode") -> None:
        self.m = model
        self.encode_method = encode_method
    def similarity(self, a: str, b: str) -> float:
        import numpy as np
        enc = getattr(self.m, self.encode_method)
        v = enc([a, b])
        va, vb = np.asarray(v[0]), np.asarray(v[1])
        va = va / (np.linalg.norm(va) + 1e-9)
        vb = vb / (np.linalg.norm(vb) + 1e-9)
        return float(np.dot(va, vb))

class _GroundingForwardWrapper:
    def __init__(self, model: Any, method: str = "detect", extra: dict[str, Any] | None = None) -> None:
        self.m = model
        self.method = method
        self.extra = dict(extra or {})
    def detect(self, image: PILImage.Image, targets: list[str]):
        fn = getattr(self.m, self.method)
        return fn(image, targets=targets, **self.extra)

class _YesNoVQAWrapper:
    def __init__(self, model: Any, method: str = "__call__", yes_key: str = "yes", no_key: str = "no") -> None:
        self.m = model
        self.method = method
        self.yes_key = yes_key
        self.no_key = no_key
    def yesno_prob(self, image: PILImage.Image, question: str) -> float:
        if self.method == "__call__":
            out = self.m(image=image, question=question)
        else:
            out = getattr(self.m, self.method)(image=image, question=question)
        if isinstance(out, (int, float)):
            return float(out)
        if isinstance(out, str):
            t = out.strip().lower()
            if t in {"yes", "y", "true"}: return 1.0
            if t in {"no", "n", "false"}: return 0.0
            return 0.5
        if isinstance(out, dict):
            if self.yes_key in out and isinstance(out[self.yes_key], (int, float)):
                return float(out[self.yes_key])
            if self.no_key in out and isinstance(out[self.no_key], (int, float)):
                return 1.0 - float(out[self.no_key])
        # HF VQA pipeline results handled by HF adapter below
        return 0.5

# ---- HuggingFace-specific adapters (top-level so they are importable via dotted path)
class HFCLIPAdapter:
    """Self-contained CLIP encoder using transformers (requires torch).
    Provides encode_image / encode_text producing numpy arrays.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._torch = torch
    def encode_image(self, image: PILImage.Image) -> Embedding:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return feats[0].detach().cpu().numpy()
    def encode_text(self, text: str) -> Embedding:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return feats[0].detach().cpu().numpy()

class HFVQAPipelineAdapter:
    """Adapter over HF 'visual-question-answering' pipeline.
    Maps answers 'yes'/'no' with associated score into a probability.
    """
    def __init__(self, model: Any) -> None:
        self.pipe = model
    def yesno_prob(self, image: PILImage.Image, question: str) -> float:
        out = self.pipe(image=image, question=question)
        item = out[0] if isinstance(out, list) and out else (out if isinstance(out, dict) else None)
        if not isinstance(item, dict):
            return 0.5
        ans = str(item.get("answer", "")).strip().lower()
        score = float(item.get("score", 0.5))
        if ans in {"yes", "y", "true"}:
            return score
        if ans in {"no", "n", "false"}:
            return 1.0 - score
        return 0.5

class HFZeroShotDetAdapter:
    """Adapter over HF 'zero-shot-object-detection' pipeline (e.g., OWL-ViT).
    Returns mapping target -> list[Detection-like dict].
    """
    def __init__(self, model: Any) -> None:
        self.pipe = model
    def detect(self, image: PILImage.Image, targets: list[str]):
        if not targets:
            return {}
        results = self.pipe(image, candidate_labels=targets)
        items = results if isinstance(results, list) else [results]
        out: dict[str, list[dict[str, Any]]] = {}
        for d in items:
            lbl = str(d.get("label", ""))
            if not lbl:
                continue
            box = d.get("box", {})
            bbox = (
                float(box.get("xmin", box.get("x1", 0.0))),
                float(box.get("ymin", box.get("y1", 0.0))),
                float(box.get("xmax", box.get("x2", 0.0))),
                float(box.get("ymax", box.get("y2", 0.0))),
            )
            out.setdefault(lbl, []).append({"score": float(d.get("score", 0.0)), "bbox": bbox, "attributes": []})
        return out


def _build_model_from_spec(spec: Mapping[str, Any] | None) -> Any | None:
    if not spec:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError("Model spec must be a mapping")
    target_path = spec.get("target")
    if not target_path:
        return None
    ctor = _load_dotted(str(target_path))
    base = _instantiate(ctor, dict(spec.get("init", {})))
    wrapper_spec = spec.get("wrapper")
    if not wrapper_spec:
        return base
    if isinstance(wrapper_spec, str):
        wrap_ctor = _load_dotted(wrapper_spec)
        wrap_kwargs: dict[str, Any] = {"model": base}
    elif isinstance(wrapper_spec, Mapping):
        wrap_ctor = _load_dotted(str(wrapper_spec.get("target")))
        wrap_kwargs = dict(wrapper_spec.get("init", {}))
        wrap_kwargs.setdefault("model", base)
    else:
        raise TypeError("wrapper must be a string or mapping")
    return _instantiate(wrap_ctor, wrap_kwargs)

# Popular default model specs (can be overridden in YAML)
DEFAULT_MODEL_SPECS: dict[str, Mapping[str, Any]] = {
    # Embedding via CLIP (transformers)
    "embedding": {
        "target": "matcher:HFCLIPAdapter",
        "init": {"model_name": "openai/clip-vit-base-patch32"},
    },
    # Captioning via BLIP (transformers pipeline)
    "captioner": {
        "target": "transformers:pipeline",
        "init": {"task": "image-to-text", "model": "Salesforce/blip-image-captioning-large"},
        "wrapper": {"target": "matcher:_CaptionCallableWrapper", "init": {"method": "__call__", "key": "generated_text"}},
    },
    # STS via Sentence-Transformers
    "sts": {
        "target": "sentence_transformers:SentenceTransformer",
        "init": {"model_name_or_path": "all-mpnet-base-v2"},
        "wrapper": {"target": "matcher:_SBERTLikeSTSWrapper", "init": {"encode_method": "encode"}},
    },
    # Grounding via OWL-ViT zero-shot detector
    "grounding": {
        "target": "transformers:pipeline",
        "init": {"task": "zero-shot-object-detection", "model": "google/owlvit-base-patch32"},
        "wrapper": {"target": "matcher:HFZeroShotDetAdapter"},
    },
    # VQA via ViLT/BLIP VQA pipeline
    "vqa": {
        "target": "transformers:pipeline",
        "init": {"task": "visual-question-answering", "model": "dandelin/vilt-b32-finetuned-vqa"},
        "wrapper": {"target": "matcher:HFVQAPipelineAdapter"},
    },
}

# ==============
# Config objects
# ==============

@dataclass(slots=True)
class Weights:
    alpha_embed: float = 0.4   # joint-embedding
    beta_caption: float = 0.2  # caption-then-compare
    gamma_ground: float = 0.2  # grounded coverage
    delta_vqa: float = 0.2     # VQA compliance


@dataclass(slots=True)
class Calibration:
    num_random_negatives: int = 16  # (hook for your sampler if you add one)


@dataclass(slots=True)
class MatcherConfig:
    weights: Weights = field(default_factory=Weights)
    calibration: Calibration = field(default_factory=Calibration)
    ground_entity_w: float = 0.5
    ground_attr_w: float = 0.3
    ground_rel_w: float = 0.2
    str_sim_threshold: float = 0.6

    # -------- YAML helpers --------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MatcherConfig":
        # weights
        w = data.get("weights", {})
        weights = Weights(
            alpha_embed=float(w.get("alpha_embed", 0.4)),
            beta_caption=float(w.get("beta_caption", 0.2)),
            gamma_ground=float(w.get("gamma_ground", 0.2)),
            delta_vqa=float(w.get("delta_vqa", 0.2)),
        )
        # calibration
        c = data.get("calibration", {})
        calibration = Calibration(
            num_random_negatives=int(c.get("num_random_negatives", 16))
        )
        # grounded weights (support both flat keys and nested group)
        gw = data.get("grounded_weights", {})
        ground_entity_w = float(data.get("ground_entity_w", gw.get("entity", 0.5)))
        ground_attr_w   = float(data.get("ground_attr_w",   gw.get("attribute", 0.3)))
        ground_rel_w    = float(data.get("ground_rel_w",    gw.get("relation", 0.2)))

        str_sim_threshold = float(data.get("str_sim_threshold", 0.6))

        return cls(
            weights=weights,
            calibration=calibration,
            ground_entity_w=ground_entity_w,
            ground_attr_w=ground_attr_w,
            ground_rel_w=ground_rel_w,
            str_sim_threshold=str_sim_threshold,
        )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "MatcherConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("Config YAML must be a mapping at the top level.")
        return cls.from_mapping(raw)


# ===========
# Main class
# ===========

class TextImageMatcher:
    """
    Model-agnostic text–image match scorer with four components:
      (1) joint-embedding similarity
      (2) caption-then-compare
      (3) grounded concept coverage
      (4) VQA compliance

    YAML configuration schema (excerpt):
    ---
    weights:
      alpha_embed: 0.45
      beta_caption: 0.2
      gamma_ground: 0.2
      delta_vqa: 0.15
    calibration:
      num_random_negatives: 16
    grounded_weights:
      entity: 0.5
      attribute: 0.3
      relation: 0.2
    str_sim_threshold: 0.6

    models:  # any missing entries fall back to built-in DEFAULT_MODEL_SPECS
    """

    def __init__(
        self,
        config: MatcherConfig | None = None,
        *,
        embed_model: EmbeddingModel | None = None,
        captioner: Captioner | None = None,
        sts_model: STSModel | None = None,
        grounding_model: GroundingModel | None = None,
        vqa_model: VQAModel | None = None,
    ) -> None:
        self.cfg = config or MatcherConfig()
        self.embed_model = embed_model
        self.captioner = captioner
        self.sts_model = sts_model
        self.grounding_model = grounding_model
        self.vqa_model = vqa_model

    # ---- Constructors from YAML ----
    @classmethod
    def from_yaml(
        cls,
        cfg_path: str | Path,
        *,
        embed_model: EmbeddingModel | None = None,
        captioner: Captioner | None = None,
        sts_model: STSModel | None = None,
        grounding_model: GroundingModel | None = None,
        vqa_model: VQAModel | None = None,
    ) -> "TextImageMatcher":
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("YAML must be a mapping at the top level")

        cfg = MatcherConfig.from_mapping(raw)

        # Build models from YAML specs, filling in defaults where missing
        specs: dict[str, Any] = dict(DEFAULT_MODEL_SPECS)
        user_specs = raw.get("models", {})
        if isinstance(user_specs, Mapping):
            specs.update(user_specs)

        def _build(name: str, required: tuple[str, ...], default_wrapper: Any) -> Any | None:
            spec = specs.get(name) if isinstance(specs, Mapping) else None
            if spec is None:
                return None
            obj = _build_model_from_spec(spec)
            if obj is None:
                return None
            if _has_methods(obj, *required):
                return obj
            return default_wrapper(obj)

        built_embed = _build("embedding", ("encode_image", "encode_text"), _CLIPLikeWrapper)
        built_caption = _build("captioner", ("caption",), _CaptionCallableWrapper)
        built_sts = _build("sts", ("similarity",), _SBERTLikeSTSWrapper)
        built_ground = _build("grounding", ("detect",), _GroundingForwardWrapper)
        built_vqa = _build("vqa", ("yesno_prob",), _YesNoVQAWrapper)

        return cls(
            config=cfg,
            embed_model=embed_model or built_embed,
            captioner=captioner or built_caption,
            sts_model=sts_model or built_sts,
            grounding_model=grounding_model or built_ground,
            vqa_model=vqa_model or built_vqa,
        )

    # ---- Data I/O ----

    def read_csv(
        self,
        csv_path: str | Path,
        *,
        image_col: str = "image",
        text_col: str = "text",
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or image_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(f"CSV must contain columns '{image_col}' and '{text_col}'")
            for r in reader:
                img = r.get(image_col, "")
                txt = r.get(text_col, "")
                if not img:
                    continue
                rows.append({"image": img, "text": txt})
        return rows

    def _load_image(self, path_or_img: str | Path | PILImage.Image) -> PILImage.Image:
        if hasattr(path_or_img, "convert"):  # already PIL image
            return path_or_img  # type: ignore[return-value]
        p = Path(path_or_img)  # type: ignore[arg-type]
        if not p.exists():
            raise FileNotFoundError(str(p))
        return PILImage.open(p).convert("RGB")  # type: ignore[return-value]

    # ---- (1) Joint-embedding ----

    def score_embedding(
        self,
        image: str | Path | PILImage.Image,
        text: str,
        *,
        negative_texts: list[str] | None = None,
    ) -> float:
        if self.embed_model is None:
            base = Path(image).name if not hasattr(image, "convert") else "image"
            return string_similarity(base, text)

        img = self._load_image(image)
        img_emb = self.embed_model.encode_image(img)
        txt_emb = self.embed_model.encode_text(text)
        s = cosine(img_emb, txt_emb)  # [-1,1]

        if negative_texts:
            neg = [cosine(img_emb, self.embed_model.encode_text(tn)) for tn in negative_texts]
            mu = (sum(neg) / len(neg)) if neg else 0.0
            sd = (sum((x - mu) ** 2 for x in neg) / max(1, (len(neg) - 1))) ** 0.5 if len(neg) > 1 else 1.0
            z = (s - mu) / (sd or 1e-6)
            return sigmoid(z)  # [0,1]
        return 0.5 * (s + 1.0)

    # ---- (2) Caption-then-compare ----

    def score_caption_compare(self, image: str | Path | PILImage.Image, text: str) -> float:
        if self.captioner is None or self.sts_model is None:
            return string_similarity(text, "an image")
        img = self._load_image(image)
        hatT = self.captioner.caption(img)
        sim = self.sts_model.similarity(text, hatT)
        return sim if 0.0 <= sim <= 1.0 else 0.5 * (sim + 1.0)

    # ---- (3) Grounded coverage ----

    def _extract_targets(self, text: str) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
        tokens = simple_tokenize(text)
        colors = {
            "red","blue","green","yellow","purple","orange","pink",
            "black","white","gray","grey","brown","gold","silver",
        }
        sizes = {"small","big","large","tiny","huge","massive","miniature"}
        attrs = [t for t in tokens if t in colors or t in sizes]
        ents = [t for t in tokens if t not in attrs and len(t) > 2]
        ents = list(dict.fromkeys(ents))[:8]

        rels: list[tuple[str, str, str]] = []
        toks = set(tokens)
        if "left" in toks and "right" in toks:
            rels.append(("objA", "left_of", "objB"))
        if "on" in toks:
            rels.append(("objA", "on", "objB"))
        if "under" in toks:
            rels.append(("objA", "under", "objB"))
        if "front" in toks and "behind" in toks:
            rels.append(("objA", "in_front_of", "objB"))
        return ents[:8], attrs[:8], rels[:6]

    def score_grounded(self, image: str | Path | PILImage.Image, text: str) -> float:
        ents_T, attrs_T, rels_T = self._extract_targets(text)

        if self.grounding_model is None:
            base_tokens: list[str] = []
            if not hasattr(image, "convert"):
                base_tokens = simple_tokenize(Path(image).name)  # type: ignore[arg-type]
            match_e, tot_e = pairwise_best_match(ents_T, base_tokens, self.cfg.str_sim_threshold)
            prec_e = match_e / max(1, len(base_tokens))
            rec_e = match_e / max(1, tot_e)
            f1_e = soft_f1(prec_e, rec_e)

            match_a, tot_a = pairwise_best_match(attrs_T, base_tokens, self.cfg.str_sim_threshold)
            prec_a = match_a / max(1, len(base_tokens))
            rec_a = match_a / max(1, tot_a)
            f1_a = soft_f1(prec_a, rec_a)

            f1_r = 1.0 if rels_T and base_tokens else 0.0
        else:
            img = self._load_image(image)
            det_map = self.grounding_model.detect(img, targets=ents_T)

            hits_e = sum(1 for e in ents_T if det_map.get(e))
            f1_e = soft_f1(
                prec=hits_e / max(1, len(det_map)),
                rec=hits_e / max(1, len(ents_T)),
            )

            attr_hits = 0
            for a in attrs_T:
                found = False
                for dets in det_map.values():
                    for d in dets:
                        for at in d.get("attributes", []):
                            if string_similarity(a, at) >= self.cfg.str_sim_threshold:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    attr_hits += 1
            f1_a = soft_f1(
                prec=attr_hits / max(1, len(attrs_T)),
                rec=attr_hits / max(1, len(attrs_T)),
            )

            f1_r = 1.0 if len(ents_T) >= 2 and det_map else 0.0

        wE, wA, wR = self.cfg.ground_entity_w, self.cfg.ground_attr_w, self.cfg.ground_rel_w
        denom = max(1e-8, wE + wA + wR)
        return (wE * f1_e + wA * f1_a + wR * f1_r) / denom

    # ---- (4) VQA compliance ----

    def score_vqa_compliance(
        self,
        image: str | Path | PILImage.Image,
        text: str,
        *,
        questions: list[str] | None = None,
    ) -> float:
        qs = list(questions) if questions else generate_vqa_questions(text)
        if not qs:
            return 0.0

        if self.vqa_model is None:
            base = Path(image).name if not hasattr(image, "convert") else "image"
            return sum(string_similarity(base, q) for q in qs) / len(qs)

        img = self._load_image(image)
        probs = [float(self.vqa_model.yesno_prob(img, q)) for q in qs]
        return sum(probs) / len(probs)

    # ---- Composite ----

    def score_all(
        self,
        image: str | Path | PILImage.Image,
        text: str,
        *,
        negative_texts: list[str] | None = None,
        vqa_questions: list[str] | None = None,
    ) -> dict[str, float]:
        s_embed = self.score_embedding(image, text, negative_texts=negative_texts)
        s_cap = self.score_caption_compare(image, text)
        s_grd = self.score_grounded(image, text)
        s_vqa = self.score_vqa_compliance(image, text, questions=vqa_questions)

        w = self.cfg.weights
        total = max(1e-8, w.alpha_embed + w.beta_caption + w.gamma_ground + w.delta_vqa)
        agg = (w.alpha_embed * s_embed +
               w.beta_caption * s_cap +
               w.gamma_ground * s_grd +
               w.delta_vqa * s_vqa) / total

        return {
            "embedding": s_embed,
            "caption_compare": s_cap,
            "grounded": s_grd,
            "vqa": s_vqa,
            "aggregate": agg,
        }

# ======================
# CLI (Click) for batch scoring
# ======================

import click


@click.command(context_settings={"show_default": True})
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to YAML config (models + weights)")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with image/text columns")
@click.option("--image-col", default="image", help="Image column name in CSV")
@click.option("--text-col", default="text", help="Text column name in CSV")
@click.option("--out", "out_path", default="", help="Output CSV path (default: print JSONL to stdout)")
@click.option("--questions", default="", type=click.Path(exists=True, dir_okay=False),
              help="Optional path to a file with one VQA question per line")
@click.option("--negatives", default="", type=click.Path(exists=True, dir_okay=False),
              help="Optional path to a file with one negative text per line for embedding z-score")
def cli(config_path: str, csv_path: str, image_col: str, text_col: str, out_path: str, questions: str, negatives: str) -> None:
    """Run text–image matching over a CSV of pairs."""
    matcher = TextImageMatcher.from_yaml(config_path)
    rows = matcher.read_csv(csv_path, image_col=image_col, text_col=text_col)

    vqa_qs: list[str] | None = None
    if questions:
        with open(questions, "r", encoding="utf-8") as f:
            vqa_qs = [ln.strip() for ln in f if ln.strip()]

    negs: list[str] | None = None
    if negatives:
        with open(negatives, "r", encoding="utf-8") as f:
            negs = [ln.strip() for ln in f if ln.strip()]

    results: list[dict[str, Any]] = []
    for r in rows:
        scores = matcher.score_all(r["image"], r["text"], negative_texts=negs, vqa_questions=vqa_qs)
        results.append({**r, **scores})

    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(results[0].keys()) if results else ["image", "text"]
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in results:
                w.writerow(row)
        click.echo(f"Wrote {len(results)} rows to {out_path}")
    else:
        for row in results:
            click.echo(_json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    cli()
