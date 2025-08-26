# matcher.py — full replacement with pluggable QG (question generation)
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable, TypedDict, Any, Mapping
import click
import importlib
import csv
import json as _json
import numpy as np
import yaml 

from utils import (
    PILImage, Embedding,
    cosine, sigmoid, soft_f1,
    simple_tokenize, string_similarity, pairwise_best_match,
    url_to_local_png
)

from qg_backends import QGBackend, T5QGAdapter

# ==========================
# Types & Protocols
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
# Default model specs (override via YAML models: block)
# ==========================
DEFAULT_MODEL_SPECS: dict[str, Any] = {
    "embedding": {
        "target": "matcher:HFCLIPAdapter",
        "init": {"model_name": "openai/clip-vit-base-patch32"},
    },
    "captioner": {
        "target": "transformers:pipeline",
        "init": {"task": "image-to-text", "model": "Salesforce/blip-image-captioning-large"},
        "wrapper": {"target": "matcher:CaptionCallableWrapper", "init": {"key": "generated_text"}},
    },
    "sts": {
        "target": "sentence_transformers:SentenceTransformer",
        "init": {"model_name_or_path": "all-mpnet-base-v2"},
        "wrapper": {"target": "matcher:SBERTLikeSTSWrapper", "init": {"encode_method": "encode"}},
    },
    "grounding": {
        "target": "transformers:pipeline",
        "init": {"task": "zero-shot-object-detection", "model": "google/owlvit-base-patch32"},
        "wrapper": {"target": "matcher:HFZeroShotDetAdapter"},
    },
    "vqa": {
        "target": "transformers:pipeline",
        "init": {"task": "visual-question-answering", "model": "dandelin/vilt-b32-finetuned-vqa"},
        "wrapper": {"target": "matcher:HFVQAPipelineAdapter"},
    },
}


# ==========================
# Config
# ==========================
@dataclass(slots=True)
class Weights:
    alpha_embed: float = 0.4
    beta_caption: float = 0.2
    gamma_ground: float = 0.2
    delta_vqa: float = 0.2


@dataclass(slots=True)
class Calibration:
    num_random_negatives: int = 16


@dataclass(slots=True)
class MatcherConfig:
    weights: Weights = field(default_factory=Weights)
    calibration: Calibration = field(default_factory=Calibration)
    ground_entity_w: float = 0.5
    ground_attr_w: float = 0.3
    ground_rel_w: float = 0.2
    str_sim_threshold: float = 0.6
    qg_max_questions: int = 12

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MatcherConfig":
        w = data.get("weights", {}) or {}
        weights = Weights(
            alpha_embed=float(w.get("alpha_embed", 0.4)),
            beta_caption=float(w.get("beta_caption", 0.2)),
            gamma_ground=float(w.get("gamma_ground", 0.2)),
            delta_vqa=float(w.get("delta_vqa", 0.2)),
        )
        c = data.get("calibration", {}) or {}
        calibration = Calibration(num_random_negatives=int(c.get("num_random_negatives", 16)))

        gw = data.get("grounded_weights", {}) or {}
        ground_entity_w = float(data.get("ground_entity_w", gw.get("entity", 0.5)))
        ground_attr_w = float(data.get("ground_attr_w", gw.get("attribute", 0.3)))
        ground_rel_w = float(data.get("ground_rel_w", gw.get("relation", 0.2)))

        str_sim_threshold = float(data.get("str_sim_threshold", 0.6))
        qg_max_questions = int(data.get("qg_max_questions", data.get("qg", {}).get("max_questions", 12)))

        return cls(
            weights=weights,
            calibration=calibration,
            ground_entity_w=ground_entity_w,
            ground_attr_w=ground_attr_w,
            ground_rel_w=ground_rel_w,
            str_sim_threshold=str_sim_threshold,
            qg_max_questions=qg_max_questions,
        )


# ==========================
# Wrappers / Adapters
# ==========================
class CLIPLikeWrapper:
    def __init__(self, model: Any, image_method: str = "encode_image", text_method: str = "encode_text") -> None:
        self.model = model
        self._img_m = image_method
        self._txt_m = text_method

    def encode_image(self, image: PILImage.Image) -> Embedding:
        return getattr(self.model, self._img_m)(image)

    def encode_text(self, text: str) -> Embedding:
        return getattr(self.model, self._txt_m)(text)


class CaptionCallableWrapper:
    def __init__(self, model: Any, method: str = "__call__", key: str | None = None) -> None:
        self.model = model
        self.method = method
        self.key = key

    def caption(self, image: PILImage.Image) -> str:
        fn = getattr(self.model, self.method)
        out = fn(image)
        if isinstance(out, str):
            return out
        if isinstance(out, list) and out and isinstance(out[0], dict) and self.key:
            return str(out[0].get(self.key, "")).strip()
        if isinstance(out, dict) and self.key:
            return str(out.get(self.key, "")).strip()
        # last resort
        return str(out)


class SBERTLikeSTSWrapper:
    def __init__(self, model: Any, encode_method: str = "encode") -> None:
        self.model = model
        self.encode_method = encode_method

    def similarity(self, a: str, b: str) -> float:
        enc = getattr(self.model, self.encode_method)
        va, vb = enc([a, b])
        va = np.asarray(va, dtype=np.float32)
        vb = np.asarray(vb, dtype=np.float32)
        s = cosine(va, vb)
        return s if 0.0 <= s <= 1.0 else 0.5 * (s + 1.0)


class _GroundingForwardWrapper:
    def __init__(self, model: Any) -> None:
        self.model = model

    def detect(self, image: PILImage.Image, targets: list[str]) -> dict[str, list[Detection]]:
        # Try HF zero-shot detection pipeline signature
        try:
            out = self.model(image, candidate_labels=targets)
        except TypeError:
            out = self.model.detect(image, targets=targets)
        # Normalize
        det_map: dict[str, list[Detection]] = {t: [] for t in targets}
        if isinstance(out, list):
            for d in out:
                lbl = str(d.get("label", ""))
                box = d.get("box", {}) or {}
                bbox = (
                    float(box.get("xmin") or box.get("x1") or 0.0),
                    float(box.get("ymin") or box.get("y1") or 0.0),
                    float(box.get("xmax") or box.get("x2") or 0.0),
                    float(box.get("ymax") or box.get("y2") or 0.0),
                )
                sc = float(d.get("score", 0.0))
                if lbl in det_map:
                    det_map[lbl].append({"score": sc, "bbox": bbox})
        elif isinstance(out, dict):
            det_map = out  # assume already normalized
        return det_map


class YesNoVQAWrapper:
    def __init__(self, model: Any) -> None:
        self.model = model

    def yesno_prob(self, image: PILImage.Image, question: str) -> float:
        try:
            out = self.model(image=image, question=question)
        except TypeError:
            out = self.model(question=question, image=image)
        # Interpret outputs
        if isinstance(out, (int, float)):
            p = float(out)
            return max(0.0, min(1.0, p))
        if isinstance(out, str):
            s = out.strip().lower()
            if s in {"yes", "true"}: return 1.0
            if s in {"no", "false"}:  return 0.0
            return 0.5
        if isinstance(out, dict):
            if "answer" in out and "score" in out:
                ans = str(out["answer"]).strip().lower()
                sc = float(out["score"])
                return sc if ans in {"yes", "true"} else (1.0 - sc)
            # maybe a {"no": 0.2, "yes": 0.8}
            if "yes" in out or "no" in out:
                py = float(out.get("yes", 0.0))
                pn = float(out.get("no", 0.0))
                s = py / (py + pn) if (py + pn) > 1e-8 else 0.5
                return max(0.0, min(1.0, s))
        return 0.5


# HF-specific light adapters (used by defaults)
class HFVQAPipelineAdapter:
    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    def yesno_prob(self, image: PILImage.Image, question: str) -> float:
        out = self.pipe(image=image, question=question)
        if isinstance(out, list):
            out = out[0]
        if isinstance(out, dict) and {"answer", "score"} <= out.keys():
            ans = str(out["answer"]).strip().lower()
            sc = float(out["score"]) if np.isfinite(out.get("score", 0.0)) else 0.5
            return sc if ans in {"yes", "true"} else (1.0 - sc)
        return 0.5


class HFZeroShotDetAdapter:
    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    def detect(self, image: PILImage.Image, targets: list[str]) -> dict[str, list[Detection]]:
        res = self.pipe(image, candidate_labels=targets)
        det_map: dict[str, list[Detection]] = {t: [] for t in targets}
        for r in res:
            lbl = str(r.get("label", ""))
            box = r.get("box", {}) or {}
            bbox = (
                float(box.get("xmin") or box.get("x1") or 0.0),
                float(box.get("ymin") or box.get("y1") or 0.0),
                float(box.get("xmax") or box.get("x2") or 0.0),
                float(box.get("ymax") or box.get("y2") or 0.0),
            )
            sc = float(r.get("score", 0.0))
            if lbl in det_map:
                det_map[lbl].append({"score": sc, "bbox": bbox})
        return det_map


class HFCLIPAdapter:
    """Minimal CLIP adapter (Transformers)."""
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None, model_kwargs: dict | None = None) -> None:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, **(model_kwargs or {}))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._torch = torch

    def encode_image(self, image: PILImage.Image) -> Embedding:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            v = self.model.get_image_features(**inputs)
        return v.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> Embedding:
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True).to(self.device)
        with self._torch.no_grad():
            v = self.model.get_text_features(**inputs)
        return v.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ==========================
# YAML helpers
# ==========================

def _load_dotted(target: str) -> Any:
    """Import a dotted object path like 'module:Symbol' or 'module.symbol'."""
    if ":" in target:
        mod, sym = target.split(":", 1)
    else:
        mod, sym = target.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, sym)


def _has_methods(obj: Any, *methods: str) -> bool:
    return all(hasattr(obj, m) for m in methods)


def _build_model_from_spec(spec: Mapping[str, Any]) -> Any:
    if not isinstance(spec, Mapping) or "target" not in spec:
        return None
    target = _load_dotted(str(spec["target"]))
    init = spec.get("init", {}) or {}
    base = target(**init) if callable(target) else target

    wrapper = spec.get("wrapper")
    if isinstance(wrapper, Mapping) and "target" in wrapper:
        w_cls = _load_dotted(str(wrapper["target"]))
        w_init = wrapper.get("init", {}) or {}
        # try common conventions
        try:
            return w_cls(base, **w_init)
        except TypeError:
            return w_cls(model=base, **w_init)
    return base


# ==========================
# Main matcher class
# ==========================
class TextImageMatcher:
    def __init__(
        self,
        config: MatcherConfig | None = None,
        *,
        embed_model: EmbeddingModel | None = None,
        captioner: Captioner | None = None,
        sts_model: STSModel | None = None,
        grounding_model: GroundingModel | None = None,
        vqa_model: VQAModel | None = None,
        qg_backend: QGBackend | None = None,
    ) -> None:
        self.cfg = config or MatcherConfig()
        self.embed_model = embed_model
        self.captioner = captioner
        self.sts_model = sts_model
        self.grounding_model = grounding_model
        self.vqa_model = vqa_model
        self.qg = qg_backend

    # ---- YAML constructor ----
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
        qg_backend: QGBackend | None = None,
    ) -> "TextImageMatcher":
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("YAML must be a mapping at the top level")

        cfg = MatcherConfig.from_mapping(raw)

        # Build model specs (defaults + user overrides)
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

        built_embed = _build("embedding", ("encode_image", "encode_text"), CLIPLikeWrapper)
        built_caption = _build("captioner", ("caption",), CaptionCallableWrapper)
        built_sts = _build("sts", ("similarity",), SBERTLikeSTSWrapper)
        built_ground = _build("grounding", ("detect",), _GroundingForwardWrapper)
        built_vqa = _build("vqa", ("yesno_prob",), YesNoVQAWrapper)

        # QG backend
        if qg_backend is None:
            qg_cfg = raw.get("qg", {}) if isinstance(raw, Mapping) else {}
            backend = (qg_cfg.get("backend") if isinstance(qg_cfg, Mapping) else None) or "builtin"
            if backend == "t5":
                t5_spec = qg_cfg.get("t5") if isinstance(qg_cfg, Mapping) else None
                if not isinstance(t5_spec, Mapping):
                    t5_spec = {
                        "target": "transformers:pipeline",
                        "init": {"task": "text2text-generation", "model": "google/flan-t5-base"},
                    }
                t5_pipe = _build_model_from_spec(t5_spec)
                qg_backend = T5QGAdapter(t5_pipe)

        return cls(
            config=cfg,
            embed_model=embed_model or built_embed,
            captioner=captioner or built_caption,
            sts_model=sts_model or built_sts,
            grounding_model=grounding_model or built_ground,
            vqa_model=vqa_model or built_vqa,
            qg_backend=qg_backend,
        )

    # ---- Data I/O ----
    def read_csv(self, csv_path: str | Path, *, image_col: str = "image", text_col: str = "text") -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or image_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(f"CSV must contain columns '{image_col}' and '{text_col}'")
            for r in reader:
                img = r.get(image_col, "").strip()
                txt = r.get(text_col, "")
                if not img:
                    continue
                img = f"{Path(csv_path).parent}/{url_to_local_png(img)}"
                rows.append({"image": img, "text": txt})
        return rows

    def _load_image(self, path_or_img: str | Path | PILImage.Image) -> PILImage.Image:
        if hasattr(path_or_img, "convert"):
            return path_or_img  # type: ignore[return-value]
        p = Path(path_or_img)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return PILImage.open(p).convert("RGB")

    # ---- (1) Joint-embedding ----
    def score_embedding(
        self,
        image: str | Path | PILImage.Image,
        text: str,
        *,
        negative_texts: Sequence[str] | None = None,
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
            return sigmoid(z)
        return 0.5 * (s + 1.0)

    # ---- (2) Caption-then-compare ----
    def score_caption_compare(self, image: str | Path | PILImage.Image, text: str) -> float:
        if self.captioner is None or self.sts_model is None:
            return string_similarity(text, "an image")
        img = self._load_image(image)
        hatT = self.captioner.caption(img)
        if not isinstance(hatT, str):
            if isinstance(hatT, list) and hatT and isinstance(hatT[0], dict):
                hatT = str(hatT[0].get("generated_text")
                           or hatT[0].get("caption")
                           or hatT[0].get("text")
                           or hatT)
            elif isinstance(hatT, dict):
                hatT = str(hatT.get("generated_text")
                           or hatT.get("caption")
                           or hatT.get("text")
                           or hatT)
            else:
                hatT = str(hatT)
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
                base_tokens = simple_tokenize(Path(image).name)
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
        questions: Sequence[str] | None = None,
    ) -> float:
        if questions:
            qs = list(questions)
        else:
            limit = getattr(self.cfg, "qg_max_questions", 12)
            if getattr(self, "qg", None) is not None:
                qs = self.qg.generate(text, limit=limit)  # type: ignore[union-attr]
            else:
                # ultra-minimal fallback in case no QG backend is set
                qs = [f"Is there {w}?" for w in simple_tokenize(text)][:limit] or ["Is something visible?"]
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
        negative_texts: Sequence[str] | None = None,
        vqa_questions: Sequence[str] | None = None,
    ) -> dict[str, float]:
        s_embed = self.score_embedding(image, text, negative_texts=negative_texts)
        s_cap = self.score_caption_compare(image, text)
        s_grd = self.score_grounded(image, text)
        s_vqa = self.score_vqa_compliance(image, text, questions=vqa_questions)

        w = self.cfg.weights
        total = max(1e-8, w.alpha_embed + w.beta_caption + w.gamma_ground + w.delta_vqa)
        agg = (w.alpha_embed * s_embed + w.beta_caption * s_cap + w.gamma_ground * s_grd + w.delta_vqa * s_vqa) / total
        return {"embedding": s_embed, "caption_compare": s_cap, "grounded": s_grd, "vqa": s_vqa, "aggregate": agg}


# ==========================
# CLI (Click)
# ==========================
@click.command(context_settings={"show_default": True})
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to YAML config (models + weights)")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, dir_okay=False), help="CSV with image/text columns")
@click.option("--image-col", default="image", help="Image column name in CSV")
@click.option("--text-col", default="text", help="Text column name in CSV")
@click.option("--out", "out_path", default="", help="Output CSV path (default: print JSONL to stdout)")
def cli(config_path: str, csv_path: str, image_col: str, text_col: str, out_path: str) -> None:
    """Run text–image matching over a CSV of pairs."""
    try:
        matcher = TextImageMatcher.from_yaml(config_path)
    except ImportError as e:
        click.echo(f"Missing optional dependency while building models from YAML: {e}", err=True)
        click.echo(
            "Hint: install transformers, sentence-transformers, accelerate, timm, torch;\n"
            "or edit your YAML models to use local adapters.",
            err=True,
        )
        raise SystemExit(1)

    rows = matcher.read_csv(csv_path, image_col=image_col, text_col=text_col)

    vqa_qs = None
    negs = None

    results: list[dict[str, Any]] = []
    for r in rows:
        scores = matcher.score_all(r["image"], r["text"], negative_texts=negs, vqa_questions=vqa_qs)
        results.append({**r, **scores})

    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(results[0].keys()) if results else ["image", "text"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in results:
                w.writerow(row)
        click.echo(f"Wrote {len(results)} rows to {out_path}")
    else:
        for row in results:
            click.echo(_json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    cli()
