from __future__ import annotations
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class QGBackend(Protocol):
    def generate(self, text: str, limit: int = 12) -> list[str]: ...

class T5QGAdapter:
    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    @staticmethod
    def _filter_yesno(lines: list[str]) -> list[str]:
        starters = ("is ","are ","do ","does ","did ","has ","have ","was ","were ","can ")
        out = []
        for ln in lines:
            s = ln.strip().rstrip("?")
            if not s: continue
            q = s if s.endswith("?") else s + "?"
            if q.lower().startswith(starters):
                out.append(q)
        return list(dict.fromkeys(out))

    def generate(self, text: str, limit: int = 12) -> list[str]:
        prompt = ("Generate concise yes/no questions that verify visual content described. "
                  "Cover objects, counts, attributes, relations; avoid style-only terms.\n"
                  f"Text: {text}\nQuestions (one per line):\n")
        out = self.pipe(prompt, max_new_tokens=128)
        item = out[0] if isinstance(out, list) and out else out
        text_out = item.get("generated_text") or item.get("text") if isinstance(item, dict) else (item or "")
        lines = [ln for ln in str(text_out).splitlines() if ln.strip()]
        qs = self._filter_yesno(lines)
        return qs[:limit] if qs else generate_vqa_questions(text, limit)

