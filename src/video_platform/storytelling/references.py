"""CreativeReferenceManager: storytelling techniques only — never copied expression.

- Keeps a provenance record of every configured reference source (title, kind, licence, purpose).
- Gives the writer a technique guide written for this project (no third-party text is ever put in a prompt).
- Originality check: generated lines are compared with any local reference texts the user has the
  rights to keep (word n-gram overlap); matching lines are flagged and rewritten.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

TECHNIQUES = (
    "Scene construction: enter late, leave early; every scene turns (something changes).",
    "Escalation: each exchange raises the stakes a little; the strongest line comes last.",
    "Comedy timing: set-up, pause, pay-off; the reaction is often funnier than the joke; rule of three.",
    "Emotional pacing: understatement before release; let silence do the work in grief.",
    "Suspense: the audience knows more than the character, or less; delay the answer.",
    "Dialogue rhythm: alternate short and long lines; subtext over stated feelings.",
    "Character voice: each person has their own vocabulary, rhythm and what they avoid saying.",
    "Cliffhanger: end on a question, a reveal or a decision — not on a summary.",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[\wऀ-ॿ]+", (text or "").lower())


class CreativeReferenceManager:
    def __init__(self, spec: dict[str, Any], project_dir: Path) -> None:
        self.spec = spec
        self.project_dir = project_dir
        self.n = int(spec.get("max_ngram_overlap", 6))
        self._grams: set[tuple[str, ...]] | None = None

    def rule(self) -> str:
        return ("All dialogue must be original: never quote or closely paraphrase famous film or TV dialogue, songs, "
                "catchphrases or recognizable scenes; use only general storytelling technique.")

    def technique_guide(self) -> str:
        if not (self.spec.get("enabled", True) and self.spec.get("learn_storytelling_techniques", True)):
            return ""
        return "Storytelling techniques to apply (general craft, not content):\n- " + "\n- ".join(TECHNIQUES)

    def provenance(self) -> list[dict[str, Any]]:
        out = []
        for src in self.spec.get("sources") or []:
            entry = {k: src.get(k) for k in ("title", "kind", "licence", "notes")}
            entry["purpose"] = "general storytelling techniques; not copied, not sent to any model"
            path = self._path(src)
            if path and path.exists():
                entry["local_text_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                entry["used_for"] = "originality check only"
            out.append(entry)
        out.append({"title": "built-in technique guide", "kind": "original_example", "licence": "project-authored",
                    "purpose": "craft guidance in the writer prompt"})
        return out

    def _path(self, src: dict) -> Path | None:
        if not src.get("path"):
            return None
        p = Path(src["path"])
        return p if p.is_absolute() else self.project_dir / p

    def _reference_grams(self) -> set[tuple[str, ...]]:
        if self._grams is None:
            grams: set[tuple[str, ...]] = set()
            for src in self.spec.get("sources") or []:
                path = self._path(src)
                if path and path.exists():
                    words = _words(path.read_text(encoding="utf-8", errors="ignore"))
                    grams.update(tuple(words[i:i + self.n]) for i in range(len(words) - self.n + 1))
            self._grams = grams
        return self._grams

    def originality_issues(self, *texts: str) -> list[str]:
        grams = self._reference_grams()
        if not grams:
            return []
        for text in texts:
            words = _words(text)
            for i in range(len(words) - self.n + 1):
                if tuple(words[i:i + self.n]) in grams:
                    return [f"originality: shares {self.n}+ consecutive words with a reference text "
                            f"(\"{' '.join(words[i:i + self.n])}\")"]
        return []
