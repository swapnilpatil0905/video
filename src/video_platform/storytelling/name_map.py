"""SourceNameMap: one fixed Indian name for every name the original dialogue says.

Scene analysis lists the names each scene's original lines use ("names_mentioned"). Cast characters keep their
registry name. Anyone else (people only talked about, nicknames the analyzer could not link) gets an Indian name
from one model call. Prompts then show the original lines with the Indian name already substituted, plus the
map itself, so the writer and the reviewers never have to invent or transliterate a name.
"""

from __future__ import annotations

import re
from typing import Any

from .names import name_distance

MIN_SOURCE_CHARS = 2  # single CJK characters are too ambiguous to substitute inside running text


def _clean(value: Any) -> str:
    return str(value or "").strip()


class SourceNameMap:
    def __init__(self, scenes: list[dict], registry: dict[str, dict]) -> None:
        self.registry = registry
        self.entries: dict[str, dict[str, Any]] = {}  # key: source (or romanized when no source)
        for scene in scenes:
            for n in scene.get("names_mentioned") or []:
                source, roman = _clean(n.get("source")), _clean(n.get("romanized"))
                key = source or roman
                if not key or (source and len(source) < MIN_SOURCE_CHARS and not roman):
                    continue
                entry = self.entries.setdefault(key, {"source": source, "romanized": roman, "character_id": None,
                                                      "scenes": [], "target": None, "devanagari": None, "origin": None})
                entry["romanized"] = entry["romanized"] or roman
                if scene.get("scene_id") and scene["scene_id"] not in entry["scenes"]:
                    entry["scenes"].append(scene["scene_id"])
                cid = n.get("character_id")
                if cid in registry and not entry["character_id"]:
                    entry["character_id"] = cid
        for entry in self.entries.values():
            c = registry.get(entry["character_id"] or "")
            if c:
                entry.update(target=c.get("display_name") or c.get("given_name"), devanagari=c.get("devanagari"),
                             origin="registry")

    # ---- names nobody on the cast list owns ----------------------------------------------------------

    def unresolved(self) -> list[dict[str, Any]]:
        return [e for e in self.entries.values() if not e["target"] and e["origin"] != "not_a_name"]

    def prompt(self, region: str, naming: dict[str, Any], examples: dict[str, str]) -> dict[str, str] | None:
        """One prompt resolving every name the cast registry does not already own. `examples`: key → a line using it."""
        todo = self.unresolved()
        if not todo:
            return None
        cast = "\n".join(
            f"- {cid}: {c.get('display_name') or c.get('given_name')} ({c.get('gender') or '?'}"
            f"{', ' + str(c.get('profession')) if c.get('profession') else ''}) {str(c.get('background') or '')[:120]}"
            for cid, c in sorted(self.registry.items()))
        rows = "\n".join(f"- {e['source'] or e['romanized']} ({e['romanized'] or '?'}): \"{examples.get(e['source'] or e['romanized'], '')[:120]}\""
                         for e in todo)
        return {"id": "source_names", "system": (
            "You adapt a foreign drama for Indian audiences. You decide what each name in the original dialogue refers "
            "to and give it one Indian replacement. Never transliterate the original, never turn a person into a place "
            "or a description. The list comes from speech recognition, so expect misspellings of the same name. "
            "Answer in JSON only."), "user": (
            f"Region: {region}. Naming conventions: {naming.get('conventions', '')}\n"
            f"The cast (already renamed; reuse these exact names when an entry refers to one of them):\n{cast or '(none)'}\n"
            "Entries from the original dialogue, with a line that uses each one:\n" + rows +
            "\n\nFor every entry choose kind:\n"
            '- "cast": it is one of the cast above (full name, nickname, "Sister X", "Mr X", misspelling) -> cast_id\n'
            '- "person": someone else -> a new Indian name; every spelling/nickname of the SAME person gets the SAME name\n'
            '- "group": a company, family or organisation -> an Indian equivalent keeping the kind (e.g. "Malhotra Group")\n'
            '- "not_a_name": titles, kinship or job words (mom, sister, boss, doctor, richest man), numbers, places, '
            "noise -> no replacement\n"
            'Return JSON: {"names": [{"source": str (exactly as listed), "kind": "cast"|"person"|"group"|"not_a_name", '
            '"cast_id": str|null, "name": str|null (roman letters, the form people say), "devanagari": str|null}]}'
        )}

    def accept(self, answer: Any, min_distance: float = 0.34) -> list[str]:
        """Apply the model's decisions; returns problems for entries it skipped (those keep no substitution)."""
        items = answer.get("names") if isinstance(answer, dict) else (answer if isinstance(answer, list) else [])
        cast_names = {_clean(c.get("display_name") or c.get("given_name")): cid for cid, c in self.registry.items()}
        cast_names.pop("", None)
        problems = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            key, name = _clean(item.get("source")), _clean(item.get("name"))
            kind = _clean(item.get("kind")) or "person"
            entry = self.entries.get(key)
            if entry is None or entry["target"]:
                continue
            if kind == "not_a_name":
                entry["origin"] = "not_a_name"
                continue
            cid = _clean(item.get("cast_id"))
            if kind == "cast" or cid in self.registry:
                c = self.registry.get(cid) or self.registry.get(cast_names.get(name, ""))
                if not c:
                    problems.append(f"{key}: unknown cast member {cid or name!r}")
                    continue
                entry.update(character_id=c["character_id"] if "character_id" in c else cid,
                             target=c.get("display_name") or c.get("given_name"), devanagari=c.get("devanagari"),
                             origin="registry")
                continue
            if not name:
                problems.append(f"{key}: no name given")
                continue
            original = entry["romanized"] or entry["source"]
            if name_distance(name, original) < min_distance:
                problems.append(f"{key}: '{name}' is too close to the original name")
                continue
            if any(name_distance(name, n) < 0.05 for n in cast_names):  # a new person must not share a cast name
                problems.append(f"{key}: '{name}' belongs to the cast")
                continue
            entry.update(target=name, devanagari=_clean(item.get("devanagari")) or None,
                         origin="group" if kind == "group" else "generated")
        return problems

    # ---- using the map ---------------------------------------------------------------------------------

    def _resolved(self) -> list[dict[str, Any]]:
        return sorted((e for e in self.entries.values() if e["target"]),
                      key=lambda e: -max(len(e["source"]), len(e["romanized"])))

    def apply(self, text: str) -> str:
        """The original line with every known name replaced by its Indian name (longest names first)."""
        out = text or ""
        for e in self._resolved():
            if e["source"] and len(e["source"]) >= MIN_SOURCE_CHARS:
                out = out.replace(e["source"], e["target"])
            if e["romanized"] and len(e["romanized"]) >= 3:
                out = re.sub(rf"\b{re.escape(e['romanized'])}\b", e["target"], out, flags=re.I)
        return out

    def note(self, texts: list[str]) -> str:
        """Prompt lines for the names these original lines say (empty when they say none)."""
        joined = " ".join(texts)
        rows = []
        for e in self._resolved():
            said = (e["source"] and e["source"] in joined) or (
                e["romanized"] and re.search(rf"\b{re.escape(e['romanized'])}\b", joined, re.I))
            if said:
                rows.append(f"{e['source'] or e['romanized']} ({e['romanized']}) → {e['target']}"
                            + (f" ({e['devanagari']})" if e["devanagari"] else ""))
        if not rows:
            return ""
        return ("Names in these lines — always use the Indian name, never the original or a transliteration: "
                + "; ".join(rows))

    def old_names(self) -> list[str]:
        return sorted({e["romanized"] for e in self.entries.values() if e["romanized"] and e["origin"] != "not_a_name"})

    def report(self) -> list[dict[str, Any]]:
        return [{k: e[k] for k in ("source", "romanized", "character_id", "target", "devanagari", "origin", "scenes")}
                for e in self.entries.values()]
