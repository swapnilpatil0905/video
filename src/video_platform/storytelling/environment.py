"""RegionalEnvironmentAdapter + LocationDatabase: real places, used only where the story puts the characters.

Geography is expressed only through zones ("same zone" = near, "adjacent", "far"). The database
never contains travel times, so the writer is told not to state any.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

NEAR_WORDS = ("paas", "पास", "bagal", "बगल", "nazdeek", "नज़दीक", "नजदीक", "saamne", "सामने", "next to", "near")
TIME_PATTERN = re.compile(r"(\d+|दस|बीस|पंद्रह|आधा|एक|दो)\s*(min|minute|मिनट|ghante|घंटे|घंटा|hour)", re.I)


class LocationDatabase:
    def __init__(self, profile: dict[str, Any]) -> None:
        self.places: list[dict] = [p for p in profile.get("locations") or [] if p.get("name")]
        self.zones: dict[str, str] = profile.get("zones") or {}
        self.adjacent = {frozenset(pair) for pair in profile.get("adjacent_zones") or [] if len(pair) == 2}
        self.rules: list[str] = profile.get("rules") or []
        self._by_key = {}
        for place in self.places:
            for key in (place["name"], place.get("devanagari")):
                if key:
                    self._by_key[key.lower()] = place

    def __bool__(self) -> bool:
        return bool(self.places)

    def get(self, name: str) -> dict | None:
        return self._by_key.get((name or "").strip().lower())

    def relation(self, a: str, b: str) -> str:
        pa, pb = self.get(a), self.get(b)
        if not pa or not pb:
            return "unknown"
        za, zb = pa.get("zone"), pb.get("zone")
        if "any" in (za, zb) or "ring" in (za, zb):
            return "unknown"
        if za == zb:
            return "near"
        return "adjacent" if frozenset((za, zb)) in self.adjacent else "far"

    def mentioned(self, text: str) -> list[dict]:
        """Real places named in a line (either script). Generic lifestyle entries are skipped."""
        low = (text or "").lower()
        found = []
        for place in self.places:
            if place.get("name", "").startswith(("a ", "an ")):
                continue
            for key in (place["name"], place.get("devanagari")):
                if key and len(key) >= 3 and key.lower() in low:
                    found.append(place)
                    break
        # drop places contained in a longer match (e.g. "Ameerpet" inside "Ameerpet Metro Station")
        names = [p["name"] for p in found]
        return [p for p in found if not any(p["name"] != n and p["name"] in n for n in names)]

    def for_prompt(self, tags: list[str] | None = None, limit: int = 40) -> str:
        """Compact place list grouped by zone (optionally the places whose tags overlap)."""
        chosen = self.places
        if tags:
            wanted = set(tags)
            ranked = sorted(self.places, key=lambda p: -len(wanted & set(p.get("tags") or [])))
            chosen = ranked[:limit]
        by_zone: dict[str, list[str]] = {}
        for p in chosen[:limit]:
            by_zone.setdefault(p.get("zone", "?"), []).append(f"{p['name']} ({p.get('kind')})")
        zones = "; ".join(f"{z}: {', '.join(v)}" for z, v in by_zone.items())
        adjacency = ", ".join(f"{a}~{b}" for a, b in (tuple(sorted(x)) for x in self.adjacent))
        return f"Places by zone — {zones}\nAdjacent zones: {adjacency or 'n/a'} (any other pair is far)"


class RegionalEnvironmentAdapter:
    """Decides which real places a scene may mention, from the scene's own location plan."""

    def __init__(self, db: LocationDatabase, enabled: bool = True) -> None:
        self.db = db
        self.enabled = enabled and bool(db)

    def plan_schema(self) -> str:
        return ('"location_plan": {"setting": str (where the scene happens, as the story shows it), '
                '"real_place": str|null (a place from the list ONLY if the setting matches it), '
                '"fictional_places": [str] (invented shops/buildings/addresses, always fictional), '
                '"destinations": [str] (real places from the list the characters talk about going to, if any), '
                '"place_references_allowed": bool (true only if the story itself calls for naming a place)}')

    def validate_plan(self, plan: Any) -> dict[str, Any]:
        """Keep only real places that exist in the database; everything else becomes fictional."""
        plan = plan if isinstance(plan, dict) else {}
        out = {"setting": str(plan.get("setting") or "")[:160], "real_place": None, "destinations": [],
               "fictional_places": [str(x)[:80] for x in plan.get("fictional_places") or []][:5],
               "place_references_allowed": bool(plan.get("place_references_allowed")) and self.enabled,
               "rejected": []}
        for key in ("real_place",):
            value = plan.get(key)
            if value and self.db.get(str(value)):
                out[key] = self.db.get(str(value))["name"]
            elif value:
                out["rejected"].append(str(value)[:80])
        for value in plan.get("destinations") or []:
            place = self.db.get(str(value))
            if place:
                out["destinations"].append(place["name"])
            else:
                out["rejected"].append(str(value)[:80])
        if not (out["real_place"] or out["destinations"]):
            out["place_references_allowed"] = False
        return out

    def allowed_places(self, plan: dict[str, Any]) -> list[str]:
        if not plan.get("place_references_allowed"):
            return []
        return [p for p in [plan.get("real_place"), *plan.get("destinations", [])] if p]

    def check_line(self, text: str, plan: dict[str, Any]) -> list[str]:
        """Geography issues in one line: unplanned places, 'near' across zones, stated travel times."""
        issues = []
        mentioned = self.db.mentioned(text)
        allowed = set(self.allowed_places(plan))
        for place in mentioned:
            if place["name"] not in allowed:
                issues.append(f"unplanned place reference: {place['name']} (not part of this scene's location plan)")
        low = (text or "").lower()
        if len(mentioned) >= 2 and any(w in low for w in NEAR_WORDS):
            for a, b in combinations(mentioned, 2):
                if self.db.relation(a["name"], b["name"]) in ("far", "adjacent"):
                    issues.append(f"geography: {a['name']} and {b['name']} are not near each other")
        if mentioned and TIME_PATTERN.search(text or ""):
            issues.append("geography: states a specific travel time (not allowed)")
        return issues
