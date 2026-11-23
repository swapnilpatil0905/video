"""Grammatical gender in the adapted dialogue (Hindi / Hindustani).

Hindi verbs and adjectives agree with the speaker's gender in the first person ("मैं जा रहा हूँ" /
"मैं जा रही हूँ", "करूँगा" / "करूँगी") and with the listener's gender in the second person ("तुम आ रहे
हो" / "तुम आ रही हो"). A translated line easily keeps the wrong form, and a TTS voice of the right
gender saying it sounds wrong at once.

``agreement_issues`` finds forms that are unambiguous: first-person forms ending in हूँ, first-person
futures (-ऊँगा/-ऊँगी), a clause that starts with मैं and ends in था/थी, and तुम / तू / आप clauses
with a gendered participle. Anything ambiguous is left alone: no false alarms is more important
than catching everything. The writer also gets the rule and each line's speaker/listener genders.
"""

from __future__ import annotations

import re
from typing import Any

RULE = ("Grammatical gender is mandatory: verbs and adjectives agree with the SPEAKER's gender in the first person "
        "(male: मैं जा रहा हूँ, मैं थक गया, करूँगा; female: मैं जा रही हूँ, मैं थक गई, करूँगी) and with the LISTENER's "
        "gender in the second person (male: तुम आ रहे हो, आप बैठिए, क्या आप थक गए?; female: तुम आ रही हो, क्या आप "
        "थक गईं?). Each line lists the speaker's and listener's gender (M/F); never copy the gender of the "
        "original wording or of another speaker.")


def norm_gender(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith(("f", "w", "girl", "महिला", "स्त्री")):
        return "female"
    if text.startswith(("m", "boy", "पुरुष")):
        return "male"
    return None


def short(gender: Any) -> str:
    g = norm_gender(gender)
    return "F" if g == "female" else "M" if g == "male" else "?"


_WORD = r"[ऀ-ॿ‌‍]+"
_HUN = r"(?:हूँ|हूं)"
_SPLIT = re.compile(r"[।?!.,;:\n—–-]+|\s(?:कि|और|लेकिन|पर|मगर|क्योंकि|जब|तो|अगर|जो)\s")
_OTHERS = {"वो", "वह", "ये", "यह", "वे", "उसने", "उसको", "उसे", "लोग", "हम", "सब"}  # another subject: ambiguous
# participles / adjectives with a masculine and a feminine form
_MASC = {"रहा", "गया", "सका", "चुका", "लगा", "आया", "हुआ", "बैठा", "खड़ा", "थका", "अकेला", "अच्छा", "पाया",
         "डरा", "मरा", "भूखा", "प्यासा", "बड़ा", "छोटा", "सच्चा", "बुरा"}
_FEM = {"रही", "गई", "गयी", "सकी", "चुकी", "लगी", "आई", "आयी", "हुई", "बैठी", "खड़ी", "थकी", "अकेली", "अच्छी",
        "पाई", "पायी", "डरी", "मरी", "भूखी", "प्यासी", "बड़ी", "छोटी", "सच्ची", "बुरी"}
_NOT_PARTICIPLE = ("बत्ती", "गलती", "ग़लती", "धरती", "रोती", "पोती", "मोती", "ज्योति", "छाती")
# second person: pronoun → (auxiliaries, masculine participles, feminine participles, future endings m/f)
_SECOND = {
    "तुम": ({"हो", "थे", "थी"}, ("गे", "गी"), "(?:ो|ओ)"),
    "आप": ({"हैं", "थे", "थीं"}, ("गे", "गी"), "(?:ें|ें|ेँ|एँ|एं)"),
    "तू": ({"है", "था", "थी"}, ("गा", "गी"), "(?:े|ए)"),
}
_PL_MASC = {"रहे", "गए", "गये", "सके", "चुके", "लगे", "आए", "आये", "हुए", "बैठे", "थके"}
_PL_FEM = {"रही", "गई", "गयी", "गईं", "सकी", "चुकी", "लगी", "आई", "आईं", "हुई", "हुईं", "बैठी", "थकी"}


def _gender_of(word: str) -> str | None:
    """Gender carried by a participle/adjective; habitual -ता/-ती counts, nouns are not guessed."""
    if word in _MASC:
        return "male"
    if word in _FEM:
        return "female"
    if len(word) > 2 and word.endswith("ता"):
        return "male"
    if len(word) > 2 and word.endswith("ती") and not word.endswith(_NOT_PARTICIPLE):
        return "female"
    return None


def _clauses(text: str) -> list[list[str]]:
    return [c.split() for c in _SPLIT.split(text or "") if c.strip()]


def _first_person(text: str) -> list[tuple[str, str]]:
    """(gender, evidence) for first-person singular forms that show a gender."""
    found = []
    for m in re.finditer(rf"({_WORD})\s+{_HUN}", text):  # "... रहा हूँ", "... जाती हूँ", "थकी हूँ" (हूँ = 1st person)
        g = _gender_of(m.group(1))
        if g:
            found.append((g, m.group(0)))
    for m in re.finditer(r"[ऀ-ॿ]*(?:ऊँ|ऊं|ूँ|ूं)(गा|गी)(?![ऀ-ॿ])", text):  # करूँगा / जाऊँगी (1st person only)
        found.append(("male" if m.group(1) == "गा" else "female", m.group(0)))
    for words in _clauses(text):
        # "मैं वहाँ था", "मैं डर गई थी"; मैंने is ergative (the verb agrees with the object) and does not match
        if len(words) < 2 or words[0] != "मैं" or words[-1] not in ("था", "थी") or _OTHERS & set(words[1:]):
            continue
        g = "male" if words[-1] == "था" else "female"
        prev = _gender_of(words[-2])
        if prev is None or prev == g:
            found.append((g, " ".join(words[-3:])))
    return found


def _second_person(text: str) -> list[tuple[str, str]]:
    """(gender of the addressee, evidence) for तुम/तू/आप clauses with a gendered participle."""
    found = []
    for words in _clauses(text):
        pronouns = [w for w in words if w in _SECOND]
        if len(set(pronouns)) != 1 or _OTHERS & set(words) or "मैं" in words:
            continue
        aux, future, stem = _SECOND[pronouns[0]]
        for i, w in enumerate(words[1:], 1):
            prev = words[i - 1]
            if w in aux:
                if pronouns[0] == "तू":
                    g = _gender_of(prev)
                elif prev in _PL_MASC or (len(prev) > 2 and prev.endswith("ते")):
                    g = "male"
                elif prev in _PL_FEM or (len(prev) > 2 and prev.endswith(("ती", "तीं"))):
                    g = "female"
                else:
                    g = None
                if g:
                    found.append((g, f"{prev} {w}"))
            m = re.fullmatch(rf"[ऀ-ॿ]+{stem}({future[0]}|{future[1]})[ंँ]?[?।!]?", w)  # करोगे/करोगी, करेंगे/करेंगी
            if m:
                found.append(("male" if m.group(1) == future[0] else "female", w))
    return found


def agreement_issues(text: str, speaker_gender: Any, listener_gender: Any = None) -> list[str]:
    """Forms in a line whose grammatical gender contradicts the speaker's (first person) or listener's (second)."""
    issues = []
    speaker, listener = norm_gender(speaker_gender), norm_gender(listener_gender)
    if speaker:
        for g, evidence in _first_person(text or ""):
            if g != speaker:
                issues.append(f"speaker is {speaker} but says '{evidence}' ({g} form)")
    if listener:
        for g, evidence in _second_person(text or ""):
            if g != listener:
                issues.append(f"listener is {listener} but is addressed with '{evidence}' ({g} form)")
    return list(dict.fromkeys(issues))


_WORDS = {"male": re.compile(r"\b(male|man|men|boy|gentleman|he|his|him)\b", re.I),
          "female": re.compile(r"\b(female|woman|women|girl|lady|she|her|hers)\b", re.I)}


def mentions_other_gender(text: str, gender: Any) -> bool:
    """True when an English description (image prompt, voice description) names the other gender only."""
    g = norm_gender(gender)
    if not g or not text:
        return False
    other = "female" if g == "male" else "male"
    return bool(_WORDS[other].search(text)) and not _WORDS[g].search(text)
