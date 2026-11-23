from pathlib import Path

import pytest

from video_platform.storytelling import (
    CreativeReferenceManager, DialogueDurationOptimizer, EmotionAndDeliveryPlanner, IndianNameGenerator,
    LocationDatabase, ModelSelector, NaturalConversationGenerator, RegionalEnvironmentAdapter,
    RegionalLanguageProfileManager, ScreenplayEnhancementEngine, ScreenplayQualityEvaluator,
)
from video_platform.storytelling.context import DialogueContextAnalyzer, group_scenes
from video_platform.storytelling.duration import syllables

DIALECT = {"name": "dakhni_urdu", "intensity": 0.65, "english_mixing": 0.45, "urdu_influence": 0.7}


@pytest.fixture(scope="module")
def hyd():
    return RegionalLanguageProfileManager.load("hyderabad", DIALECT, "natural_hinglish")


def test_every_profile_loads_and_zones_are_consistent():
    root = Path(__file__).resolve().parents[2] / "configs" / "regional_profiles"
    for path in root.glob("*.yaml"):
        pm = RegionalLanguageProfileManager.load(path.stem, DIALECT, "natural_hindi")
        db = LocationDatabase(pm.profile)
        zones = set(pm.profile.get("zones") or {}) | {"any", "ring", "far"}
        assert all(p["zone"] in zones for p in db.places), path.name
        assert pm.dialect_info(), path.name


def test_location_relations_and_line_checks(hyd):
    db = LocationDatabase(hyd.profile)
    assert db.relation("Gachibowli", "HITEC City") == "near"
    assert db.relation("Banjara Hills", "Gachibowli") == "adjacent"
    assert db.relation("Charminar", "Gachibowli") == "far"
    env = RegionalEnvironmentAdapter(db)
    plan = env.validate_plan({"setting": "office", "real_place": "HITEC City", "destinations": ["Old City", "Fake Mall"],
                              "place_references_allowed": True})
    assert plan["real_place"] == "HITEC City" and plan["destinations"] == ["Old City"] and "Fake Mall" in plan["rejected"]
    assert env.check_line("Tumhara office HITEC City mein hai na?", plan) == []
    assert any("unplanned" in i for i in env.check_line("Charminar chalein?", plan))
    issues = env.check_line("Gachibowli Old City ke paas hai, 10 minute lagenge", {**plan, "destinations": ["Old City", "Gachibowli"]})
    assert any("not near" in i for i in issues) and any("travel time" in i for i in issues)
    no_places = env.validate_plan({"setting": "a kitchen", "place_references_allowed": True})
    assert no_places["place_references_allowed"] is False


def test_character_profiles_and_slang(hyd):
    p = hyd.normalize_character_profile({"accent": "Hyderabadi", "dialect_intensity": 0.9, "english_mixing": "low"})
    assert p["accent"] == "hyderabadi" and p["english_mixing"] == 0.15
    assert "Dakhni" in hyd.voice_accent(p) or "Hyderabadi" in hyd.voice_accent(p)
    assert hyd.slang_density("अरे मियाँ, नक्को यार") >= 2
    assert hyd.slang_density("चलो, पक्का?") == 0  # pan-Indian words are not dialect markers


def test_syllables_and_duration():
    assert syllables("नमस्ते") == 3
    assert syllables("क्या") == 1
    opt = DialogueDurationOptimizer(5.0, 1.12, 0.35)
    assert opt.check("हाँ", 3.0)["verdict"] == "too_short"
    assert opt.check("मैं तुमसे पूछ रहा हूँ, यहाँ आए क्यों?", 1.0)["verdict"] == "too_long"
    assert opt.check("तुम यहाँ क्या कर रहे हो?", 2.2)["verdict"] == "ok"
    assert opt.budget(2.0) == {"target": 9, "max": 11}


def test_names_validation():
    gen = IndianNameGenerator({}, "Hyderabad", min_distance=0.34)
    reg = [{"character_id": "A", "given_name": "Arjun", "family_name": "Reddy"},
           {"character_id": "B", "given_name": "Arjoon", "family_name": "Rao"},
           {"character_id": "C", "given_name": "Meera", "family_name": "Sharma"}]
    issues = gen.validate(reg, [{"a": "A", "b": "C", "relation": "elder brother"}])
    assert any("too similar" in i for i in issues)
    assert any("family names differ" in i for i in issues)
    assert not gen.validate(reg[2:], [])
    # marriage does not require a shared family name
    assert not [i for i in gen.validate(reg[::2], [{"a": "A", "b": "C", "relation": "wife"}]) if "family" in i]


def test_enhancement_never_adds_comedy_to_tragedy():
    eng = ScreenplayEnhancementEngine({"mode": "creative_adaptation", "enhance_comedy": True, "enhance_tragedy": True,
                                       "enhance_emotions": True, "avoid_forced_humour": True})
    sad = eng.plan({"mood": "tragedy", "intensity": 0.9, "enhancement": {"comedy": True}})
    assert "comedy" not in sad["allowed"] and "tragedy" in sad["allowed"]
    light = eng.plan({"mood": "neutral", "intensity": 0.3, "enhancement": {"comedy": True}})
    assert "comedy" in light["allowed"]
    forced = eng.plan({"mood": "neutral", "intensity": 0.3, "enhancement": {}})
    assert "comedy" not in forced["allowed"]


def test_delivery_mapping():
    planner = EmotionAndDeliveryPlanner({"max_pause_seconds": 0.8, "conversational_pauses": True})
    d = planner.normalize({"emotion": "angry but controlled", "intensity": 0.85, "delivery": "controlled_aggressive",
                           "speaking_rate": 1.05, "pause_before": 0.4, "voice_break": True}, character_id="X", accent="hyderabadi")
    assert d["emotion"] == "angry" and d["pause_before"] == 0.4
    c = planner.tts_controls(d, base_cfg=0.5)
    assert 0.6 < c["exaggeration"] < 1.0 and c["cfg_weight"] < 0.5 and "voice_break" in c["unsupported"]
    timed = planner.apply_timing({"out_start": 10.0, "available_s": 3.0}, d)
    assert timed["out_start"] > 10.0 and timed["available_s"] < 3.0


def test_scene_grouping_and_analysis_normalize():
    segs = [{"segment_id": f"S{i}", "start_s": t, "end_s": t + 1.0} for i, t in enumerate([0, 2, 4, 20, 22])]
    scenes = group_scenes(segs, {s["segment_id"]: "c0" for s in segs}, gap_s=5.0, max_lines=10)
    assert [len(s["segment_ids"]) for s in scenes] == [3, 2]
    norm = DialogueContextAnalyzer.normalize({"mood": "Romantic comedy", "intensity": 3}, scenes[0])
    assert norm["mood"] in ("comedy", "romance") and norm["intensity"] == 1.0


def test_originality_and_provenance(tmp_path):
    (tmp_path / "ref.txt").write_text("this is a very famous line from a film nobody should copy", encoding="utf-8")
    refs = CreativeReferenceManager({"max_ngram_overlap": 5, "sources": [
        {"title": "Reference", "kind": "licensed_screenplay", "licence": "personal copy", "path": "ref.txt"}]}, tmp_path)
    assert refs.originality_issues("he said: a very famous line from a film again")
    assert not refs.originality_issues("an entirely original sentence")
    assert refs.provenance()[0]["used_for"] == "originality check only"


def test_quality_line_issues(hyd):
    db = LocationDatabase(hyd.profile)
    ev = ScreenplayQualityEvaluator(min_score=70, duration=DialogueDurationOptimizer(), environment=RegionalEnvironmentAdapter(db),
                                    references=CreativeReferenceManager({}, Path(".")), profiles=hyd,
                                    conversation=NaturalConversationGenerator(), old_names=["Lin Xiao"])
    line = {"target_text": "अरे मियाँ, नक्को, काइकू, मेरेकू नहीं चाहिए", "roman_text": "Lin Xiao ko bolo",
            "target_duration_s": 3.0, "direction": {}}
    flags = {i["flag"] for i in ev.line_issues(line, {"dialect_intensity": 0.1}, {"place_references_allowed": False},
                                               ["Arjun"])}
    assert {"excessive_slang", "incorrect_name"} <= flags
    assert ev.line_needs_rewrite({"issues": [{"flag": "too_long"}]})
    assert not ev.line_needs_rewrite({"issues": [], "review": {"natural": 40, "applied": True}})
    # grammatical gender: a woman speaking with the masculine form must be rewritten
    female = {"target_text": "मैं घर जा रहा हूँ।", "target_duration_s": 3.0, "direction": {},
              "speaker_gender": "female", "listener_gender": "male"}
    issues = ev.line_issues(female, None, None, ["Arjun"])
    assert "wrong_gender" in {i["flag"] for i in issues}
    assert ev.line_needs_rewrite({"issues": issues})
    female["target_text"] = "मैं घर जा रही हूँ।"
    assert "wrong_gender" not in {i["flag"] for i in ev.line_issues(female, None, None, ["Arjun"])}


def test_model_selector_prefers_what_fits():
    class Rec:
        def __init__(self, gb):
            self.hardware = type("H", (), {"min_vram_gb": gb})()
    reg = {"big": Rec(80), "small": Rec(24)}
    sel = ModelSelector({"writer": ["big", "small"]}, {"analyzer": "small"}, reg.__getitem__, gpu_memory_gb=48)
    assert sel.pick("writer") == "small" and sel.pick("analyzer") == "small"
    assert ModelSelector({"writer": ["big", "small"]}, {}, reg.__getitem__, 96).pick("writer") == "big"


def test_prompts_are_identical_across_processes():
    """Prompt text must not depend on hash randomisation (reuse of saved answers + vLLM prefix caching)."""
    import os
    import subprocess
    import sys

    code = ("from video_platform.storytelling import *;"
            "pm=RegionalLanguageProfileManager.load('hyderabad',{'name':'dakhni_urdu','intensity':0.6},'natural_hinglish');"
            "db=LocationDatabase(pm.profile);env=RegionalEnvironmentAdapter(db);"
            "print(db.for_prompt());print(env.plan_schema());print(pm.expressions_for_prompt(0.7));"
            "print(ScreenplayEnhancementEngine({'enhance_comedy':True,'enhance_emotions':True,'enhance_tragedy':True})"
            ".plan({'mood':'tragedy','intensity':0.8,'enhancement':{'emotion':True,'tragedy':True}}))")
    src = str(Path(__file__).resolve().parents[2] / "src")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": src, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2", "3")}
    assert len(outs) == 1 and next(iter(outs)).strip()


def test_source_name_map_maps_cast_and_talked_about_people():
    from video_platform.storytelling.name_map import SourceNameMap

    registry = {"SRC_CHAR_0001": {"display_name": "Zahra", "given_name": "Zahra", "devanagari": "ज़हरा"}}
    scenes = [{"scene_id": "SCN_0", "names_mentioned": [
        {"source": "陈戴", "romanized": "Chen Dai", "character_id": "SRC_CHAR_0001"},
        {"source": "许明", "romanized": "Xu Ming", "character_id": None},
        {"source": "夏", "romanized": "", "character_id": None}]}]  # one CJK character: too ambiguous
    m = SourceNameMap(scenes, registry)
    assert [e["source"] for e in m.unresolved()] == ["许明"]
    assert "许明" in m.prompt("Hyderabad", {}, {"许明": "有人想让许明死了，"})["user"]
    problems = m.accept({"names": [{"source": "许明", "name": "Xu Min"}]})  # too close to the original
    assert problems and m.unresolved()
    assert m.accept({"names": [{"source": "许明", "name": "Imran", "devanagari": "इमरान"}]}) == []
    assert m.apply("有人想让许明死了，陈戴知道") == "有人想让Imran死了，Zahra知道"
    note = m.note(["有人想让许明死了，"])
    assert "许明 (Xu Ming) → Imran (इमरान)" in note and "Zahra" not in note
    assert m.old_names() == ["Chen Dai", "Xu Ming"]


def test_name_score_counts_lines_not_repeated_flags():
    from video_platform.storytelling.quality import ScreenplayQualityEvaluator

    lines = [{"issues": [{"flag": "incorrect_name"}] * 3}, {"issues": []}]
    scores = ScreenplayQualityEvaluator.deterministic_scores(None, lines)
    assert scores["name_usage"] == 50


def test_source_name_map_aliases_cast_links_and_non_names():
    from video_platform.storytelling.name_map import SourceNameMap

    registry = {"SRC_CHAR_0004": {"character_id": "SRC_CHAR_0004", "display_name": "Zahra", "devanagari": "ज़हरा",
                                  "gender": "female"}}
    scenes = [{"scene_id": "SCN_0", "names_mentioned": [
        {"source": "夏夏", "romanized": "Xia Xia"}, {"source": "霞霞姐姐", "romanized": "Xiaxia jiejie"},
        {"source": "陈大", "romanized": "Chen Da"}, {"source": "陈败", "romanized": "Chen Bai"},
        {"source": "妈妈", "romanized": "Mama"}, {"source": "大夏集团", "romanized": "Daxia Group"}]}]
    m = SourceNameMap(scenes, registry)
    assert "SRC_CHAR_0004: Zahra (female)" in m.prompt("Hyderabad", {}, {})["user"]
    problems = m.accept({"names": [
        {"source": "夏夏", "kind": "cast", "cast_id": "SRC_CHAR_0004"},
        {"source": "霞霞姐姐", "kind": "cast", "cast_id": "SRC_CHAR_0004"},
        {"source": "陈大", "kind": "person", "name": "Chandrashekhar"},
        {"source": "陈败", "kind": "person", "name": "Chandrashekhar"},  # same person, misheard: same name is fine
        {"source": "妈妈", "kind": "not_a_name"},
        {"source": "大夏集团", "kind": "group", "name": "Malhotra Group"}]})
    assert problems == []
    assert m.apply("夏夏和霞霞姐姐见了陈大和陈败，妈妈在大夏集团") == "Zahra和Zahra见了Chandrashekhar和Chandrashekhar，妈妈在Malhotra Group"
    assert m.unresolved() == [] and "Mama" not in m.old_names()
    assert m.accept({"names": []}) == []
    m2 = SourceNameMap([{"scene_id": "S", "names_mentioned": [{"source": "许明", "romanized": "Xu Ming"}]}], registry)
    assert m2.accept({"names": [{"source": "许明", "kind": "person", "name": "Zahra"}]}) == ["许明: 'Zahra' belongs to the cast"]
