from video_platform.workers import llm as worker


def _fake(responses):
    calls = []

    def factory(_request):
        def generate(conversations):
            calls.append(len(conversations))
            return [responses(c[1]["content"], len(calls)) for c in conversations]
        return generate
    return factory, calls


def test_truncated_answer_is_split_not_retried(monkeypatch):
    def responses(user, call):
        if user.startswith("BIG"):
            return '{"lines": [{"id": "a", "text": "x"}, {"id": "b", "te', True
        if user.startswith("BAD") and call == 1:
            return "not json", False
        return '{"lines": [{"id": "%s", "text": "ok"}]}' % user[:3], False

    factory, calls = _fake(responses)
    monkeypatch.setattr(worker, "_vllm_generate", factory)
    out = worker.run({"retries": 2, "prompts": [
        {"id": "p0", "system": "", "user": "BIG", "split": [{"id": "p0_s0", "system": "", "user": "s00"},
                                                          {"id": "p0_s1", "system": "", "user": "s01"}]},
        {"id": "p1", "system": "", "user": "BIG"},
        {"id": "p2", "system": "", "user": "BAD"},
    ]})
    assert calls == [3, 3]  # split pieces + one strict retry in a single second pass
    assert out["results"]["p1"] == {"lines": [{"id": "a", "text": "x"}]}  # salvaged, not retried
    assert out["results"]["p0_s0"] and out["results"]["p0_s1"] and out["results"]["p2"]
    assert out["stats"]["truncated"] == 2 and out["stats"]["split"] == 1 and out["stats"]["failed"] == 0


def test_salvage_lines_keeps_complete_objects():
    text = '{"lines": [{"id": "S1", "text": "नमस्ते"}, {"id": "S2", "text": "चलो"}, {"id": "S3", "te'
    assert [x["id"] for x in worker.salvage_lines(text)["lines"]] == ["S1", "S2"]
    assert worker.salvage_lines("nothing here") is None
