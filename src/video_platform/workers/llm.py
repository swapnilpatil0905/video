"""LLM worker (vLLM, fallback: transformers). Runtime: llm.

Request: {"hf_repo", "prompts": [{"id","system","user"}], "max_model_len", "max_tokens",
          "temperature", "tensor_parallel_size", "gpu_memory_utilization", "backend", "retries"}
Response: {"results": {id: parsed JSON | null}, "raw": {id: text for failures}, "stats": {...}}

A prompt may carry "split": [smaller prompts]. If its answer is cut off or unparseable, the
smaller prompts run in the next attempt with the already loaded model (no reload, no retry at
the same size). Without "split", a cut-off answer is salvaged at once (complete items kept)
instead of being retried with the same token budget, which would be cut off again.
"""

from __future__ import annotations

import sys

from .common import extract_json

_STRICT = "\n\nReturn ONLY valid JSON. No explanations, no markdown."


def _vllm_generate(request: dict):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=request["hf_repo"],
        max_model_len=request["max_model_len"],
        tensor_parallel_size=request["tensor_parallel_size"],
        gpu_memory_utilization=request["gpu_memory_utilization"],
    )
    params = SamplingParams(temperature=request["temperature"], max_tokens=request["max_tokens"])

    def generate(conversations):
        try:
            outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outputs = llm.chat(conversations, params)
        return [(o.outputs[0].text, o.outputs[0].finish_reason == "length") for o in outputs]

    return generate


def _transformers_generate(request: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(request["hf_repo"])
    model = AutoModelForCausalLM.from_pretrained(request["hf_repo"], torch_dtype="auto", device_map="auto")

    def generate(conversations):
        texts = []
        for conversation in conversations:
            try:
                prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=request["max_tokens"],
                                     do_sample=request["temperature"] > 0, temperature=max(request["temperature"], 1e-3))
            new = out[0][inputs["input_ids"].shape[1]:]
            texts.append((tokenizer.decode(new, skip_special_tokens=True), len(new) >= request["max_tokens"]))
        return texts

    return generate


def salvage_lines(text: str) -> dict | None:
    """Recover the complete {"id": ...} objects from a truncated JSON answer."""
    import json
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    decoder, found, pos = json.JSONDecoder(), [], 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and "id" in obj:
            found.append(obj)
        pos = end if isinstance(obj, dict) and "id" in obj else start + 1
    return {"lines": found} if found else None


def run(request: dict) -> dict:
    if request.get("backend", "vllm") == "vllm":
        try:
            generate = _vllm_generate(request)
        except ImportError:
            generate = _transformers_generate(request)
    else:
        generate = _transformers_generate(request)

    pending = list(request["prompts"])
    results: dict = {}
    raw: dict = {}
    stats = {"prompts": len(pending), "truncated": 0, "split": 0, "salvaged": 0, "failed": 0, "empty": 0}
    answers_log = request.get("answers_path")  # every raw answer, for debugging (jsonl)
    attempts = int(request.get("retries", 2)) + 1
    for attempt in range(attempts):
        if not pending:
            break
        conversations = [
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"] + (_STRICT if attempt else "")}]
            for p in pending
        ]
        answers = generate(conversations)
        still, parsed = [], 0
        last = attempt == attempts - 1
        if answers_log:
            import json

            with open(answers_log, "a", encoding="utf-8") as handle:
                for prompt, (text, truncated) in zip(pending, answers):
                    handle.write(json.dumps({"attempt": attempt + 1, "id": prompt["id"], "truncated": truncated,
                                             "chars": len(text), "text": text}, ensure_ascii=False) + "\n")
        for prompt, (text, _) in list(zip(pending, answers))[:2]:
            print(f"[llm] sample answer {prompt['id']}: {' '.join(text.split())[:300]}", file=sys.stderr, flush=True)
        for prompt, (text, truncated) in zip(pending, answers):
            stats["truncated"] += int(truncated)
            stats["empty"] += int(len(text.strip()) < 5)
            try:
                results[prompt["id"]] = extract_json(text)
                raw.pop(prompt["id"], None)
                parsed += 1
                continue
            except ValueError:
                pass
            salvaged = salvage_lines(text) if (truncated or last) else None
            if salvaged:
                results[prompt["id"]] = salvaged
                stats["salvaged"] += 1
            if prompt.get("split") and not last:
                still.extend(prompt["split"])  # smaller prompts, same loaded model
                stats["split"] += 1
            elif truncated and salvaged:
                pass  # retrying at the same size would be cut off again
            elif not last:
                raw[prompt["id"]] = text[-2000:]
                still.append(prompt)
            else:
                raw[prompt["id"]] = text[-2000:]
        print(f"[llm] attempt {attempt + 1}/{attempts}: {parsed}/{len(pending)} parsed, "
              f"{len(still)} to retry (truncated so far {stats['truncated']}, split {stats['split']}, "
              f"salvaged {stats['salvaged']})", file=sys.stderr, flush=True)
        pending = still
    for prompt in pending:
        results.setdefault(prompt["id"], None)
    stats["failed"] = sum(v is None for v in results.values())
    return {"results": results, "raw": raw, "stats": stats}
