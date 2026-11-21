"""LLM worker (vLLM, fallback: transformers). Runtime: llm.

Request: {"hf_repo", "prompts": [{"id","system","user"}], "max_model_len", "max_tokens",
          "temperature", "tensor_parallel_size", "gpu_memory_utilization", "backend", "retries"}
Response: {"results": {id: parsed JSON | null}, "raw": {id: text for failures}}
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
        return [o.outputs[0].text for o in outputs]

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
            texts.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
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
    attempts = int(request.get("retries", 2)) + 1
    for attempt in range(attempts):
        if not pending:
            break
        conversations = [
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"] + (_STRICT if attempt else "")}]
            for p in pending
        ]
        texts = generate(conversations)
        still = []
        for prompt, text in zip(pending, texts):
            try:
                results[prompt["id"]] = extract_json(text)
                raw.pop(prompt["id"], None)
            except ValueError:
                salvaged = salvage_lines(text) if attempt == attempts - 1 else None
                if salvaged:
                    results[prompt["id"]] = salvaged
                    print(f"[llm] {prompt['id']}: answer cut off, kept {len(salvaged['lines'])} complete lines",
                          file=sys.stderr, flush=True)
                    continue
                raw[prompt["id"]] = text[-2000:]
                still.append(prompt)
        print(f"[llm] attempt {attempt + 1}/{attempts}: {len(pending) - len(still)}/{len(pending)} prompts parsed",
              file=sys.stderr, flush=True)
        pending = still
    for prompt in pending:
        results[prompt["id"]] = None
    return {"results": results, "raw": raw}
