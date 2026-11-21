"""Voice design worker (Indic Parler-TTS): original voices from text descriptions. Runtime: parler."""

from __future__ import annotations

from .common import Progress, device


def run(request: dict) -> dict:
    import soundfile as sf
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    dev = "cuda:0" if device() == "cuda" and torch.cuda.is_available() else "cpu"
    repo = request["hf_repo"]
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(dev)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    rate = model.config.sampling_rate

    out = {}
    progress = Progress("voice-design", len(request["items"]))
    for item in request["items"]:
        progress.step()
        torch.manual_seed(int(item.get("seed", 0)))
        description = description_tokenizer(item["description"], return_tensors="pt").to(dev)
        prompt = tokenizer(item["text"], return_tensors="pt").to(dev)
        generation = model.generate(
            input_ids=description.input_ids,
            attention_mask=description.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        audio = generation.cpu().numpy().squeeze()
        sf.write(item["out"], audio, rate)
        out[item["id"]] = {"path": item["out"], "duration_s": float(len(audio) / rate), "sample_rate": rate}
    return {"voices": out}
