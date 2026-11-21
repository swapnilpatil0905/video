"""Helpers shared by workers (stdlib only)."""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Any


def device() -> str:
    return os.environ.get("VP_DEVICE", "cuda")


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in model output (handles ``` fences and <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model output")


class Progress:
    """Item progress for long worker loops: '[tts] 40/120 (33%) 2m10s elapsed, eta 4m20s'.

    Printed to stderr (the worker log); the platform heartbeat copies the latest line into run.log.
    Throttled to one line every ``every`` seconds, plus the first and last item.
    """

    def __init__(self, tag: str, total: int, every: float = 20.0) -> None:
        import sys
        import time

        self.tag, self.total, self.every = tag, max(int(total), 0), every
        self.done, self._t0, self._last = 0, time.time(), 0.0
        self._time, self._out = time.time, sys.stderr
        print(f"[{tag}] starting {self.total} items", file=self._out, flush=True)

    def step(self, note: str = "") -> None:
        """Call at the start of each item."""
        self.done += 1
        now = self._time()
        if self.done not in (1, self.total) and now - self._last < self.every:
            return
        self._last = now
        elapsed, finished = now - self._t0, self.done - 1
        eta = f", eta {_fmt(elapsed / finished * (self.total - finished))}" if finished else ""
        pct = 100 * finished / self.total if self.total else 100
        print(f"[{self.tag}] item {self.done}/{self.total} ({pct:.0f}% done) {_fmt(elapsed)} elapsed{eta}"
              + (f" — {note}" if note else ""), file=self._out, flush=True)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
