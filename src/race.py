"""Live, apples-to-apples speed race: the SAME model on GPU vs Cerebras.

Both providers run **Gemma 4 31B** through an OpenAI-compatible streaming
endpoint, so the only variable is the inference hardware:

  - Cerebras: gemma-4-31b on the Cerebras wafer-scale engine.
  - Gemini (GPU): gemma-4-31b-it served by Google's Gemini API.

We stream identical prompts from both at once and emit progress events
(time-to-first-token, tokens, tokens/sec) so the UI can show two lanes racing.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import httpx

EventSink = Callable[[dict], None]

# Both lanes hit an OpenAI-compatible /chat/completions endpoint.
CEREBRAS_BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.environ.get("RACE_CEREBRAS_MODEL", "gemma-4-31b")

GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
GEMINI_MODEL = os.environ.get("RACE_GEMINI_MODEL", "gemma-4-31b-it")

RACE_SYSTEM = "You are an expert tutor who writes vivid, clear explanations."
RACE_MAX_TOKENS = int(os.environ.get("RACE_MAX_TOKENS", "700"))

# Rough token estimate from characters (good enough for a live tok/s readout).
def _est_tokens(chars: int) -> int:
    return max(0, round(chars / 4))


def _build_prompt(topic: str | None) -> str:
    topic = (topic or "projectile motion").strip()
    return (
        f"Explain {topic} to a curious student in about 200 words. "
        "Use an intuitive analogy, then state the key idea precisely. "
        "Write flowing prose (no headings or lists)."
    )


@dataclass
class LaneConfig:
    lane: str          # UI label, e.g. "Cerebras" / "Gemini (GPU)"
    provider: str      # short id
    base_url: str
    api_key: str | None
    model: str


def _stream_lane(cfg: LaneConfig, user: str, sink: EventSink) -> dict:
    """Stream one provider and emit start/progress/done (or error) events."""
    base = {"lane": cfg.lane, "provider": cfg.provider, "model": cfg.model}

    if not cfg.api_key:
        sink({**base, "status": "unconfigured"})
        return {**base, "status": "unconfigured"}

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": RACE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "temperature": 0.4,
        "max_tokens": RACE_MAX_TOKENS,
    }

    sink({**base, "status": "start"})
    start = time.time()
    first_at: float | None = None
    text = ""        # full streamed output so far
    emitted_len = 0  # chars already sent to the client as deltas
    last_emit = 0.0

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        delta = None
                    if not delta:
                        continue
                    if first_at is None:
                        first_at = time.time()
                    text += delta
                    now = time.time()
                    # Throttle progress events to ~20/s per lane.
                    if now - last_emit > 0.05:
                        last_emit = now
                        elapsed = now - start
                        toks = _est_tokens(len(text))
                        sink(
                            {
                                **base,
                                "status": "progress",
                                "tokens": toks,
                                "elapsed_s": round(elapsed, 3),
                                "ttft_s": round((first_at - start), 3),
                                "tps": round(toks / elapsed, 1) if elapsed > 0 else 0,
                                "delta": text[emitted_len:],
                            }
                        )
                        emitted_len = len(text)
    except Exception as exc:  # noqa: BLE001 - surface to UI, keep other lane alive
        sink({**base, "status": "error", "message": str(exc)[:200]})
        return {**base, "status": "error", "message": str(exc)[:200]}

    elapsed = time.time() - start
    toks = _est_tokens(len(text))
    summary = {
        **base,
        "status": "done",
        "tokens": toks,
        "elapsed_s": round(elapsed, 3),
        "ttft_s": round((first_at - start), 3) if first_at else round(elapsed, 3),
        "tps": round(toks / elapsed, 1) if elapsed > 0 else 0,
        "delta": text[emitted_len:],  # flush any un-emitted tail
        "text": text,                 # full output for correctness
    }
    sink(summary)
    return summary


def run_race(sink: EventSink, *, topic: str | None = None) -> dict:
    """Run both lanes concurrently; return a summary with the speedup."""
    user = _build_prompt(topic)
    lanes = [
        LaneConfig(
            lane="Cerebras",
            provider="cerebras",
            base_url=CEREBRAS_BASE_URL,
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model=CEREBRAS_MODEL,
        ),
        LaneConfig(
            lane="Gemini (GPU)",
            provider="gemini",
            base_url=GEMINI_BASE_URL,
            api_key=os.environ.get("GEMINI_API_KEY"),
            model=GEMINI_MODEL,
        ),
    ]

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(_stream_lane, cfg, user, sink) for cfg in lanes]
        results = [f.result() for f in futures]

    by_provider = {r["provider"]: r for r in results}
    cb = by_provider.get("cerebras", {})
    gm = by_provider.get("gemini", {})

    speedup = None
    if cb.get("status") == "done" and gm.get("status") == "done":
        c_total, g_total = cb.get("elapsed_s", 0), gm.get("elapsed_s", 0)
        if c_total:
            speedup = round(g_total / c_total, 1)

    return {
        "topic": topic or "projectile motion",
        "lanes": results,
        "speedup_total": speedup,
    }
