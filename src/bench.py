"""Latency comparison harness: Cerebras vs a GPU-based OpenAI-compatible baseline.

Runs the same prompt against both providers, measures time-to-first-token and
total time, and writes a side-by-side summary to ``out/bench.json`` for the demo.

Configure the baseline via env (any OpenAI-compatible endpoint), e.g.:
    BASELINE_BASE_URL=https://api.openai.com/v1
    BASELINE_API_KEY=sk-...
    BASELINE_MODEL=gpt-4o-mini

Usage:
    uv run python -m src.bench --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import httpx

from src.llm import CerebrasLLM

OUT_PATH = Path("out/bench.json")

PROMPT_SYSTEM = "You are a concise physics tutor."
PROMPT_USER = (
    "Explain, in about 120 words, how to find the displacement of an object "
    "given a velocity-vs-time graph. Include the key idea that displacement is "
    "the area under the curve."
)


@dataclass
class RunStat:
    ttft_s: float
    total_s: float


@dataclass
class ProviderResult:
    provider: str
    model: str
    runs: list[RunStat]

    @property
    def avg_ttft(self) -> float:
        return mean(r.ttft_s for r in self.runs) if self.runs else float("nan")

    @property
    def avg_total(self) -> float:
        return mean(r.total_s for r in self.runs) if self.runs else float("nan")


def bench_cerebras(model: str, runs: int) -> ProviderResult:
    llm = CerebrasLLM(model=model)
    stats: list[RunStat] = []
    for _ in range(runs):
        r = llm.chat(PROMPT_SYSTEM, PROMPT_USER, temperature=0.2, max_tokens=512)
        stats.append(RunStat(ttft_s=r.ttft_s, total_s=r.total_s))
    return ProviderResult(provider="Cerebras", model=model, runs=stats)


def bench_openai_compatible(
    base_url: str, api_key: str, model: str, runs: int
) -> ProviderResult:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 512,
    }

    stats: list[RunStat] = []
    with httpx.Client(timeout=120) as client:
        for _ in range(runs):
            start = time.time()
            first_token_at: float | None = None
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
                    if delta and first_token_at is None:
                        first_token_at = time.time()
            total = time.time() - start
            stats.append(
                RunStat(
                    ttft_s=(first_token_at - start) if first_token_at else total,
                    total_s=total,
                )
            )
    return ProviderResult(provider="GPU baseline", model=model, runs=stats)


def _print_row(name: str, ttft: float, total: float) -> None:
    print(f"  {name:<28} ttft={ttft:6.3f}s   total={total:6.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cerebras vs GPU baseline latency.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cerebras-model", default="gemma-4-31b")
    args = parser.parse_args()

    print(f"Benchmarking {args.runs} runs/provider on identical prompt…\n")

    cerebras = bench_cerebras(args.cerebras_model, args.runs)
    print("Cerebras:")
    _print_row(cerebras.model, cerebras.avg_ttft, cerebras.avg_total)

    baseline: ProviderResult | None = None
    base_url = os.environ.get("BASELINE_BASE_URL")
    api_key = os.environ.get("BASELINE_API_KEY")
    model = os.environ.get("BASELINE_MODEL")
    if base_url and api_key and model:
        baseline = bench_openai_compatible(base_url, api_key, model, args.runs)
        print("\nGPU baseline:")
        _print_row(baseline.model, baseline.avg_ttft, baseline.avg_total)
        speedup = baseline.avg_total / cerebras.avg_total if cerebras.avg_total else 0
        print(f"\n  >> Cerebras is {speedup:.1f}x faster (total time).")
    else:
        print(
            "\n  (GPU baseline not configured. Set BASELINE_BASE_URL / "
            "BASELINE_API_KEY / BASELINE_MODEL to compare.)"
        )

    summary = {
        "prompt": PROMPT_USER,
        "runs_per_provider": args.runs,
        "cerebras": {
            "provider": cerebras.provider,
            "model": cerebras.model,
            "avg_ttft_s": round(cerebras.avg_ttft, 3),
            "avg_total_s": round(cerebras.avg_total, 3),
            "runs": [asdict(r) for r in cerebras.runs],
        },
    }
    if baseline:
        summary["baseline"] = {
            "provider": baseline.provider,
            "model": baseline.model,
            "avg_ttft_s": round(baseline.avg_ttft, 3),
            "avg_total_s": round(baseline.avg_total, 3),
            "runs": [asdict(r) for r in baseline.runs],
        }
        summary["speedup_total"] = round(
            baseline.avg_total / cerebras.avg_total, 2
        ) if cerebras.avg_total else None

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
