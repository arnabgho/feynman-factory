"""Thin Cerebras + Gemma 4 31B client wrapper shared by all agents.

Supports plain text, JSON-object responses, optional image (vision) input, and
per-call latency stats so we can showcase Cerebras speed in the demo.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemma-4-31b"

# OpenAI-compatible endpoints used for the apples-to-apples GPU-vs-Cerebras race.
CEREBRAS_BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.environ.get("RACE_CEREBRAS_MODEL", "gemma-4-31b")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
GEMINI_MODEL = os.environ.get("RACE_GEMINI_MODEL", "gemma-4-31b-it")


@dataclass
class LLMResult:
    text: str
    ttft_s: float
    total_s: float
    chunks: int = 0
    raw: Any = field(default=None, repr=False)

    def as_json(self) -> Any:
        """Parse the response as JSON, tolerating code fences and trailing junk.

        Falls back to extracting the first balanced top-level {...} object, which
        guards against occasional runaway generations after a valid object.
        """
        cleaned = self.text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.removeprefix("json").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return json.loads(_first_json_object(cleaned))


def _first_json_object(text: str) -> str:
    """Return the first balanced {...} substring, respecting strings/escapes."""
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise json.JSONDecodeError("unbalanced JSON object", text, start)


def _image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class CerebrasLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = Cerebras(api_key=api_key or os.environ.get("CEREBRAS_API_KEY"))

    def chat(
        self,
        system: str,
        user: str,
        *,
        image_paths: list[str | Path] | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stream_to_stdout: bool = False,
    ) -> LLMResult:
        """Single-turn chat. Optionally attach images and/or constrain output.

        Pass ``schema`` (a JSON Schema dict) for guaranteed structured output, or
        ``json_mode=True`` for a free-form JSON object.
        """
        user_content: Any = user
        if image_paths:
            user_content = [{"type": "text", "text": user}]
            for p in image_paths:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(p)},
                    }
                )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "top_p": 1,
            "stream": True,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        first_token_at: float | None = None
        chunks = 0
        parts: list[str] = []

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and first_token_at is None:
                first_token_at = time.time()
            if delta:
                parts.append(delta)
                chunks += 1
                if stream_to_stdout:
                    print(delta, end="", flush=True)

        if stream_to_stdout:
            print()

        total = time.time() - start
        ttft = (first_token_at - start) if first_token_at else total
        return LLMResult(
            text="".join(parts),
            ttft_s=ttft,
            total_s=total,
            chunks=chunks,
        )


class OpenAICompatLLM:
    """Drop-in replacement for :class:`CerebrasLLM` against any OpenAI-compatible
    ``/chat/completions`` endpoint (used for the Gemini/GPU race lane).

    Mirrors ``CerebrasLLM.chat``'s signature and ``LLMResult`` return so the same
    agents and orchestrator run unchanged on a different inference backend.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # The ExplainerVideo step can stream for minutes on a GPU backend, so we
        # allow an unbounded read timeout while still failing fast on connect.
        self._timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)

    def chat(
        self,
        system: str,
        user: str,
        *,
        image_paths: list[str | Path] | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stream_to_stdout: bool = False,
    ) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("OpenAICompatLLM is missing an API key")

        system_content = system
        response_format: dict[str, Any] | None = None
        # Gemini's OpenAI-compatible surface is most reliable with plain
        # json_object mode; nudge the schema via the system prompt and lean on
        # LLMResult.as_json()'s tolerant parsing.
        if schema is not None:
            response_format = {"type": "json_object"}
            system_content = (
                f"{system}\n\nReturn ONLY a single JSON object that conforms to this "
                f"JSON Schema (no prose, no markdown fences):\n{json.dumps(schema)}"
            )
        elif json_mode:
            response_format = {"type": "json_object"}

        user_content: Any = user
        if image_paths:
            user_content = [{"type": "text", "text": user}]
            for p in image_paths:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(p)}}
                )

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": True,
        }
        if response_format is not None:
            body["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self.base_url + "/chat/completions"

        start = time.time()
        first_token_at: float | None = None
        chunks = 0
        parts: list[str] = []

        with httpx.Client(timeout=self._timeout) as client:
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
                    if first_token_at is None:
                        first_token_at = time.time()
                    parts.append(delta)
                    chunks += 1
                    if stream_to_stdout:
                        print(delta, end="", flush=True)

        if stream_to_stdout:
            print()

        total = time.time() - start
        ttft = (first_token_at - start) if first_token_at else total
        return LLMResult(text="".join(parts), ttft_s=ttft, total_s=total, chunks=chunks)


def make_llm(provider: str):
    """Build an LLM client for a race lane. ``provider`` is "cerebras" or "gemini"."""
    if provider == "cerebras":
        return CerebrasLLM(model=CEREBRAS_MODEL)
    if provider == "gemini":
        return OpenAICompatLLM(
            base_url=GEMINI_BASE_URL,
            model=GEMINI_MODEL,
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unknown provider: {provider}")
