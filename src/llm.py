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

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemma-4-31b"


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
