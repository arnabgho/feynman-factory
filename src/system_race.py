"""Full-system speed race: run the WHOLE lesson pipeline on GPU vs Cerebras.

Unlike :mod:`src.race` (a single streamed prompt), this runs the entire
``LessonGenerator`` swarm - Planner -> SlideDeckPlanner -> ExplainerVideo +
AppletBuilder -> Critic - twice in parallel, once per inference backend, using
the SAME Gemma 4 31B weights:

  - cerebras: gemma-4-31b on the Cerebras wafer-scale engine.
  - gemini (GPU): gemma-4-31b-it served by Google's Gemini API.

Each lane streams its live agent events (tagged with the provider) and, on
completion, emits the finished explainer video + interactive applet plus the
lane's wall-clock time, so the UI can render both systems racing side by side.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from src.llm import make_llm
from src.orchestrator import LessonGenerator
from src.schemas import Concept, StudentProfile

EventSink = Callable[[dict], None]

DEFAULT_DIFFICULTY = 4
DEFAULT_SUBJECT = "physics"


def _run_lane(
    provider: str,
    lane_label: str,
    topic: str,
    student: StudentProfile,
    sink: EventSink,
) -> dict:
    """Run the full pipeline for one backend; emit lane-tagged progress events."""
    base = {"provider": provider, "lane": lane_label}

    # Gracefully skip a lane with no credentials (e.g. missing GEMINI_API_KEY).
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        sink({**base, "status": "unconfigured"})
        return {**base, "status": "unconfigured"}

    try:
        llm = make_llm(provider)
    except Exception as exc:  # noqa: BLE001
        sink({**base, "status": "error", "message": str(exc)[:200]})
        return {**base, "status": "error", "message": str(exc)[:200]}

    concept = Concept(
        id="race",
        title=topic,
        summary=topic,
        prose="",
        chapter=topic,
        order=0,
        seed_difficulty=DEFAULT_DIFFICULTY,
    )

    model_time = 0.0

    def lane_sink(event: dict) -> None:
        nonlocal model_time
        if event.get("status") == "done" and isinstance(event.get("total_s"), (int, float)):
            model_time += float(event["total_s"])
        sink({**base, **event})

    sink({**base, "status": "start"})
    start = time.time()
    try:
        generator = LessonGenerator(llm=llm)
        lesson = generator.generate(
            concept,
            DEFAULT_DIFFICULTY,
            None,
            student,
            subject=DEFAULT_SUBJECT,
            event_sink=lane_sink,
        )
    except Exception as exc:  # noqa: BLE001 - surface to UI, keep other lane alive
        sink({**base, "status": "error", "message": str(exc)[:300]})
        return {**base, "status": "error", "message": str(exc)[:300]}

    elapsed = time.time() - start
    summary = {
        **base,
        "status": "lesson_done",
        "elapsed_s": round(elapsed, 3),
        "model_time_s": round(model_time, 3),
        "video_html": lesson.video_composition,
        "applet_html": lesson.applet.html if lesson.applet else "",
        "applet_approved": bool(lesson.applet and lesson.applet.approved),
        "concept": lesson.concept.title,
    }
    sink(summary)
    # Don't echo the (large) artifacts back into the aggregate result payload.
    return {**base, "status": "lesson_done", "elapsed_s": round(elapsed, 3)}


def run_system_race(
    sink: EventSink,
    *,
    topic: str | None = None,
    grade_level: str = "8th grade",
) -> dict:
    """Run the full pipeline on both lanes concurrently; return a speedup summary."""
    topic = (topic or "projectile motion").strip()
    student = StudentProfile(name="Demo", grade_level=grade_level)

    lanes = [
        ("cerebras", "Cerebras"),
        ("gemini", "Gemini (GPU)"),
    ]

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [
            pool.submit(_run_lane, provider, label, topic, student, sink)
            for provider, label in lanes
        ]
        results = [f.result() for f in futures]

    by_provider = {r["provider"]: r for r in results}
    cb = by_provider.get("cerebras", {})
    gm = by_provider.get("gemini", {})

    speedup = None
    if cb.get("status") == "lesson_done" and gm.get("status") == "lesson_done":
        c_total, g_total = cb.get("elapsed_s", 0), gm.get("elapsed_s", 0)
        if c_total:
            speedup = round(g_total / c_total, 1)

    return {"topic": topic, "lanes": results, "speedup_total": speedup}
