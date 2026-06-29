"""FastAPI backend exposing the Lesson Factory as a streaming API.

Run with:
    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import sessions
from src import race, system_race
from src.data import pdf_course
from src.schemas import CourseSession, Lesson, StudentProfile

app = FastAPI(title="Lesson Factory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(OUT_DIR)), name="assets")


# --- request/response models -------------------------------------------------


class StartRequest(BaseModel):
    name: str = "Student"
    grade_level: str = "8th grade"
    known_concepts: list[str] = []


def _course_outline(session: CourseSession) -> dict[str, Any]:
    return {
        "session_id": session.id,
        "chapter": session.chapter,
        "subject": session.subject,
        "current_index": session.current_index,
        "total": len(session.concepts),
        "finished": session.finished,
        "concepts": [
            {"order": c.order, "title": c.title, "seed_difficulty": c.seed_difficulty}
            for c in session.concepts
        ],
    }


def _lesson_payload(session: CourseSession, lesson: Lesson) -> dict[str, Any]:
    return {
        "session_id": session.id,
        "concept": {"title": lesson.concept.title, "order": lesson.concept.order},
        "difficulty": lesson.difficulty,
        "variant": lesson.variant,
        "video_html": lesson.video_composition,
        "applet_html": lesson.applet.html if lesson.applet else "",
        "applet_approved": bool(lesson.applet and lesson.applet.approved),
        "exercise": lesson.exercise.public_dict() if lesson.exercise else None,
        "mp4_url": lesson.video_mp4_url,
        "progress": {"index": session.current_index, "total": len(session.concepts)},
    }


# --- SSE helper --------------------------------------------------------------


async def _stream_blocking(
    blocking: Callable[[Callable[[dict], None]], Any],
    result_type: str,
    result_serializer: Callable[[Any], dict],
) -> AsyncIterator[dict]:
    """Run a blocking function (that takes an event sink) and stream its events.

    Agent progress events are forwarded as ``agent`` SSE messages; the function's
    return value is emitted as a final ``result_type`` message.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def sink(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "agent", **event})

    async def worker() -> None:
        try:
            result = await loop.run_in_executor(None, lambda: blocking(sink))
            payload = result_serializer(result)
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": result_type, **payload}
            )
        except Exception as exc:  # noqa: BLE001 - surface error to client
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    task = asyncio.create_task(worker())
    while True:
        event = await queue.get()
        if event is DONE:
            break
        yield {"data": json.dumps(event)}
    await task


# --- endpoints ---------------------------------------------------------------


@app.post("/course/start")
def start_course(req: StartRequest) -> dict[str, Any]:
    student = StudentProfile(
        name=req.name,
        grade_level=req.grade_level,
        known_concepts=req.known_concepts,
    )
    session = sessions.runner.start(student, seed=sessions.get_seed())
    sessions.create_session(session)
    return _course_outline(session)


@app.post("/course/start_from_pdf")
async def start_course_from_pdf(
    pdf: UploadFile = File(...),
    name: str = Form("Student"),
    grade_level: str = Form("8th grade"),
) -> dict[str, Any]:
    suffix = Path(pdf.filename or "upload.pdf").suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await pdf.read())
    tmp.close()

    try:
        text = pdf_course.extract_text(tmp.name)
        course = pdf_course.build_course(sessions.runner.llm, text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    student = StudentProfile(name=name, grade_level=grade_level)
    session = sessions.runner.start(student, seed=course)
    sessions.create_session(session)
    return _course_outline(session)


@app.get("/course/{session_id}/state")
def course_state(session_id: str) -> dict[str, Any]:
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    return _course_outline(session)


@app.post("/course/{session_id}/advance")
def advance_course(session_id: str) -> dict[str, Any]:
    """Advance to the next concept without grading (document/concept-only courses)."""
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    sessions.runner.advance(session)
    return _course_outline(session)


@app.post("/course/{session_id}/lesson/generate")
async def generate_lesson(session_id: str, remix: str | None = None):
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    if session.finished:
        raise HTTPException(400, "Course finished")

    def blocking(sink: Callable[[dict], None]) -> Lesson:
        return sessions.runner.generate_lesson(session, remix=remix, event_sink=sink)

    return EventSourceResponse(
        _stream_blocking(
            blocking,
            result_type="lesson",
            result_serializer=lambda lesson: _lesson_payload(session, lesson),
        )
    )


@app.post("/course/{session_id}/lesson/submit")
async def submit_answer(
    session_id: str,
    answer_text: str = Form(""),
    image: UploadFile | None = File(None),
):
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    if session.current_lesson is None:
        raise HTTPException(400, "No active lesson")

    image_path: str | None = None
    if image is not None:
        suffix = Path(image.filename or "upload.png").suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image.read())
        tmp.close()
        image_path = tmp.name

    def blocking(sink: Callable[[dict], None]):
        return sessions.runner.submit(
            session, answer_text, image_path=image_path, event_sink=sink
        )

    def serialize(result) -> dict[str, Any]:
        grade, decision = result
        return {
            "grade": {
                "correct": grade.correct,
                "score": grade.score,
                "mastery": grade.mastery,
                "feedback": grade.feedback,
                "misconception": grade.misconception,
            },
            "decision": {
                "action": decision.action,
                "difficulty": decision.difficulty,
                "rationale": decision.rationale,
            },
            "finished": session.finished,
            "next_index": session.current_index,
        }

    return EventSourceResponse(
        _stream_blocking(blocking, result_type="result", result_serializer=serialize)
    )


@app.post("/race")
async def speed_race(topic: str | None = None):
    """Stream a live GPU-vs-Cerebras race of the SAME model (Gemma 4 31B)."""

    def blocking(sink: Callable[[dict], None]) -> dict:
        return race.run_race(lambda ev: sink({"type": "race", **ev}), topic=topic)

    async def gen():
        async for item in _stream_blocking(
            blocking,
            result_type="race_summary",
            result_serializer=lambda summary: summary,
        ):
            yield item

    return EventSourceResponse(gen())


@app.post("/race/system")
async def system_speed_race(topic: str | None = None, grade_level: str = "8th grade"):
    """Stream a full-pipeline GPU-vs-Cerebras race (the whole agent swarm)."""

    def blocking(sink: Callable[[dict], None]) -> dict:
        return system_race.run_system_race(
            lambda ev: sink({"type": "sysrace", **ev}),
            topic=topic,
            grade_level=grade_level,
        )

    async def gen():
        async for item in _stream_blocking(
            blocking,
            result_type="sysrace_summary",
            result_serializer=lambda summary: summary,
        ):
            yield item

    return EventSourceResponse(gen())


@app.get("/race/config")
def race_config() -> dict[str, Any]:
    """Report which lanes are configured (so the UI can prompt for keys)."""
    import os

    return {
        "cerebras": {"configured": bool(os.environ.get("CEREBRAS_API_KEY")), "model": race.CEREBRAS_MODEL},
        "gemini": {"configured": bool(os.environ.get("GEMINI_API_KEY")), "model": race.GEMINI_MODEL},
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


WEB_INDEX = Path("web/index.html")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_INDEX))
