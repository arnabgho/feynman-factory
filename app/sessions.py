"""In-memory session store and shared CourseRunner (sufficient for the hackathon)."""

from __future__ import annotations

from src.course import CourseRunner, load_seed_course
from src.schemas import CourseSession

# A single shared runner reuses one Cerebras client across requests.
runner = CourseRunner()

# Cache the seed course once at process start.
_seed_cache: dict | None = None

_sessions: dict[str, CourseSession] = {}


def get_seed() -> dict:
    global _seed_cache
    if _seed_cache is None:
        _seed_cache = load_seed_course()
    return _seed_cache


def create_session(session: CourseSession) -> None:
    _sessions[session.id] = session


def get_session(session_id: str) -> CourseSession | None:
    return _sessions.get(session_id)
