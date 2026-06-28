"""Tutor / Diagnostician agent.

Looks at the student's profile and (optionally) a photo/screenshot of their
work, then decides what concept to teach next, at what difficulty, and what
misconception to target.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import Diagnosis, StudentProfile

SYSTEM = """You are an expert physics tutor and diagnostician.
Given a student's profile and (optionally) an image of their work, decide the
single best next concept to teach, the appropriate difficulty (1-10), and the
specific misconception to address (or null if none)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "next_concept": {"type": "string"},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 10},
        "misconception": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["next_concept", "difficulty", "misconception", "rationale"],
    "additionalProperties": False,
}


class Tutor:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(self, student: StudentProfile) -> tuple[Diagnosis, LLMResult]:
        attempt = (
            f"Last attempt: {student.last_attempt!r} "
            f"(correct={student.last_attempt_correct})"
            if student.last_attempt is not None
            else "No prior attempt yet."
        )
        user = (
            f"Student: {student.name}, grade level: {student.grade_level}.\n"
            f"Known concepts: {', '.join(student.known_concepts) or 'none recorded'}.\n"
            f"{attempt}\n"
            "Diagnose and choose the next concept."
        )
        image_paths = [student.work_image_path] if student.work_image_path else None
        result = self.llm.chat(
            SYSTEM,
            user,
            image_paths=image_paths,
            schema=SCHEMA,
            schema_name="diagnosis",
            temperature=0.3,
        )
        data = result.as_json()
        return (
            Diagnosis(
                next_concept=data["next_concept"],
                difficulty=int(data["difficulty"]),
                misconception=data.get("misconception"),
                rationale=data["rationale"],
            ),
            result,
        )
