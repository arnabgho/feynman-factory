"""Planner agent.

Turns a diagnosed concept into a micro-lesson outline and a concrete spec for
the interactive applet the Applet Builder should create.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import Diagnosis, LessonPlan, StudentProfile

SYSTEM = """You are an instructional designer for physics micro-lessons.
Given a target concept, difficulty, and the student's level, produce a tight
micro-lesson plan plus a spec for ONE interactive browser applet that lets the
student actively explore the concept (e.g., a draggable projectile launcher,
a velocity-vs-time slider, a force-balance simulator).
Use 3-5 short explanation beats. The applet_spec should describe what the
interactive does and which misconception it targets."""

SCHEMA = {
    "type": "object",
    "properties": {
        "concept": {"type": "string"},
        "learning_objective": {"type": "string"},
        "explanation_beats": {"type": "array", "items": {"type": "string"}},
        "applet_spec": {"type": "string"},
    },
    "required": ["concept", "learning_objective", "explanation_beats", "applet_spec"],
    "additionalProperties": False,
}


class Planner:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(
        self,
        diagnosis: Diagnosis,
        student: StudentProfile,
        grounding: str | None = None,
    ) -> tuple[LessonPlan, LLMResult]:
        user = (
            f"Concept: {diagnosis.next_concept}\n"
            f"Difficulty (1-10): {diagnosis.difficulty}\n"
            f"Misconception to target: {diagnosis.misconception or 'none'}\n"
            f"Student grade level: {student.grade_level}\n"
        )
        if grounding:
            user += f"\nSource material to ground the lesson in:\n{grounding[:2000]}\n"
        user += "\nDesign the micro-lesson and applet spec."
        result = self.llm.chat(
            SYSTEM, user, schema=SCHEMA, schema_name="lesson_plan", temperature=0.4
        )
        data = result.as_json()
        return (
            LessonPlan(
                concept=data["concept"],
                learning_objective=data["learning_objective"],
                explanation_beats=list(data["explanation_beats"]),
                applet_spec=data["applet_spec"],
            ),
            result,
        )
