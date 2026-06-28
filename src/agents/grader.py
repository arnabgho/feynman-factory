"""Grader agent.

Evaluates a student's answer to an exercise (text and/or a photo/screenshot of
their work) against the known solution, and estimates concept mastery.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import Exercise, GradeResult

SYSTEM = """You are a fair, encouraging physics grader.
You are given a problem, its reference solution, and a student's answer (which
may include an image of their handwritten or on-screen work).

Judge correctness generously on physics reasoning and final result (accept
equivalent forms and reasonable rounding). Estimate:
- score: how correct THIS answer is, 0.0 to 1.0
- mastery: your confidence the student understands the concept, 0.0 to 1.0
Give short, specific, encouraging feedback, and name the misconception if any
(else null)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "mastery": {"type": "number", "minimum": 0, "maximum": 1},
        "feedback": {"type": "string"},
        "misconception": {"type": ["string", "null"]},
    },
    "required": ["correct", "score", "mastery", "feedback", "misconception"],
    "additionalProperties": False,
}


class Grader:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(
        self,
        exercise: Exercise,
        answer_text: str,
        *,
        image_path: str | None = None,
    ) -> tuple[GradeResult, LLMResult]:
        user = (
            f"Problem:\n{exercise.problem}\n\n"
            f"Reference solution steps:\n{exercise.solution_steps or 'n/a'}\n"
            f"Reference final answer: {exercise.final_answer or 'n/a'}\n\n"
            f"Student's answer (text): {answer_text or '(see attached image)'}\n"
            "Grade the student's answer."
        )
        result = self.llm.chat(
            SYSTEM,
            user,
            image_paths=[image_path] if image_path else None,
            schema=SCHEMA,
            schema_name="grade",
            temperature=0.1,
        )
        data = result.as_json()
        return (
            GradeResult(
                correct=bool(data["correct"]),
                score=float(data["score"]),
                mastery=float(data["mastery"]),
                feedback=data["feedback"],
                misconception=data.get("misconception"),
            ),
            result,
        )
