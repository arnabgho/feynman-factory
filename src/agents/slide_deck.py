"""SlideDeckPlanner agent.

Turns a lesson plan into a tight, timed slide-deck outline (~90-120s, 8-14
slides). The ExplainerVideo agent then builds the actual animated composition
from this outline, which keeps long videos reliable (each slide has a small,
well-scoped spec instead of one-shot improvisation).
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import LessonPlan, SlideDeck

SYSTEM = """You are a presentation designer who plans short educational explainer videos as a slide deck.

Given a lesson plan, produce an ordered outline of 8-14 slides that together run
about 90-120 seconds. Pace it like a real explainer: open with a title slide,
build understanding across content and diagram slides, and close with a recap.

Each slide has:
- kind: one of "title" (opening), "content" (text/bullets), "diagram" (a visual
  the animator should draw - a graph, vectors, a labeled figure, a worked
  step), or "recap" (closing summary).
- title: a short, punchy slide headline.
- bullets: 0-4 very short bullet phrases (<= ~8 words each). Title/diagram/recap
  slides may use 0-2; content slides usually 2-4.
- visual: a concise description of the on-screen visual or animation for this
  slide (e.g., "a dot moving along a number line showing displacement vs
  distance"). Use "none" only for pure text slides.
- duration: seconds this slide is on screen (typically 6-14). The sum of all
  durations MUST equal total_duration.

Make the sequence build logically, one idea per slide, with at least 2-3
diagram slides so the video is visual, not just text. Keep language at the
student's level."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "total_duration": {"type": "number"},
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["title", "content", "diagram", "recap"],
                    },
                    "title": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "visual": {"type": "string"},
                    "duration": {"type": "number"},
                },
                "required": ["kind", "title", "bullets", "visual", "duration"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "total_duration", "slides"],
    "additionalProperties": False,
}


class SlideDeckPlanner:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(
        self,
        plan: LessonPlan,
        difficulty: int,
        *,
        subject: str = "physics",
        remix: str | None = None,
    ) -> tuple[SlideDeck, LLMResult]:
        beats = "\n".join(f"- {b}" for b in plan.explanation_beats)
        remix_line = (
            f"REMIX DIRECTIVE (shape the deck this way): {remix}\n" if remix else ""
        )
        user = (
            f"Subject: {subject}\n"
            f"Concept: {plan.concept}\n"
            f"Learning objective: {plan.learning_objective}\n"
            f"Target difficulty (1-10): {difficulty}\n"
            f"Explanation beats:\n{beats}\n"
            f"{remix_line}\n"
            "Plan the 90-120s slide deck now (8-14 slides). Ensure slide "
            "durations sum to total_duration."
        )
        result = self.llm.chat(
            SYSTEM, user, schema=SCHEMA, schema_name="slide_deck", temperature=0.4
        )
        return SlideDeck.from_dict(result.as_json()), result
