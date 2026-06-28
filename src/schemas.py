"""Shared state passed between agents in the lesson-factory swarm."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StudentProfile:
    name: str
    grade_level: str
    last_attempt: str | None = None
    last_attempt_correct: bool | None = None
    known_concepts: list[str] = field(default_factory=list)
    work_image_path: str | None = None


@dataclass
class Diagnosis:
    """Output of the Tutor/Diagnostician agent."""

    next_concept: str
    difficulty: int  # 1-10
    misconception: str | None
    rationale: str


@dataclass
class LessonPlan:
    """Output of the Planner agent."""

    concept: str
    learning_objective: str
    explanation_beats: list[str]
    applet_spec: str  # natural-language spec of the interactive element


@dataclass
class Slide:
    """One slide in the explainer deck (input spec for the video build)."""

    kind: str  # "title" | "content" | "diagram" | "recap"
    title: str
    bullets: list[str] = field(default_factory=list)
    visual: str = ""  # description of the animation/diagram, or "none"
    duration: float = 10.0  # seconds this slide is on screen

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Slide":
        return cls(
            kind=str(d.get("kind", "content")),
            title=str(d.get("title", "")),
            bullets=[str(b) for b in d.get("bullets", [])],
            visual=str(d.get("visual", "")),
            duration=float(d.get("duration", 10.0)),
        )


@dataclass
class SlideDeck:
    """Output of the SlideDeckPlanner: an ordered, timed outline for the video."""

    title: str
    total_duration: float
    slides: list[Slide] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlideDeck":
        slides = [Slide.from_dict(s) for s in d.get("slides", [])]
        total = float(d.get("total_duration", 0) or sum(s.duration for s in slides))
        return cls(title=str(d.get("title", "")), total_duration=total, slides=slides)


@dataclass
class Applet:
    """Output of the Applet Builder agent (a self-contained HTML document)."""

    html: str
    approved: bool = False
    critique: str | None = None


@dataclass
class LessonState:
    """Full state threaded through the orchestrated pipeline."""

    student: StudentProfile
    diagnosis: Diagnosis | None = None
    plan: LessonPlan | None = None
    applet: Applet | None = None
    timings: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


# --- Seed-course data models -------------------------------------------------


@dataclass
class Concept:
    """A unit of curriculum, sourced from OpenStax."""

    id: str
    title: str
    summary: str
    prose: str
    chapter: str = ""
    order: int = 0
    seed_difficulty: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Concept":
        return cls(
            id=d["id"],
            title=d["title"],
            summary=d.get("summary", ""),
            prose=d.get("prose", ""),
            chapter=d.get("chapter", ""),
            order=int(d.get("order", 0)),
            seed_difficulty=int(d.get("seed_difficulty", 3)),
        )


@dataclass
class Exercise:
    """A practice problem, sourced from PhysicsEval."""

    id: str
    problem: str
    category: str = ""
    difficulty: int = 0
    solution_steps: str = ""
    final_answer: str = ""
    simplified: str = ""
    soft_labels: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exercise":
        return cls(
            id=str(d.get("id", "")),
            problem=d.get("problem", ""),
            category=d.get("category", ""),
            difficulty=int(d.get("difficulty", 0)),
            solution_steps=d.get("solution_steps", ""),
            final_answer=d.get("final_answer", ""),
            simplified=d.get("simplified", ""),
            soft_labels=list(d.get("soft_labels", [])),
        )

    def public_dict(self) -> dict[str, Any]:
        """Student-facing view (hides the solution)."""
        return {
            "id": self.id,
            "problem": self.problem,
            "category": self.category,
            "difficulty": self.difficulty,
        }


# --- Lesson, grading, progression -------------------------------------------


@dataclass
class Lesson:
    """A generated lesson: an explainer video plus an interactive exercise."""

    concept: Concept
    difficulty: int
    plan: LessonPlan | None = None
    video_composition: str = ""  # self-contained HyperFrames/HTML composition
    applet: Applet | None = None
    exercise: Exercise | None = None
    variant: Literal["base", "harder", "remediate"] = "base"
    video_mp4_url: str | None = None


@dataclass
class GradeResult:
    """Output of the Grader agent."""

    correct: bool
    score: float  # 0-1 on this attempt
    mastery: float  # 0-1 estimate of concept mastery
    feedback: str
    misconception: str | None = None


@dataclass
class ProgressionDecision:
    """How the course should proceed after a graded attempt."""

    action: Literal["advance", "harder", "remediate"]
    difficulty: int
    rationale: str


@dataclass
class Attempt:
    """A record of one graded student attempt within a session."""

    concept_id: str
    exercise_id: str
    answer_text: str
    grade: GradeResult
    decision: ProgressionDecision


@dataclass
class CourseSession:
    """In-memory state for one learner moving through a seeded course."""

    student: StudentProfile
    chapter: str
    concepts: list[Concept]
    exercise_bank: list[Exercise]
    subject: str = "physics"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    current_index: int = 0
    current_difficulty: int = 0  # 0 => use concept.seed_difficulty
    current_lesson: Lesson | None = None
    history: list[Attempt] = field(default_factory=list)

    @property
    def current_concept(self) -> Concept | None:
        if 0 <= self.current_index < len(self.concepts):
            return self.concepts[self.current_index]
        return None

    @property
    def finished(self) -> bool:
        return self.current_index >= len(self.concepts)

    def effective_difficulty(self) -> int:
        concept = self.current_concept
        base = concept.seed_difficulty if concept else 3
        return self.current_difficulty or base
