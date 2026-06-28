"""Course runner: drives a learner through a seeded course with a mastery loop.

Loads ``data/seed_course.json``, generates each lesson (video + applet +
exercise), grades the student's response, and decides whether to advance to the
next concept, give a harder variant, or remediate the current concept.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agents import Grader
from src.llm import CerebrasLLM
from src.orchestrator import EventSink, LessonGenerator
from src.video import hyperframes
from src.schemas import (
    Attempt,
    Concept,
    CourseSession,
    Exercise,
    GradeResult,
    Lesson,
    ProgressionDecision,
    StudentProfile,
)

SEED_PATH = Path("data/seed_course.json")

# Progression thresholds.
ADVANCE_SCORE = 0.8
ADVANCE_MASTERY = 0.75
HARDER_SCORE = 0.5


def load_seed_course(path: Path | str = SEED_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_STOPWORDS = {
    "the", "and", "vs", "of", "in", "a", "an", "to", "with", "for", "on",
    "motion", "physics", "relative",
}


def concept_keywords(concept: Concept) -> set[str]:
    """Salient keywords from a concept title, used for exercise relevance."""
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in concept.title)
    return {
        w.lower()
        for w in words.split()
        if len(w) > 2 and w.lower() not in _STOPWORDS
    }


def select_exercise(
    bank: list[Exercise],
    difficulty: int,
    *,
    keywords: set[str] | None = None,
    exclude_ids: set[str] | None = None,
) -> Exercise | None:
    """Pick an exercise that is relevant to the concept (keyword overlap),
    near the target difficulty, fresh, and preferably conceptual/diagram-tagged."""
    exclude_ids = exclude_ids or set()
    keywords = keywords or set()
    candidates = [e for e in bank if e.id not in exclude_ids] or bank
    if not candidates:
        return None

    def relevance(e: Exercise) -> int:
        haystack = f"{e.problem} {e.simplified} {' '.join(e.soft_labels)}".lower()
        return sum(1 for kw in keywords if kw in haystack)

    def score(e: Exercise) -> tuple:
        pref = 0 if ({"conceptual", "diagram"} & set(e.soft_labels)) else 1
        # Higher relevance first (negate), then closest difficulty, then label pref.
        return (-relevance(e), abs(e.difficulty - difficulty), pref)

    return min(candidates, key=score)


class CourseRunner:
    """Stateful controller around a single ``CourseSession``."""

    def __init__(self, llm: CerebrasLLM | None = None):
        self.llm = llm or CerebrasLLM()
        self.generator = LessonGenerator(self.llm)
        self.grader = Grader(self.llm)

    def start(
        self,
        student: StudentProfile,
        *,
        seed: dict | None = None,
    ) -> CourseSession:
        seed = seed or load_seed_course()
        concepts = [Concept.from_dict(c) for c in seed["concepts"]]
        bank = [Exercise.from_dict(e) for e in seed["exercise_bank"]]
        return CourseSession(
            student=student,
            chapter=seed.get("chapter", ""),
            concepts=concepts,
            exercise_bank=bank,
        )

    def _used_exercise_ids(self, session: CourseSession) -> set[str]:
        return {a.exercise_id for a in session.history}

    def generate_lesson(
        self,
        session: CourseSession,
        *,
        variant: str = "base",
        misconception: str | None = None,
        event_sink: EventSink | None = None,
        render_video: bool = False,
    ) -> Lesson:
        concept = session.current_concept
        if concept is None:
            raise RuntimeError("Course is already finished.")

        difficulty = session.effective_difficulty()
        exercise = select_exercise(
            session.exercise_bank,
            difficulty,
            keywords=concept_keywords(concept),
            exclude_ids=self._used_exercise_ids(session),
        )
        lesson = self.generator.generate(
            concept,
            difficulty,
            exercise,
            session.student,
            variant=variant,
            misconception=misconception,
            event_sink=event_sink,
        )

        # Persist the composition; optionally kick off a background MP4 render.
        lesson_id = f"{session.id}_{len(session.history)}_{concept.id}_{variant}"
        if lesson.video_composition:
            hyperframes.save_composition(lesson.video_composition, lesson_id)
            if render_video:
                lesson.video_mp4_url = hyperframes.render_mp4_async(lesson_id)

        session.current_lesson = lesson
        return lesson

    def decide(self, session: CourseSession, grade: GradeResult) -> ProgressionDecision:
        """Map a grade to the next action (advance / harder / remediate)."""
        current = session.effective_difficulty()
        if grade.score >= ADVANCE_SCORE and grade.mastery >= ADVANCE_MASTERY:
            return ProgressionDecision(
                action="advance",
                difficulty=0,
                rationale="Strong mastery shown; moving to the next concept.",
            )
        if grade.score >= HARDER_SCORE:
            return ProgressionDecision(
                action="harder",
                difficulty=min(10, current + 2),
                rationale="Good progress; reinforcing with a tougher variant.",
            )
        return ProgressionDecision(
            action="remediate",
            difficulty=max(1, current - 2),
            rationale="Struggling; re-teaching the concept more gently.",
        )

    def apply_decision(
        self, session: CourseSession, decision: ProgressionDecision
    ) -> None:
        if decision.action == "advance":
            session.current_index += 1
            session.current_difficulty = 0
        else:
            session.current_difficulty = decision.difficulty

    def submit(
        self,
        session: CourseSession,
        answer_text: str,
        *,
        image_path: str | None = None,
        event_sink: EventSink | None = None,
    ) -> tuple[GradeResult, ProgressionDecision]:
        lesson = session.current_lesson
        if lesson is None or lesson.exercise is None:
            raise RuntimeError("No active lesson/exercise to grade.")

        if event_sink:
            event_sink({"agent": "Grader", "status": "start"})
        grade, t = self.grader.run(lesson.exercise, answer_text, image_path=image_path)
        if event_sink:
            event_sink(
                {
                    "agent": "Grader",
                    "status": "done",
                    "ttft_s": round(t.ttft_s, 3),
                    "total_s": round(t.total_s, 3),
                    "correct": grade.correct,
                    "score": grade.score,
                }
            )

        decision = self.decide(session, grade)
        session.history.append(
            Attempt(
                concept_id=lesson.concept.id,
                exercise_id=lesson.exercise.id,
                answer_text=answer_text,
                grade=grade,
                decision=decision,
            )
        )
        self.apply_decision(session, decision)
        return grade, decision
