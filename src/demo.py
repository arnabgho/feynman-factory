"""Runnable demo: generate a personalized physics lesson + applet end-to-end.

Usage:
    uv run python -m src.demo
"""

from __future__ import annotations

from pathlib import Path

from src.orchestrator import LessonFactory
from src.schemas import StudentProfile

OUT_DIR = Path("out")


def main() -> None:
    student = StudentProfile(
        name="Maya",
        grade_level="8th grade",
        last_attempt="Thinks a heavier ball falls faster than a lighter one.",
        last_attempt_correct=False,
        known_concepts=["speed", "distance"],
    )

    factory = LessonFactory()
    state = factory.run(student)

    if state.applet and state.applet.html:
        OUT_DIR.mkdir(exist_ok=True)
        out_file = OUT_DIR / "applet.html"
        out_file.write_text(state.applet.html, encoding="utf-8")
        print(f"\nApplet written to {out_file.resolve()}")
        print(f"Approved by critic: {state.applet.approved}")


if __name__ == "__main__":
    main()
