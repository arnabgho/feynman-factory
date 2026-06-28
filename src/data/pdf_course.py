"""Build a course from an uploaded document (PDF).

Extracts text with ``pypdf`` (no OCR: scanned/image-only PDFs won't yield text)
and uses the :class:`~src.agents.doc_course.DocumentCourseBuilder` agent to turn
that text into an ordered, subject-agnostic set of concepts. The resulting
course has an empty exercise bank: document lessons are video + applet only.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.agents import DocumentCourseBuilder
from src.llm import CerebrasLLM
from src.schemas import Concept


def extract_text(path: str | Path, *, max_pages: int | None = None) -> str:
    """Extract concatenated text from a PDF file."""
    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    parts: list[str] = []
    for page in pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - skip unreadable pages
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def build_course(
    llm: CerebrasLLM,
    text: str,
    *,
    max_concepts: int = 8,
) -> dict:
    """Return a seed-course-shaped dict built from document ``text``.

    Shape mirrors ``data/seed_course.json`` enough for ``CourseRunner.start``:
    ``{subject, chapter, concepts: [...], exercise_bank: []}``.
    """
    if not text.strip():
        raise ValueError(
            "No extractable text found in the document. Scanned/image-only PDFs "
            "are not supported (no OCR)."
        )

    builder = DocumentCourseBuilder(llm)
    data, _ = builder.run(text, max_concepts=max_concepts)

    subject = (data.get("subject") or "general").strip() or "general"
    chapter = (data.get("title") or "Uploaded Document").strip() or "Uploaded Document"

    concepts: list[dict] = []
    for order, c in enumerate(data.get("concepts", [])):
        concepts.append(
            {
                "id": f"doc-{order}",
                "title": c.get("title", f"Concept {order + 1}"),
                "summary": c.get("summary", ""),
                "prose": c.get("prose", ""),
                "chapter": chapter,
                "order": order,
                "seed_difficulty": 3,
            }
        )

    if not concepts:
        raise ValueError("Could not derive any concepts from the document.")

    return {
        "subject": subject,
        "chapter": chapter,
        "concepts": concepts,
        "exercise_bank": [],
        "sources": {
            "concepts": {"name": "Uploaded document", "license": "user-provided"}
        },
    }


def course_concepts(course: dict) -> list[Concept]:
    return [Concept.from_dict(c) for c in course["concepts"]]
