"""DocumentCourseBuilder agent.

Turns the extracted text of an uploaded document (a lesson handout or a short
paper, any subject) into an ordered set of teachable concepts. Each concept
carries a short summary and a grounding ``prose`` passage condensed from the
document, which the Planner later uses to ground the lesson.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult

SYSTEM = """You are a subject-agnostic instructional designer.

You are given the raw text of a document (a lesson, handout, or short paper).
Identify its subject and break it into an ordered sequence of 3-8 teachable
concepts that a student should learn, in a sensible teaching order.

For each concept provide:
- title: a short concept name.
- summary: one sentence describing what the student will learn.
- prose: a self-contained 2-4 sentence explanation GROUNDED IN THE DOCUMENT
  (condense/paraphrase the relevant part of the source; do not invent facts that
  contradict it). This is the teaching material for that concept.

Also return the document's overall subject (e.g. "physics", "biology",
"machine learning", "history") and a concise course title.
Only use information supported by the document."""

SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "title": {"type": "string"},
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "prose": {"type": "string"},
                },
                "required": ["title", "summary", "prose"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subject", "title", "concepts"],
    "additionalProperties": False,
}

# How much document text to feed the model (small papers fit comfortably).
MAX_DOC_CHARS = 12000


class DocumentCourseBuilder:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(self, text: str, *, max_concepts: int = 8) -> tuple[dict, LLMResult]:
        excerpt = text.strip()[:MAX_DOC_CHARS]
        user = (
            f"Build a course of at most {max_concepts} concepts from this "
            "document. Return ordered concepts grounded in the text.\n\n"
            f"--- DOCUMENT START ---\n{excerpt}\n--- DOCUMENT END ---"
        )
        result = self.llm.chat(
            SYSTEM, user, schema=SCHEMA, schema_name="doc_course", temperature=0.3
        )
        data = result.as_json()
        if isinstance(data.get("concepts"), list):
            data["concepts"] = data["concepts"][:max_concepts]
        return data, result
