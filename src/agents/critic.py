"""Critic / Validator agent.

Reviews the generated applet for physics correctness and self-containment.
Returns an approval flag plus actionable fix notes for another build pass.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import Applet, LessonPlan

SYSTEM = """You are a meticulous reviewer of educational physics applets.
You are given a lesson plan and an HTML applet. Check:
1. Physics correctness (formulas, units, behavior).
2. Self-containment (no external scripts/styles/network calls).
3. Whether it actually fulfills the applet spec and is interactive.

Report whether it is approved, list any issues (empty if approved), and give
concise fix_notes (null if approved)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "fix_notes": {"type": ["string", "null"]},
    },
    "required": ["approved", "issues", "fix_notes"],
    "additionalProperties": False,
}


class Critic:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(self, plan: LessonPlan, applet: Applet) -> tuple[Applet, LLMResult]:
        # Keep token budget sane: critique the head + a slice of the body.
        snippet = applet.html if len(applet.html) < 12000 else applet.html[:12000]
        user = (
            f"Applet spec: {plan.applet_spec}\n\n"
            f"HTML to review:\n{snippet}\n\n"
            "Review it and respond with the JSON verdict."
        )
        result = self.llm.chat(
            SYSTEM, user, schema=SCHEMA, schema_name="critique", temperature=0.1
        )
        data = result.as_json()
        applet.approved = bool(data["approved"])
        applet.critique = data.get("fix_notes")
        return applet, result
