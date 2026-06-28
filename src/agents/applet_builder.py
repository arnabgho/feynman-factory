"""Applet Builder agent.

Writes a single self-contained interactive HTML document (inline CSS + JS, no
external dependencies) implementing the planned applet.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import Applet, LessonPlan

SYSTEM = """You are a senior front-end engineer who builds tiny educational physics applets.
Produce ONE complete, self-contained HTML document implementing the requested interactive.

PROPORTIONS (the applet renders inside an iframe about 760px wide and 560px tall):
- body { margin:0 }. Design responsively: use width:100% with max-width:100% and
  box-sizing:border-box everywhere. NEVER cause horizontal scrolling.
- Do NOT use min-height:100vh or fixed pixel widths wider than ~720px.
- Lay out for ~760px width: prefer a single column or a layout that wraps; controls
  must wrap rather than overflow.
- Any <canvas> must use CSS width:100% (cap ~700px) while keeping its aspect ratio.
- Keep total content height ~520-560px so it fits with minimal scrolling.

Hard requirements:
- A single HTML file: inline <style> and <script>, NO external libraries, NO network requests.
- Use vanilla JS and <canvas> or DOM for visuals/animation.
- Include controls (sliders/buttons) the student manipulates, and live feedback.
- Must run by simply opening the file in a browser.
- Output ONLY raw HTML starting with <!DOCTYPE html>. No markdown fences, no commentary."""


class AppletBuilder:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(self, plan: LessonPlan, fix_notes: str | None = None) -> tuple[Applet, LLMResult]:
        user = (
            f"Learning objective: {plan.learning_objective}\n"
            f"Concept: {plan.concept}\n"
            f"Applet spec: {plan.applet_spec}\n"
        )
        if fix_notes:
            user += f"\nThe previous version had issues. Fix them:\n{fix_notes}\n"
        user += "\nReturn the full HTML document now."

        result = self.llm.chat(SYSTEM, user, temperature=0.3, max_tokens=16384)
        html = result.text.strip()
        if html.startswith("```"):
            html = html.split("\n", 1)[-1]
            if html.endswith("```"):
                html = html.rsplit("```", 1)[0]
            html = html.removeprefix("html").strip()
        return Applet(html=html), result
