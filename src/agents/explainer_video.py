"""ExplainerVideo agent.

Produces a self-contained animated HTML explainer for a concept. The document
doubles as:
  1. A standalone "video" that auto-plays timed scenes in an iframe (instant,
     no render) for the live app.
  2. A HyperFrames composition (carries ``data-composition-id`` / ``data-start``
     / ``data-duration`` / ``data-track-index`` and a seekable GSAP timeline at
     ``window.__timelines[id]``) so it can be rendered to MP4 offline.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import LessonPlan

SYSTEM = """You are a motion-graphics engineer who builds short animated physics explainer videos as a SINGLE self-contained HTML document.

The document must satisfy BOTH of these at once:

1) It auto-plays in a browser iframe with NO user action: a sequence of timed
   scenes (one per explanation beat) that fade/slide in, hold, and advance
   automatically, ending with a short recap. Use CSS animations and/or vanilla
   JS timers. Include a simple animated visual relevant to the concept
   (e.g., a moving dot on an axis, vectors, a position/velocity graph drawn on
   <canvas> or SVG). Keep total runtime ~20-35 seconds.

2) It is also a HyperFrames composition. Wrap everything in:
   <div id="stage" data-composition-id="explainer" data-start="0" data-width="1280" data-height="720"> ... </div>
   Give each scene element class="clip" with data-start, data-duration, and
   data-track-index attributes matching its on-screen timing (seconds).

Hard requirements:
- ONE HTML file. Inline <style> and <script> only. NO external libraries, NO
  network requests, NO <img> with remote src.
- Visually clean: large readable captions, a clear title card, good contrast,
  a calm color palette. 1280x720 stage.
- Output ONLY raw HTML starting with <!DOCTYPE html>. No markdown fences, no commentary."""


class ExplainerVideo:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(self, plan: LessonPlan, difficulty: int) -> tuple[str, LLMResult]:
        beats = "\n".join(f"- {b}" for b in plan.explanation_beats)
        user = (
            f"Concept: {plan.concept}\n"
            f"Learning objective: {plan.learning_objective}\n"
            f"Target difficulty (1-10): {difficulty}\n"
            f"Explanation beats (one scene each):\n{beats}\n\n"
            "Build the animated explainer document now."
        )
        result = self.llm.chat(SYSTEM, user, temperature=0.4, max_tokens=16384)
        html = result.text.strip()
        if html.startswith("```"):
            html = html.split("\n", 1)[-1]
            if html.endswith("```"):
                html = html.rsplit("```", 1)[0]
            html = html.removeprefix("html").strip()
        return html, result
