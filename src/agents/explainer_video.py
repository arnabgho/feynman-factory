"""ExplainerVideo agent.

Produces a self-contained animated HTML explainer for a concept as a *seekable*
composition. The model exposes a deterministic frame function
(``window.player = { duration, render(t) }``); a host-injected runtime
(``src.video.runtime``) owns the clock, play/pause/seek/replay, and the
postMessage bridge to the React player UI.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import LessonPlan
from src.video import runtime

SYSTEM = """You are a senior motion-graphics engineer building a short, premium-quality animated physics explainer as a SINGLE self-contained HTML document.

PLAYBACK CONTRACT (critical - the host owns the clock and the play/pause/seek controls):
- Expose exactly one global: window.player = { duration: <total seconds, number>, render: function(t) { ... } }.
- render(t) is a PURE FUNCTION OF TIME t (seconds, 0..duration): it sets which
  scene is visible and draws every visual (canvas, positions, opacity) for that
  exact instant. Calling render(t) twice with the same t MUST look identical.
- Do NOT auto-play. Do NOT use setInterval, setTimeout, or requestAnimationFrame
  to advance time. Do NOT read Date.now() or performance.now(). The host drives
  time by calling window.player.render(t); it also scales and plays the stage.
- Drive scene visibility from t (e.g., compute the active scene from per-scene
  start/duration and toggle opacity), and compute all motion as a function of t
  (e.g., progress = (t - sceneStart) / sceneDuration, clamped 0..1).
- Aim for a total duration of ~24-32 seconds, one scene per explanation beat,
  plus a short recap scene.

STRUCTURE / HyperFrames compatibility:
- Wrap everything in:
  <div id="stage" data-composition-id="explainer" data-start="0" data-width="1280" data-height="720"> ... </div>
- Give each scene element class="clip" with data-start, data-duration, and
  data-track-index attributes matching its timing (seconds).

PROPORTIONS (the host scales the stage to fit, so sizing must be exact):
- #stage MUST be EXACTLY width:1280px; height:720px; position:relative. It is a
  fixed 1280x720 coordinate space - the entire visible canvas.
- body { margin:0 }. Put #stage at the top-left (do NOT center it in the body).
- Do NOT use 100vw/100vh or vw/vh units. ALL content must fit inside 1280x720
  with ~64px safe margins; a scene's stacked content must total <= 720px tall.
- Size budget: title font-size <= 64px; caption font-size <= 30px, max-width ~1000px;
  any <canvas> <= 920x380.

DESIGN SYSTEM (make it look like a polished, modern explainer):
- Define a cohesive palette as CSS variables and use it consistently. Prefer a
  refined dark theme: a deep background with a subtle radial/linear gradient,
  one or two vivid accent colors used for key terms and highlights.
- Strong typographic hierarchy: a bold, confident title; lighter body captions;
  generous letter/line spacing. (The Inter font is available.)
- Present content on tasteful surfaces: rounded cards (border-radius ~16-20px),
  thin 1px translucent borders, soft layered shadows, comfortable padding.
- Smooth, eased motion: entrances use opacity + small translate with an ease like
  cubic-bezier(0.22, 1, 0.36, 1); nothing snaps or flickers.
- Crisp vector/graph visuals: clean anti-aliased strokes, labeled axes/vectors,
  clear legends, accent-colored emphasis. Use color to teach (e.g., distance vs
  displacement in two distinct accents).
- Keep it uncluttered: lots of negative space, max ~2 focal elements per scene.

Hard requirements:
- ONE HTML file. Inline <style> and <script> only. NO external scripts and NO
  network requests other than fonts. NO <img> with remote src.
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
            "Build the seekable explainer document now. Remember: define "
            "window.player = { duration, render(t) } and do not start any timers."
        )
        result = self.llm.chat(SYSTEM, user, temperature=0.4, max_tokens=16384)
        html = result.text.strip()
        if html.startswith("```"):
            html = html.split("\n", 1)[-1]
            if html.endswith("```"):
                html = html.rsplit("```", 1)[0]
            html = html.removeprefix("html").strip()
        return runtime.inject(html), result
