"""ExplainerVideo agent.

Produces a self-contained animated HTML explainer for a concept as a *seekable*
composition. The model exposes a deterministic frame function
(``window.player = { duration, render(t) }``); a host-injected runtime
(``src.video.runtime``) owns the clock, play/pause/seek/replay, and the
postMessage bridge to the React player UI.

The video is built from a pre-planned :class:`~src.schemas.SlideDeck` so longer
(~2 minute) videos stay reliable: each slide is a tightly scoped scene.
"""

from __future__ import annotations

from src.llm import CerebrasLLM, LLMResult
from src.schemas import LessonPlan, SlideDeck
from src.video import runtime

SYSTEM = """You are a senior motion-graphics engineer building a premium-quality animated explainer VIDEO as a SINGLE self-contained HTML document.

You are given a SLIDE DECK outline (ordered slides, each with a kind, title,
bullets, a visual description, and a duration in seconds). Turn it into one
continuous, narrated-style explainer where each slide is an on-screen scene
shown for its time window, in order. The whole video is typically ~90-120s.

PLAYBACK CONTRACT (critical - the host owns the clock and the play/pause/seek controls):
- Expose exactly one global: window.player = { duration: <total seconds, number>, render: function(t) { ... } }.
  duration MUST equal the deck's total_duration (sum of slide durations).
- render(t) is a PURE FUNCTION OF TIME t (seconds, 0..duration): it shows the
  slide whose [start, start+duration) window contains t, hides the others, and
  draws every visual (canvas, positions, opacity, bullet reveals) for that exact
  instant. Calling render(t) twice with the same t MUST look identical.
- Do NOT auto-play. Do NOT use setInterval, setTimeout, or requestAnimationFrame
  to advance time. Do NOT read Date.now() or performance.now(). The host drives
  time by calling window.player.render(t); it also scales and plays the stage.
- Compute each slide's start as the cumulative sum of prior durations. Drive all
  motion from local progress p = clamp((t - slideStart) / slideDuration, 0, 1):
  fade/slide slides in near p~0, reveal bullets progressively across p, animate
  diagrams as a function of p. Cross-fade briefly between adjacent slides.

STRUCTURE / HyperFrames compatibility:
- Wrap everything in:
  <div id="stage" data-composition-id="explainer" data-start="0" data-width="1280" data-height="720"> ... </div>
- Give each slide element class="clip" with data-start, data-duration, and
  data-track-index attributes matching its timing (seconds).

PROPORTIONS (the host scales the stage to fit, so sizing must be exact):
- #stage MUST be EXACTLY width:1280px; height:720px; position:relative. It is a
  fixed 1280x720 coordinate space - the entire visible canvas.
- body { margin:0 }. Put #stage at the top-left (do NOT center it in the body).
- Each slide should be a full-stage layer (position:absolute; inset:0) so slides
  stack and you toggle visibility by time. Do NOT use 100vw/100vh or vw/vh units.
- ALL content fits inside 1280x720 with ~64px safe margins; a slide's stacked
  content must total <= 720px tall. Size budget: slide title <= 56px; bullet text
  <= 30px; any <canvas> <= 920x380. Reuse one <canvas> per diagram slide.

DESIGN SYSTEM (make it look like a polished, modern explainer):
- Define a cohesive palette as CSS variables and use it consistently. Prefer a
  refined dark theme: a deep background with a subtle radial/linear gradient,
  one or two vivid accent colors used for key terms and highlights.
- Strong typographic hierarchy: a bold, confident title; lighter body captions;
  generous letter/line spacing. (The Inter font is available.)
- Present content on tasteful surfaces: rounded cards (border-radius ~16-20px),
  thin 1px translucent borders, soft layered shadows, comfortable padding.
- A persistent slide-progress affordance is welcome (e.g. small slide dots or a
  thin progress line), but keep it subtle.
- Smooth, eased motion: entrances use opacity + small translate with an ease like
  cubic-bezier(0.22, 1, 0.36, 1); nothing snaps or flickers.
- Crisp vector/graph visuals: clean anti-aliased strokes, labeled axes/vectors,
  clear legends, accent-colored emphasis. Use color to teach.
- Keep each slide uncluttered: lots of negative space, max ~2 focal elements.

Hard requirements:
- ONE HTML file. Inline <style> and <script> only. NO external scripts and NO
  network requests other than fonts. NO <img> with remote src.
- Render EVERY slide from the deck, in order, honoring each duration.
- Output ONLY raw HTML starting with <!DOCTYPE html>. No markdown fences, no commentary."""


def _format_deck(deck: SlideDeck) -> str:
    lines = [f'Deck title: "{deck.title}"', f"total_duration: {deck.total_duration:g}s", "Slides:"]
    start = 0.0
    for i, s in enumerate(deck.slides):
        bullets = "; ".join(s.bullets) if s.bullets else "(none)"
        lines.append(
            f"  {i + 1}. [{s.kind}] start={start:g}s dur={s.duration:g}s | "
            f'title="{s.title}" | bullets: {bullets} | visual: {s.visual or "none"}'
        )
        start += s.duration
    return "\n".join(lines)


class ExplainerVideo:
    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    def run(
        self,
        plan: LessonPlan,
        deck: SlideDeck,
        difficulty: int,
        *,
        subject: str = "physics",
        remix: str | None = None,
    ) -> tuple[str, LLMResult]:
        remix_line = (
            f"REMIX DIRECTIVE (reflect this in tone/wording/visuals): {remix}\n\n"
            if remix
            else ""
        )
        user = (
            f"Subject: {subject}\n"
            f"Concept: {plan.concept}\n"
            f"Learning objective: {plan.learning_objective}\n"
            f"Target difficulty (1-10): {difficulty}\n\n"
            f"{remix_line}"
            f"{_format_deck(deck)}\n\n"
            "Build the seekable slide-deck explainer document now. Define "
            "window.player = { duration, render(t) } with duration = "
            f"{deck.total_duration:g}, render every slide in order, and do not "
            "start any timers."
        )
        result = self.llm.chat(SYSTEM, user, temperature=0.4, max_tokens=32768)
        html = result.text.strip()
        if html.startswith("```"):
            html = html.split("\n", 1)[-1]
            if html.endswith("```"):
                html = html.rsplit("```", 1)[0]
            html = html.removeprefix("html").strip()
        return runtime.inject(html), result
