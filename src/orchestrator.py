"""Orchestrator: runs the lesson-factory swarm with plain Python orchestration.

Two entry points:
  - ``LessonFactory``: the original Tutor -> Planner -> AppletBuilder -> Critic
    pipeline (used by ``src.demo``).
  - ``LessonGenerator``: course-aware generation that runs the ExplainerVideo and
    AppletBuilder agents concurrently, then validates the applet.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from src.agents import (
    AppletBuilder,
    Critic,
    ExplainerVideo,
    Planner,
    Tutor,
)
from src.llm import CerebrasLLM, LLMResult
from src.schemas import (
    Applet,
    Concept,
    Diagnosis,
    Exercise,
    Lesson,
    LessonState,
    StudentProfile,
)

# Callback invoked with structured progress events (for SSE streaming).
EventSink = Callable[[dict], None]


class LessonFactory:
    def __init__(self, llm: CerebrasLLM | None = None, max_build_iters: int = 2):
        self.llm = llm or CerebrasLLM()
        self.tutor = Tutor(self.llm)
        self.planner = Planner(self.llm)
        self.builder = AppletBuilder(self.llm)
        self.critic = Critic(self.llm)
        self.max_build_iters = max_build_iters

    def _log(self, state: LessonState, agent: str, summary: str, timing) -> None:
        state.timings[agent] = state.timings.get(agent, 0.0) + timing.total_s
        state.trace.append(
            {
                "agent": agent,
                "summary": summary,
                "ttft_s": round(timing.ttft_s, 3),
                "total_s": round(timing.total_s, 3),
            }
        )
        print(
            f"  [{agent:14}] ttft={timing.ttft_s:5.3f}s total={timing.total_s:5.3f}s  {summary}"
        )

    def run(self, student: StudentProfile) -> LessonState:
        state = LessonState(student=student)
        print(f"\n=== Generating lesson for {student.name} ({student.grade_level}) ===")

        diagnosis, t = self.tutor.run(student)
        state.diagnosis = diagnosis
        self._log(state, "Tutor", f"-> {diagnosis.next_concept} (diff {diagnosis.difficulty})", t)

        plan, t = self.planner.run(diagnosis, student)
        state.plan = plan
        self._log(state, "Planner", f"objective: {plan.learning_objective[:60]}", t)

        applet: Applet | None = None
        fix_notes: str | None = None
        for i in range(1, self.max_build_iters + 1):
            applet, t = self.builder.run(plan, fix_notes=fix_notes)
            self._log(state, "AppletBuilder", f"pass {i}: {len(applet.html)} chars HTML", t)

            applet, t = self.critic.run(plan, applet)
            verdict = "approved" if applet.approved else f"needs work: {applet.critique}"
            self._log(state, "Critic", f"pass {i}: {verdict[:60]}", t)

            if applet.approved:
                break
            fix_notes = applet.critique

        state.applet = applet
        total = sum(state.timings.values())
        print(f"=== Done. Total model time: {total:.3f}s across {len(state.trace)} calls ===")
        return state


def _strip_html_fence(text: str) -> str:
    html = text.strip()
    if html.startswith("```"):
        html = html.split("\n", 1)[-1]
        if html.endswith("```"):
            html = html.rsplit("```", 1)[0]
        html = html.removeprefix("html").strip()
    return html


class LessonGenerator:
    """Course-aware lesson generation: plan, then build video + applet in parallel.

    Emits progress events via an optional ``event_sink`` so a server can stream
    per-agent latency to the UI.
    """

    def __init__(self, llm: CerebrasLLM | None = None, max_build_iters: int = 2):
        self.llm = llm or CerebrasLLM()
        self.planner = Planner(self.llm)
        self.video = ExplainerVideo(self.llm)
        self.builder = AppletBuilder(self.llm)
        self.critic = Critic(self.llm)
        self.max_build_iters = max_build_iters

    @staticmethod
    def _emit(sink: EventSink | None, agent: str, status: str, timing: LLMResult | None = None, **extra):
        if sink is None:
            return
        event = {"agent": agent, "status": status, **extra}
        if timing is not None:
            event["ttft_s"] = round(timing.ttft_s, 3)
            event["total_s"] = round(timing.total_s, 3)
        sink(event)

    def generate(
        self,
        concept: Concept,
        difficulty: int,
        exercise: Exercise | None,
        student: StudentProfile,
        *,
        variant: str = "base",
        misconception: str | None = None,
        event_sink: EventSink | None = None,
    ) -> Lesson:
        diagnosis = Diagnosis(
            next_concept=concept.title,
            difficulty=difficulty,
            misconception=misconception,
            rationale=f"Course concept {concept.order}: {concept.title}",
        )

        self._emit(event_sink, "Planner", "start")
        plan, t_plan = self.planner.run(diagnosis, student, grounding=concept.prose)
        self._emit(event_sink, "Planner", "done", t_plan, summary=plan.learning_objective)

        # Build the explainer video and the interactive applet concurrently.
        self._emit(event_sink, "ExplainerVideo", "start")
        self._emit(event_sink, "AppletBuilder", "start")
        with ThreadPoolExecutor(max_workers=2) as pool:
            video_future = pool.submit(self.video.run, plan, difficulty)
            applet_future = pool.submit(self.builder.run, plan)
            video_html, t_video = video_future.result()
            applet, t_applet = applet_future.result()

        video_html = _strip_html_fence(video_html)
        self._emit(event_sink, "ExplainerVideo", "done", t_video, chars=len(video_html))
        self._emit(event_sink, "AppletBuilder", "done", t_applet, chars=len(applet.html))

        # Validate (and if needed, repair) the applet.
        applet, t_critic = self.critic.run(plan, applet)
        self._emit(
            event_sink,
            "Critic",
            "done",
            t_critic,
            approved=applet.approved,
        )
        if not applet.approved and self.max_build_iters > 1:
            self._emit(event_sink, "AppletBuilder", "start", note="repair")
            applet, t_fix = self.builder.run(plan, fix_notes=applet.critique)
            self._emit(event_sink, "AppletBuilder", "done", t_fix, chars=len(applet.html))
            applet, t_critic2 = self.critic.run(plan, applet)
            self._emit(event_sink, "Critic", "done", t_critic2, approved=applet.approved)

        return Lesson(
            concept=concept,
            difficulty=difficulty,
            plan=plan,
            video_composition=video_html,
            applet=applet,
            exercise=exercise,
            variant=variant,  # type: ignore[arg-type]
        )
