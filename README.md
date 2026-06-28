# Lesson Factory

A multi-agent personalized physics lesson factory powered by **Cerebras** ultra-fast inference and **Gemma 4 31B** (multimodal).

Built for the Cerebras x Google DeepMind Gemma 4 Hackathon — Track 1: Multiverse Agents.

## Idea

An agent swarm generates a bespoke micro-lesson per student in seconds: a concept explanation plus an interactive HTML applet to practice. When the student attempts the exercise, Gemma 4 vision reads their work and the swarm regenerates the next step — a closed adaptive loop made interactive by Cerebras speed.

Each lesson is an explainer **video** plus an interactive **exercise**. The
student answers; the Grader evaluates it (text and/or a photo of their work) and
the course either **advances** to the next concept, gives a **harder** variant,
or **remediates** the current concept.

## Agents

- **Planner** — turns a concept into a micro-lesson outline + an applet spec.
- **ExplainerVideo** — writes a self-contained animated explainer (also a
  HyperFrames composition, so it can be rendered to MP4 offline).
- **Applet Builder** — writes a self-contained interactive HTML/JS exercise.
- **Critic / Validator** — checks correctness and self-containment, repairs the
  applet before the student sees it.
- **Grader** — evaluates the student's answer (with vision) and estimates mastery.
- **Tutor / Diagnostician** — initial placement / diagnosis (used by `src.demo`).

`ExplainerVideo` and `AppletBuilder` run **concurrently** to exploit Cerebras speed.

## Architecture

```
data/         OpenStax + PhysicsEval ingestion -> data/seed_course.json
src/llm.py    Cerebras wrapper (text, json_schema structured outputs, vision, timing)
src/agents/   Planner, ExplainerVideo, AppletBuilder, Critic, Grader, Tutor
src/orchestrator.py   LessonGenerator (parallel video + applet)
src/course.py         CourseRunner: exercise selection + mastery loop
src/video/            HyperFrames composition writer + background MP4 render
src/bench.py          Cerebras vs GPU baseline latency harness
app/                  FastAPI streaming backend (SSE agent events)
web/index.html        Single-file React UI (loaded via ESM CDN, no build step)
```

## Setup

```bash
uv sync
cp .env.example .env            # add your CEREBRAS_API_KEY
```

## Build the seed course

```bash
uv run python -m src.data.build_seed --chapter "Motion in One Dimension"
```

## Run the app

```bash
uv run uvicorn app.main:app --port 8000
# open http://localhost:8000
```

## Other entry points

```bash
uv run python -m src.demo            # single-lesson swarm demo (terminal)
uv run python -m src.bench --runs 3  # latency benchmark (writes out/bench.json)
```

## Requirements

- Python 3.12+ and a Cerebras API key (`CEREBRAS_API_KEY` in `.env`).
- The web UI loads React via an ESM CDN, so **no Node/npm is required** to run it.
- Optional: Node.js + FFmpeg only if you want offline MP4 renders of the
  explainer videos via the HyperFrames CLI (`npx hyperframes`). The live app
  plays the explainer HTML directly, so this is not needed for the demo.

## Data licensing

- Lessons are seeded from **OpenStax *Physics*** (CC BY 4.0).
- Exercises are drawn from the **PhysicsEval** dataset.
