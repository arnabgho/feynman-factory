"""Persist explainer compositions and optionally render them to MP4.

The explainer HTML doubles as a HyperFrames composition. For the live app we
just serve/iframe the HTML (it self-plays, instant). For a downloadable
deliverable we can render an MP4 via the HyperFrames CLI in the background,
which is slow (headless Chrome + FFmpeg) and therefore kept off the request
path.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

COMPOSITIONS_DIR = Path("out/compositions")
RENDERS_DIR = Path("out/renders")


def save_composition(html: str, lesson_id: str) -> Path:
    """Write the composition HTML and return its file path.

    The file is named ``index.html`` inside a per-lesson directory so it is also
    a valid HyperFrames project entry point.
    """
    comp_dir = COMPOSITIONS_DIR / lesson_id
    comp_dir.mkdir(parents=True, exist_ok=True)
    path = comp_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def composition_url(lesson_id: str) -> str:
    """Relative URL the backend serves the composition under."""
    return f"/assets/compositions/{lesson_id}/index.html"


def _hyperframes_available() -> bool:
    return shutil.which("npx") is not None


def _render_blocking(comp_dir: Path, output: Path, *, quality: str = "draft") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "npx",
        "hyperframes",
        "render",
        "--non-interactive",
        "--output",
        str(output),
        "--quality",
        quality,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(comp_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
            print(f"[hyperframes] render failed for {comp_dir}: {proc.stderr[:300]}")
        else:
            print(f"[hyperframes] rendered {output} ({output.stat().st_size} bytes)")
    except Exception as exc:  # noqa: BLE001 - best-effort background task
        print(f"[hyperframes] render error for {comp_dir}: {exc}")


def render_mp4_async(lesson_id: str, *, quality: str = "draft") -> str | None:
    """Kick off a background MP4 render. Returns the expected URL or None.

    Best-effort: if the HyperFrames CLI is unavailable, this is a no-op.
    """
    if not _hyperframes_available():
        print("[hyperframes] npx not found; skipping MP4 render (live HTML still works).")
        return None

    comp_dir = COMPOSITIONS_DIR / lesson_id
    if not (comp_dir / "index.html").exists():
        print(f"[hyperframes] no composition at {comp_dir}; skipping render.")
        return None

    output = RENDERS_DIR / f"{lesson_id}.mp4"
    thread = threading.Thread(
        target=_render_blocking,
        args=(comp_dir, output),
        kwargs={"quality": quality},
        daemon=True,
    )
    thread.start()
    return f"/assets/renders/{lesson_id}.mp4"
