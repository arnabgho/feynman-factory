"""Build ``data/seed_course.json`` from OpenStax concepts + PhysicsEval exercises.

Usage:
    uv run python -m src.data.build_seed --chapter "Motion in One Dimension"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data import openstax, physics_eval

OUT_PATH = Path("data/seed_course.json")

SOURCE_META = {
    "concepts": {
        "name": "OpenStax Physics",
        "url": "https://github.com/openstax/osbooks-physics",
        "license": "CC BY 4.0",
        "attribution": "Physics, OpenStax (Rice University), CC BY 4.0",
    },
    "exercises": {
        "name": "PhysicsEval",
        "url": "https://huggingface.co/datasets/IUTVanguard/PhysicsEval",
    },
}


def _seed_difficulty(order: int, total: int) -> int:
    """Ramp base difficulty roughly 2 -> 6 across the chapter."""
    if total <= 1:
        return 3
    return 2 + round((order / (total - 1)) * 4)


def build(
    chapter: str = openstax.DEFAULT_CHAPTER,
    *,
    max_concepts: int | None = 6,
    max_exercise_rows: int = 4000,
) -> dict[str, Any]:
    concepts = openstax.get_chapter_concepts(chapter, max_concepts=max_concepts)
    total = len(concepts)
    for c in concepts:
        c["seed_difficulty"] = _seed_difficulty(c["order"], total)

    categories = physics_eval.categories_for_chapter(chapter)
    bank = physics_eval.load_exercise_bank(
        categories=categories, max_rows=max_exercise_rows
    )

    return {
        "chapter": chapter,
        "sources": SOURCE_META,
        "concepts": concepts,
        "exercise_bank": bank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the seed course JSON.")
    parser.add_argument("--chapter", default=openstax.DEFAULT_CHAPTER)
    parser.add_argument("--max-concepts", type=int, default=6)
    parser.add_argument("--max-exercise-rows", type=int, default=4000)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    course = build(
        args.chapter,
        max_concepts=args.max_concepts,
        max_exercise_rows=args.max_exercise_rows,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(course, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out}")
    print(f"  chapter:       {course['chapter']}")
    print(f"  concepts:      {len(course['concepts'])}")
    for c in course["concepts"]:
        print(f"    [{c['order']}] {c['title']} (seed diff {c['seed_difficulty']})")
    print(f"  exercise_bank: {len(course['exercise_bank'])} problems")


if __name__ == "__main__":
    main()
