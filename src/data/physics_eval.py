"""Load the PhysicsEval dataset (IUTVanguard/PhysicsEval) into an exercise bank.

Fields used: ``problem``, ``simplified_problem_statement``, ``category``,
``problem_difficulty`` (1-10), ``elaborated_solution_steps``,
``final_answers_in_brief``, ``soft_labels``.
"""

from __future__ import annotations

from typing import Any

DATASET_ID = "IUTVanguard/PhysicsEval"

# Map OpenStax chapter titles to PhysicsEval categories we want to draw from.
CHAPTER_CATEGORY_HINTS = {
    "motion in one dimension": ["mechanics", "kinematics"],
    "acceleration": ["mechanics", "kinematics"],
    "forces and newton's laws of motion": ["mechanics", "dynamics"],
    "motion in two dimensions": ["mechanics", "kinematics"],
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    soft = row.get("soft_labels")
    if isinstance(soft, str):
        labels = [s.strip().lower() for s in soft.replace(";", ",").split(",") if s.strip()]
    elif isinstance(soft, (list, tuple)):
        labels = [str(s).strip().lower() for s in soft]
    else:
        labels = []
    try:
        difficulty = int(row.get("problem_difficulty") or 0)
    except (TypeError, ValueError):
        difficulty = 0
    return {
        "id": str(row.get("Problem_ID", "")),
        "problem": _as_text(row.get("problem")),
        "simplified": _as_text(row.get("simplified_problem_statement")),
        "category": _as_text(row.get("category")).strip(),
        "difficulty": difficulty,
        "solution_steps": _as_text(row.get("elaborated_solution_steps")),
        "final_answer": _as_text(row.get("final_answers_in_brief")),
        "soft_labels": labels,
    }


def load_exercise_bank(
    *,
    split: str = "train",
    categories: list[str] | None = None,
    prefer_labels: tuple[str, ...] = ("conceptual", "diagram"),
    max_rows: int = 4000,
) -> list[dict[str, Any]]:
    """Load and filter PhysicsEval into a list of normalized exercise dicts.

    ``categories`` are matched case-insensitively as substrings against the
    dataset ``category`` field. ``max_rows`` bounds how much we scan for speed.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, streaming=True)

    cats = [c.lower() for c in (categories or [])]
    bank: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        ex = _normalize(row)
        if not ex["problem"] or ex["difficulty"] <= 0:
            continue
        if cats and not any(c in ex["category"].lower() for c in cats):
            continue
        bank.append(ex)

    # Sort so preferred (conceptual/diagram) exercises come first, then by difficulty.
    def sort_key(ex: dict[str, Any]):
        has_pref = any(lbl in ex["soft_labels"] for lbl in prefer_labels)
        return (0 if has_pref else 1, ex["difficulty"])

    bank.sort(key=sort_key)
    return bank


def categories_for_chapter(chapter_title: str) -> list[str]:
    return CHAPTER_CATEGORY_HINTS.get(chapter_title.strip().lower(), ["mechanics"])
