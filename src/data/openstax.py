"""Fetch and parse the OpenStax *Physics* book (CC BY 4.0) into ordered concepts.

The book lives at github.com/openstax/osbooks-physics as CNXML:
- ``collections/physics.collection.xml`` defines chapter (subcollection) order
  and the modules within each chapter.
- ``modules/<id>/index.cnxml`` holds each module's title and prose.

We parse one chapter (default: "Motion in One Dimension") into a list of
concept dicts that downstream code (``build_seed``) turns into a course.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests
from lxml import etree

RAW_BASE = "https://raw.githubusercontent.com/openstax/osbooks-physics/main"
COLLECTION_URL = f"{RAW_BASE}/collections/physics.collection.xml"
MODULE_URL = RAW_BASE + "/modules/{module_id}/index.cnxml"

NS = {
    "col": "http://cnx.rice.edu/collxml",
    "md": "http://cnx.rice.edu/mdml",
    "c": "http://cnx.rice.edu/cnxml",
}

CACHE_DIR = Path("data/raw")
DEFAULT_CHAPTER = "Motion in One Dimension"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _fetch(url: str, cache_name: str) -> str:
    """Fetch a URL with a simple on-disk cache."""
    cached = _cache_path(cache_name)
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(resp.text, encoding="utf-8")
    return resp.text


def fetch_collection() -> list[dict[str, Any]]:
    """Return ordered chapters: ``[{"title": str, "module_ids": [str, ...]}]``."""
    xml = _fetch(COLLECTION_URL, "physics.collection.xml")
    root = etree.fromstring(xml.encode("utf-8"))

    chapters: list[dict[str, Any]] = []
    for subcol in root.iter(f"{{{NS['col']}}}subcollection"):
        title_el = subcol.find(f"{{{NS['md']}}}title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        module_ids = [
            m.get("document")
            for m in subcol.iter(f"{{{NS['col']}}}module")
            if m.get("document")
        ]
        if title and module_ids:
            chapters.append({"title": title, "module_ids": module_ids})
    return chapters


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_module(module_id: str) -> dict[str, Any]:
    """Parse a module's CNXML into ``{id, title, summary, prose}``."""
    xml = _fetch(MODULE_URL.format(module_id=module_id), f"modules/{module_id}.cnxml")
    root = etree.fromstring(xml.encode("utf-8"))

    md_title = root.find(f".//{{{NS['md']}}}title")
    doc_title = root.find(f"{{{NS['c']}}}title")
    title = ""
    for el in (md_title, doc_title):
        if el is not None and (el.text or "").strip():
            title = el.text.strip()
            break

    content = root.find(f"{{{NS['c']}}}content")
    paragraphs: list[str] = []
    if content is not None:
        for para in content.iter(f"{{{NS['c']}}}para"):
            # Skip teacher-support / answer notes; keep student-facing prose.
            ancestor_classes = {
                (a.get("class") or "") for a in para.iterancestors()
            }
            if any("os-teacher" in c for c in ancestor_classes):
                continue
            text = _clean_text("".join(para.itertext()))
            if len(text) > 40:
                paragraphs.append(text)

    prose = "\n\n".join(paragraphs)
    summary = paragraphs[0] if paragraphs else ""
    return {
        "id": module_id,
        "title": title or module_id,
        "summary": summary,
        "prose": prose,
    }


def get_chapter_concepts(
    chapter_title: str = DEFAULT_CHAPTER,
    *,
    max_concepts: int | None = None,
    skip_intro: bool = True,
) -> list[dict[str, Any]]:
    """Return ordered concept dicts for a chapter, enriched with ordering."""
    chapters = fetch_collection()
    match = next(
        (c for c in chapters if c["title"].lower() == chapter_title.lower()), None
    )
    if match is None:
        available = ", ".join(c["title"] for c in chapters)
        raise ValueError(
            f"Chapter {chapter_title!r} not found. Available: {available}"
        )

    concepts: list[dict[str, Any]] = []
    for module_id in match["module_ids"]:
        concept = parse_module(module_id)
        concept["chapter"] = match["title"]
        if skip_intro and concept["title"].strip().lower() == "introduction":
            continue
        if not concept["prose"]:
            continue
        concepts.append(concept)

    if max_concepts is not None:
        concepts = concepts[:max_concepts]

    for order, concept in enumerate(concepts):
        concept["order"] = order
    return concepts
