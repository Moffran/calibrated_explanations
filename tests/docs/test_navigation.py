from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _resolve_doc(slug: str) -> Path:
    slug = slug.lstrip("./")
    candidate = DOCS / slug
    if candidate.suffix:
        return candidate
    md_candidate = candidate.with_suffix(".md")
    if md_candidate.exists():
        return md_candidate
    rst_candidate = candidate.with_suffix(".rst")
    return rst_candidate


def test_top_level_toctree_targets_exist() -> None:
    expected = (
        "get-started/index",
        "practitioner/index",
        "researcher/index",
        "contributor/index",
        "foundations/index",
    )
    index_path = _resolve_doc("index")
    index = index_path.read_text()
    for slug in expected:
        assert slug in index, f"Missing toctree entry for {slug}"
        target = _resolve_doc(slug)
        assert target.exists(), f"Document referenced by {slug} is missing"

