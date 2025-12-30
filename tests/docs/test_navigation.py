from __future__ import annotations

from pathlib import Path

from tests.helpers.doc_utils import resolve_doc

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_top_level_toctree_targets_exist() -> None:
    expected = (
        "get-started/index",
        "practitioner/index",
        "researcher/index",
        "contributor/index",
        "foundations/index",
    )
    index_path = resolve_doc("index")
    index = index_path.read_text()
    for slug in expected:
        assert slug in index, f"Missing toctree entry for {slug}"
        target = resolve_doc(slug)
        assert target.exists(), f"Document referenced by {slug} is missing"


def test_maintenance_docs_exist() -> None:
    expected = [
        "maintenance/legacy-plotting-reference.md",
    ]
    for path in expected:
        assert (DOCS / path).exists(), f"Maintenance document {path} is missing"
