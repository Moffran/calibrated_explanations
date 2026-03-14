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


def test_reject_policy_doc_contract_sections_exist() -> None:
    reject_doc = resolve_doc("practitioner/advanced/reject-policy")
    content = reject_doc.read_text(encoding="utf-8")

    required_snippets = (
        "## Reject-aware return types",
        "`predict` / `predict_proba`: returns `RejectResult`",
        "### Threshold tie behavior",
        "`y_cal < threshold`",
        "eps = meta.get(\"epsilon\")  # scalar float",
        "if len(result.explanations) == 0:",
        'result.metadata["source_indices"]',
    )

    for snippet in required_snippets:
        assert snippet in content, f"Missing reject-policy contract snippet: {snippet}"
