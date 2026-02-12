from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.parity_reference.run_parity_reference import (
    build_payload,
    compute_outputs,
    dump_json,
    load_json,
)


def _contains_is_conjunctive_true(obj: Any) -> bool:
    if isinstance(obj, dict):
        if obj.get("is_conjunctive") is True:
            return True
        return any(_contains_is_conjunctive_true(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_is_conjunctive_true(v) for v in obj)
    return False


def test_build_payload__should_prefer_conjunctive_rules_when_present() -> None:
    class DummyExp:
        has_conjunctive_rules = True
        conjunctive_rules = {"rule": ["x & y"], "feature": [[0, 1]]}
        rules = {"rule": ["x"], "feature": [0]}

        def get_mode(self) -> str:
            return "classification"

    payload = build_payload(DummyExp())

    assert payload["rules"] is DummyExp.conjunctive_rules


def test_compute_outputs__should_include_is_conjunctive_true_in_json_payload() -> None:
    dataset = load_json(Path("tests/parity_reference/canonical_dataset.json"))

    outputs = compute_outputs(dataset, condition_source="observed")

    assert _contains_is_conjunctive_true(outputs["factual"])
    assert _contains_is_conjunctive_true(outputs["alternatives"])


def test_dump_json__should_write_conjunctive_rules_to_json_files(tmp_path: Path) -> None:
    dataset = load_json(Path("tests/parity_reference/canonical_dataset.json"))
    outputs = compute_outputs(dataset, condition_source="observed")

    factual_path = tmp_path / "factual.json"
    dump_json(factual_path, outputs["factual"])

    factual_payload = load_json(factual_path)
    assert _contains_is_conjunctive_true(factual_payload)
