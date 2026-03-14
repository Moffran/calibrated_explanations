from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.reject import (
    scenario_a_policy_matrix as scenario_a,
    scenario_d_regression_threshold as scenario_d,
    scenario_e_binary_coverage_sweep as scenario_e,
    scenario_f_multiclass_coverage as scenario_f,
    scenario_g_regression_coverage as scenario_g,
    scenario_h_ncf_grid as scenario_h,
    scenario_k_mondrian_regression as scenario_k,
)
from evaluation.reject.common_reject import (
    RunConfig,
    breakdown_from_reject_output,
    build_regression_bundle,
    load_binary_datasets,
    load_multiclass_datasets,
    load_regression_datasets,
    reject_breakdown,
    task_specs,
)


def _normalize_meta(meta: dict[str, object]) -> dict[str, object]:
    """Convert numpy scalars/arrays to JSON-ready Python values for comparisons."""
    normalized: dict[str, object] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        elif isinstance(value, np.floating):
            normalized[key] = float(value)
        elif isinstance(value, np.integer):
            normalized[key] = int(value)
        elif isinstance(value, np.bool_):
            normalized[key] = bool(value)
        elif isinstance(value, dict):
            normalized[key] = _normalize_meta(value)
        elif isinstance(value, list):
            out: list[object] = []
            for item in value:
                if isinstance(item, np.ndarray):
                    out.append(item.tolist())
                elif isinstance(item, np.floating):
                    out.append(float(item))
                elif isinstance(item, np.integer):
                    out.append(int(item))
                elif isinstance(item, np.bool_):
                    out.append(bool(item))
                else:
                    out.append(item)
            normalized[key] = out
        else:
            normalized[key] = value
    return normalized


def test_should_load_registered_datasets_when_quick_registry_requested() -> None:
    binary = load_binary_datasets(quick=True)
    multiclass = load_multiclass_datasets(quick=True)
    regression = load_regression_datasets(quick=True)

    assert len(binary) >= 2
    assert len(multiclass) >= 2
    assert len(regression) >= 2
    assert {name for name, _, _ in binary} >= {"breast_cancer", "colic"}
    assert {name for name, _, _ in multiclass} >= {"balance", "iris"}
    assert {name for name, _, _ in regression} >= {"diabetes_reg", "abalone"}


def test_should_build_deterministic_regression_bundle_when_seed_is_reused() -> None:
    spec = task_specs("regression", quick=True)[0]

    first = build_regression_bundle(spec, RunConfig(seed=42, quick=True))
    second = build_regression_bundle(spec, RunConfig(seed=42, quick=True))

    assert np.allclose(first.x_test, second.x_test)
    assert np.allclose(first.y_test, second.y_test)
    assert np.allclose(first.baseline_pred, second.baseline_pred)
    assert first.target_scale == second.target_scale


def test_should_emit_binary_coverage_artifact_shape_when_running_scenario_e(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_e,
        "task_specs",
        lambda task_type, quick=False: [task_specs(task_type, quick=True)[0]],
    )
    monkeypatch.setattr(
        scenario_e,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )

    scenario_e.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "scenario_e_binary_coverage"
    assert {"coverage", "lower_ci", "upper_ci", "accepted_accuracy_empirical"} <= set(table.columns)


def test_should_mark_multiclass_results_empirical_when_running_scenario_f(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_f,
        "task_specs",
        lambda task_type, quick=False: [
            spec for spec in task_specs(task_type, quick=True) if spec.name == "iris"
        ],
    )
    monkeypatch.setattr(
        scenario_f,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )

    scenario_f.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "scenario_f_multiclass_coverage"
    assert {"accepted_top1_accuracy", "guarantee_status"} <= set(table.columns)
    assert set(table["guarantee_status"]) == {"empirical"}


def test_should_emit_empirical_columns_when_running_scenario_g(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_g,
        "task_specs",
        lambda task_type, quick=False: [task_specs(task_type, quick=True)[0]],
    )
    monkeypatch.setattr(
        scenario_g,
        "quantile_grid",
        lambda quick: np.asarray([0.10]),
    )
    monkeypatch.setattr(
        scenario_g,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )

    scenario_g.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "scenario_g_regression_coverage"
    assert {
        "accepted_coverage_empirical",
        "accepted_interval_width_empirical",
        "accepted_mse_empirical",
    } <= set(table.columns)


def test_should_emit_policy_effective_semantics_columns_when_running_scenario_a(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_a,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )
    monkeypatch.setattr(scenario_a, "confidence_grid", lambda quick, start=0.80, stop=0.99: np.asarray([0.9]))

    scenario_a.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "policy_matrix"
    assert {"effective_confidence", "policy", "confidence", "reject_rate"} <= set(table.columns)
    assert np.allclose(table["effective_confidence"], table["confidence"], equal_nan=False)


def test_should_emit_threshold_effective_semantics_columns_when_running_scenario_d(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_d,
        "quantile_grid",
        lambda quick: np.asarray([0.10]),
    )
    monkeypatch.setattr(
        scenario_d,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )

    scenario_d.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "regression_threshold"
    assert {
        "effective_threshold",
        "effective_confidence",
        "threshold_source",
        "threshold_value",
    } <= set(table.columns)
    assert np.allclose(
        table["effective_threshold"].astype(float),
        table["threshold_value"].astype(float),
        equal_nan=False,
    )


def test_should_compare_primary_and_heuristic_methods_when_running_scenario_k(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_k,
        "task_specs",
        lambda task_type, quick=False: [task_specs(task_type, quick=True)[0]],
    )
    monkeypatch.setattr(
        scenario_k,
        "quantile_grid",
        lambda quick: np.asarray([0.10]),
    )
    monkeypatch.setattr(
        scenario_k,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": meta}
        ),
    )

    scenario_k.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert captured["prefix"] == "scenario_k_mondrian_regression"
    assert {
        "method",
        "difficulty_estimator",
        "requested_reject_rate",
        "empirical_reject_rate",
        "accepted_coverage",
        "accepted_interval_width",
        "accepted_mae",
        "accepted_mse",
    } <= set(table.columns)
    assert {"paper_difficulty_mondrian", "threshold_baseline", "value_bin_width_baseline"} <= set(
        table["method"]
    )
    assert {"target_formal_result", "heuristic"} <= set(table["guarantee_status"])


def test_should_use_accept_rate_not_coverage_column_in_scenario_h(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_h,
        "task_specs",
        lambda task_type, quick=False: [task_specs(task_type, quick=True)[0]],
    )
    monkeypatch.setattr(
        scenario_h,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update({"table": table.copy()}),
    )

    scenario_h.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    assert "accept_rate" in table.columns, "scenario H must use 'accept_rate' column"
    assert (
        "coverage" not in table.columns
    ), "scenario H must not use 'coverage' for the accept fraction"


def test_should_have_non_increasing_accepted_width_for_k1_as_reject_rate_increases(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_k,
        "task_specs",
        lambda task_type, quick=False: [task_specs(task_type, quick=True)[0]],
    )
    monkeypatch.setattr(scenario_k, "quantile_grid", lambda quick: np.linspace(0.10, 0.40, 3))
    monkeypatch.setattr(
        scenario_k,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update({"table": table.copy()}),
    )

    scenario_k.run(RunConfig(seed=42, quick=True))

    table = captured["table"]
    k1 = table[table["method"] == "paper_difficulty_mondrian"].sort_values("requested_reject_rate")
    widths = k1["accepted_interval_width"].dropna().tolist()
    if len(widths) >= 2:
        violations = sum(widths[i] < widths[i + 1] - 0.05 for i in range(len(widths) - 1))
        assert (
            violations == 0
        ), "K1 accepted_interval_width must be non-increasing as reject rate rises"


def test_should_generate_deterministic_policy_matrix_outputs_across_repeated_runs(
    monkeypatch,
) -> None:
    captures: list[tuple[str, pd.DataFrame, dict[str, object]]] = []

    def _capture(prefix, table, meta):
        captures.append((prefix, table.copy(), dict(meta)))

    monkeypatch.setattr(scenario_a, "write_csv_json_md", _capture)
    monkeypatch.setattr(
        scenario_a,
        "confidence_grid",
        lambda quick, start=0.80, stop=0.99: np.asarray([0.85, 0.95]),
    )

    scenario_a.run(RunConfig(seed=42, quick=True))
    scenario_a.run(RunConfig(seed=42, quick=True))

    (_, table1, meta1), (_, table2, meta2) = captures
    pd.testing.assert_frame_equal(table1.reset_index(drop=True), table2.reset_index(drop=True))
    assert meta1["outcome"] == meta2["outcome"]


def test_should_generate_deterministic_regression_threshold_outputs_across_repeated_runs(
    monkeypatch,
) -> None:
    captures: list[tuple[str, pd.DataFrame, dict[str, object]]] = []

    def _capture(prefix, table, meta):
        captures.append((prefix, table.copy(), dict(meta)))

    monkeypatch.setattr(scenario_d, "write_csv_json_md", _capture)
    monkeypatch.setattr(
        scenario_d,
        "quantile_grid",
        lambda quick: np.asarray([0.10, 0.30]),
    )

    scenario_d.run(RunConfig(seed=42, quick=True))
    scenario_d.run(RunConfig(seed=42, quick=True))

    (_, table1, meta1), (_, table2, meta2) = captures
    pd.testing.assert_frame_equal(table1.reset_index(drop=True), table2.reset_index(drop=True))
    assert meta1["outcome"] == meta2["outcome"]


def test_should_match_checked_in_policy_matrix_artifact_when_recomputed(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_a,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": dict(meta)}
        ),
    )

    artifact_meta_path = Path("evaluation/reject/artifacts/policy_matrix.json")
    artifact_csv_path = Path("evaluation/reject/artifacts/policy_matrix.csv")
    assert artifact_meta_path.exists()
    assert artifact_csv_path.exists()

    artifact_meta = json.loads(artifact_meta_path.read_text(encoding="utf-8"))
    artifact_table = pd.read_csv(artifact_csv_path)

    scenario_a.run(RunConfig(seed=42, quick=bool(artifact_meta["quick"])))

    generated_table = captured["table"].reset_index(drop=True)
    expected_table = artifact_table.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        generated_table,
        expected_table,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    generated_meta = _normalize_meta(captured["meta"])
    assert generated_meta == artifact_meta


def test_should_match_checked_in_regression_threshold_artifact_when_recomputed(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scenario_d,
        "write_csv_json_md",
        lambda prefix, table, meta: captured.update(
            {"prefix": prefix, "table": table.copy(), "meta": dict(meta)}
        ),
    )

    artifact_meta_path = Path("evaluation/reject/artifacts/regression_threshold.json")
    artifact_csv_path = Path("evaluation/reject/artifacts/regression_threshold.csv")
    assert artifact_meta_path.exists()
    assert artifact_csv_path.exists()

    artifact_meta = json.loads(artifact_meta_path.read_text(encoding="utf-8"))
    artifact_table = pd.read_csv(artifact_csv_path)

    scenario_d.run(RunConfig(seed=42, quick=bool(artifact_meta["quick"])))

    generated_table = captured["table"].reset_index(drop=True)
    expected_table = artifact_table.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        generated_table,
        expected_table,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    generated_meta = _normalize_meta(captured["meta"])
    assert generated_meta == artifact_meta


def test_should_use_policy_coupled_predict_path_when_building_reject_breakdown() -> None:
    class DummyResult:
        def __init__(self):
            self.rejected = np.array([True, False, True], dtype=bool)
            self.metadata = {
                "reject_rate": 2 / 3,
                "error_rate": 0.1,
                "error_rate_defined": True,
                "ambiguity_rate": 2 / 3,
                "novelty_rate": 0.0,
                "ambiguity_mask": np.array([True, False, True], dtype=bool),
                "novelty_mask": np.array([False, False, False], dtype=bool),
                "prediction_set_size": np.array([2, 1, 2]),
                "prediction_set": np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
                "effective_confidence": 0.93,
                "effective_threshold": 0.4,
                "threshold_source": "call",
                "epsilon": 0.07,
            }

    class DummyWrapper:
        def __init__(self):
            self.calls = []

        def predict(self, x, **kwargs):
            self.calls.append(kwargs)
            return DummyResult()

    wrapper = DummyWrapper()
    breakdown = reject_breakdown(
        wrapper,
        np.array([[0.0], [1.0], [2.0]]),
        confidence=0.93,
        threshold=0.4,
        ncf="ensured",
        w=0.6,
    )
    assert len(wrapper.calls) == 1
    call_kwargs = wrapper.calls[0]
    assert "reject_policy" in call_kwargs
    assert call_kwargs["confidence"] == pytest.approx(0.93)
    assert call_kwargs["threshold"] == pytest.approx(0.4)
    assert breakdown["effective_confidence"] == pytest.approx(0.93)
    assert breakdown["effective_threshold"] == pytest.approx(0.4)
    assert breakdown["threshold_source"] == "call"


def test_should_fill_breakdown_defaults_when_metadata_is_sparse() -> None:
    class SparseResult:
        def __init__(self):
            self.rejected = np.array([False, True], dtype=bool)
            self.metadata = {"epsilon": 0.05}

    breakdown = breakdown_from_reject_output(SparseResult(), default_confidence=0.95)
    assert breakdown["reject_rate"] == pytest.approx(0.5)
    assert breakdown["effective_confidence"] == pytest.approx(0.95)
    assert breakdown["prediction_set_size"].tolist() == [1, 0]
    assert breakdown["prediction_set"] is None


def test_should_not_claim_proved_guarantee_in_scenario_f_metadata() -> None:
    json_path = Path("evaluation/reject/artifacts/scenario_f_multiclass_coverage.json")
    if not json_path.exists():
        pytest.skip("scenario_f artifact not present")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    highlights = " ".join(meta.get("highlights", []))
    assert (
        "proved" not in highlights.lower() or "formalization target" in highlights.lower()
    ), "Scenario F metadata must not claim a proved guarantee for multiclass correctness"
