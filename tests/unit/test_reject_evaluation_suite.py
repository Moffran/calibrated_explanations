from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.reject import (
    scenario_e_binary_coverage_sweep as scenario_e,
    scenario_f_multiclass_coverage as scenario_f,
    scenario_g_regression_coverage as scenario_g,
    scenario_h_ncf_grid as scenario_h,
    scenario_k_mondrian_regression as scenario_k,
)
from evaluation.reject.common_reject import (
    RunConfig,
    build_regression_bundle,
    load_binary_datasets,
    load_multiclass_datasets,
    load_regression_datasets,
    task_specs,
)


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


def test_should_not_claim_proved_guarantee_in_scenario_f_metadata() -> None:
    json_path = Path("evaluation/reject/artifacts/scenario_f_multiclass_coverage.json")
    if not json_path.exists():
        pytest.skip("scenario_f artifact not present")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    highlights = " ".join(meta.get("highlights", []))
    assert (
        "proved" not in highlights.lower() or "formalization target" in highlights.lower()
    ), "Scenario F metadata must not claim a proved guarantee for multiclass correctness"
