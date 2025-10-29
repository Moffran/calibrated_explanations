from __future__ import annotations

import json

import pytest

from tests.docs.get_started.test_quickstart_classification_doc import (
    _run_quickstart_classification,
)


def test_export_explanations_snippets(tmp_path):
    context = _run_quickstart_classification()
    batch = context.explainer.explain_factual(context.X_test[:10])
    payload = batch.to_json()

    output_path = tmp_path / "factual.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    from calibrated_explanations.explanations import CalibratedExplanations

    exported = CalibratedExplanations.from_json(payload)
    for explanation in exported.explanations:
        print(explanation.task, explanation.prediction)

    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    rows = []
    for idx, exp in enumerate(batch):
        rows.append(
            {
                "instance": idx,
                "prediction": exp.predict,
                "low": exp.prediction_interval[0],
                "high": exp.prediction_interval[1],
            }
        )

    df = pandas.DataFrame(rows)
    parquet_path = tmp_path / "factual_rules.parquet"
    df.to_parquet(parquet_path)

    telemetry = context.explainer.explainer.runtime_telemetry
    telemetry_path = tmp_path / "factual.telemetry.json"
    with telemetry_path.open("w", encoding="utf-8") as fh:
        json.dump(telemetry, fh, indent=2)

    assert parquet_path.exists()
    assert telemetry_path.exists()
