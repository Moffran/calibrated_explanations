from __future__ import annotations

import json

from tests.docs.get_started.test_quickstart_classification_doc import (
    _run_quickstart_classification,
)


def test_configure_telemetry_snippets(tmp_path):
    context = _run_quickstart_classification()
    factual = context.explainer.explain_factual(context.X_test[:1])
    telemetry = getattr(factual, "telemetry", {})
    print(telemetry["interval_source"], telemetry["plot_source"])
    print(telemetry["uncertainty"]["calibrated_value"])
    print(telemetry.get("preprocessor", {}))

    payload = context.explainer.explainer.runtime_telemetry
    output = tmp_path / "telemetry.json"
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    assert output.exists()
