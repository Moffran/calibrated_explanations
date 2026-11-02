"""Capability contract tests for explanation narrative generation.

Requirements verified:
  CE-REQ-NARR-API-001 — Narrative output API contract (CE-CAP-NARR-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove narrative quality, fluency, or domain correctness.
See development/capabilities/requirements/CE-REQ-NARR-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

pytest.importorskip(
    "yaml",
    reason="pyyaml not installed; skipping narrative contract tests (requires 'narrative' extra)",
)

from sklearn.datasets import load_breast_cancer, load_iris, make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer  # noqa: E402

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


def _find_narrative_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"Could not find narrative line starting with {prefix!r} in:\n{text}")


@pytest.fixture(scope="module")
def factual_explanations():
    """Return a factual explanation collection from a fitted and calibrated explainer."""
    X, y = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, _ = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)
    return explainer.explain_factual(X_test)


# ---------------------------------------------------------------------------
# CE-REQ-NARR-API-001 — Narrative output API contract
# ---------------------------------------------------------------------------


def test_should_return_non_empty_string_when_narrative_text_format(
    factual_explanations,
):
    """Verify CE-REQ-NARR-API-001: to_narrative(output_format='text') returns non-empty string.

    Acceptance criteria (from CE-REQ-NARR-API-001):
    - to_narrative(output_format='text') completes without error.
    - The result is not None.
    - isinstance(result, str) is True.
    - len(result) > 0.
    """
    result = factual_explanations.to_narrative(output_format="text")

    assert result is not None, (
        "CE-REQ-NARR-API-001: to_narrative(output_format='text') must return non-None"
    )
    assert isinstance(result, str), f"CE-REQ-NARR-API-001: result must be str, got {type(result)}"
    assert len(result) > 0, "CE-REQ-NARR-API-001: to_narrative text output must be non-empty"


def test_should_report_same_binary_prediction_label_in_narrative_as_predict_output():
    """Verify binary narratives state the same predicted label as the calibrated predictor."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=_RNG_SEED,
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=_RNG_SEED,
    )

    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=100, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal, feature_names=dataset.feature_names.tolist())

    predictions = explainer.predict(X_test)
    class_zero_idx = int(np.flatnonzero(predictions == 0)[0])
    class_one_idx = int(np.flatnonzero(predictions == 1)[0])
    selected_indices = [class_zero_idx, class_one_idx]

    explanations = explainer.explain_factual(X_test[selected_indices])
    narratives = [
        explanations[idx].to_narrative(output_format="text", expertise_level="advanced")
        for idx in range(len(selected_indices))
    ]

    assert _find_narrative_line(narratives[0], "Prediction:") == "Prediction: 0"
    assert _find_narrative_line(narratives[1], "Prediction:") == "Prediction: 1"


def test_should_render_multiclass_runner_up_and_truthful_predicted_class_in_narrative():
    """Verify multiclass narratives include runner-up details and name the predicted class."""
    dataset = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.3,
        stratify=dataset.target,
        random_state=_RNG_SEED,
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=_RNG_SEED,
    )

    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=100, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal, feature_names=list(dataset.feature_names))

    prediction = explainer.predict(X_test[:1])[0]
    narrative = explainer.explain_factual(X_test[:1])[0].to_narrative(
        output_format="text",
        expertise_level="advanced",
    )

    assert _find_narrative_line(
        narrative,
        "Calibrated Probability for class",
    ).startswith(f"Calibrated Probability for class {prediction}:")
    assert "Runner-up Class:  (margin )" not in narrative

    match = re.search(r"Runner-up Class:\s+(.+?)\s+\(margin\s+([0-9.]+)\)", narrative)
    assert match is not None
    assert match.group(1).strip() != ""
    assert float(match.group(2)) >= 0.0


@pytest.fixture
def no_network(monkeypatch):
    """Make every socket connection and DNS lookup fail for the duration of a test."""
    import socket

    def refuse(*args, **kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_should_return_documented_narrative_formats_offline(factual_explanations, no_network):
    """Verify the existing narrative entry points keep their formats with networking blocked (ADR-041)."""
    pd = pytest.importorskip("pandas")

    from calibrated_explanations.explanations import build_narrative_spec

    single = factual_explanations[0]
    expected = {"dataframe": pd.DataFrame, "text": str, "html": str, "markdown": str}

    for output_format, kind in expected.items():
        assert isinstance(factual_explanations.to_narrative(output_format=output_format), kind)
        assert isinstance(single.to_narrative(output_format=output_format), kind)
    assert isinstance(factual_explanations.to_narrative(output_format="dict"), list)
    assert isinstance(single.to_narrative(output_format="dict"), dict)
    assert isinstance(factual_explanations.to_dataframe(), pd.DataFrame)
    assert isinstance(single.narrate(output_format="text"), str)
    assert isinstance(factual_explanations.plot(style="narrative", output="dict"), list)
    spec = build_narrative_spec(factual_explanations.to_json()["explanations"][0])
    assert spec.context["problem_type"] == "binary_classification"
