"""Benchmark the optimised explain path against the legacy implementation.

The script builds lightweight classification and regression explainers, runs
the modern :meth:`CalibratedExplainer.explain` implementation and the preserved
legacy variant from :mod:`calibrated_explanations.core._legacy_explain`, and
prints a timing comparison.  It also validates that both paths yield identical
explanation outputs for the sampled datasets to guard against behavioural drift.

Run from the repository root::

    PYTHONPATH=. python evaluation/scripts/compare_explain_performance.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core._legacy_explain import explain as legacy_explain
from calibrated_explanations import CalibratedExplainer


@dataclass
class BenchmarkResult:
    """Container describing a single benchmark measurement."""

    scenario: str
    modern_time: float
    legacy_time: float
    ratio: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "modern_time": self.modern_time,
            "legacy_time": self.legacy_time,
            "speedup": self.ratio,
        }


def _split_dataset(x: np.ndarray, y: np.ndarray, *, calibration: int, test: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into train/calibration/test partitions."""

    if calibration + test >= len(x):
        raise ValueError("Requested calibration/test sizes exceed dataset size")
    x_train = x[: -(calibration + test)]
    y_train = y[: -(calibration + test)]
    x_cal = x[-(calibration + test) : -test]
    y_cal = y[-(calibration + test) : -test]
    x_test = x[-test:]
    y_test = y[-test:]
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def _make_classification_explainer(random_state: int = 0) -> Tuple[CalibratedExplainer, np.ndarray]:
    """Construct a calibrated explainer for a synthetic classification task."""

    x, y = make_classification(
        n_samples=1500,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        random_state=random_state,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split_dataset(x, y, calibration=400, test=200)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    feature_names = [f"f{i}" for i in range(x.shape[1])]
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=[],
        fast=False,
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)
    return explainer, x_test


def _make_regression_explainer(random_state: int = 0) -> Tuple[CalibratedExplainer, np.ndarray]:
    """Construct a calibrated explainer for a synthetic regression task."""

    x, y = make_regression(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        noise=0.2,
        random_state=random_state,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split_dataset(x, y, calibration=400, test=200)
    model = RandomForestRegressor(n_estimators=120, random_state=random_state)
    model.fit(x_train, y_train)
    feature_names = [f"r{i}" for i in range(x.shape[1])]
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        mode="regression",
        feature_names=feature_names,
        categorical_features=[],
        fast=False,
        suppress_crepes_errors=True,
    )
    explainer.set_discretizer(None)
    return explainer, x_test


def _time_call(fn: Callable[[], None], *, repeat: int = 3) -> float:
    """Return the average runtime of ``fn`` over ``repeat`` executions."""

    timings: List[float] = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        timings.append(perf_counter() - start)
    return float(np.mean(timings))


def _collect_payload(collection) -> Dict[str, np.ndarray]:
    """Extract a deterministic payload for comparing explanation results."""

    weights_predict = []
    weights_low = []
    weights_high = []
    predict_vals = []
    low_vals = []
    high_vals = []
    for explanation in collection.explanations:
        weights_predict.append(np.asarray(explanation.feature_weights["predict"], dtype=float))
        weights_low.append(np.asarray(explanation.feature_weights.get("low", []), dtype=float))
        weights_high.append(np.asarray(explanation.feature_weights.get("high", []), dtype=float))
        predict_vals.append(np.asarray(explanation.feature_predict["predict"], dtype=float))
        low_vals.append(np.asarray(explanation.feature_predict.get("low", []), dtype=float))
        high_vals.append(np.asarray(explanation.feature_predict.get("high", []), dtype=float))

    return {
        "weights_predict": np.stack(weights_predict),
        "weights_low": np.stack(weights_low),
        "weights_high": np.stack(weights_high),
        "predict": np.stack(predict_vals),
        "low": np.stack(low_vals),
        "high": np.stack(high_vals),
    }


def _assert_equivalent(modern, legacy) -> None:
    """Validate that legacy and modern explanations carry identical payloads."""

    modern_payload = _collect_payload(modern)
    legacy_payload = _collect_payload(legacy)
    for key in modern_payload:
        if not np.allclose(modern_payload[key], legacy_payload[key], atol=1e-8, rtol=1e-7):
            raise AssertionError(f"Mismatch detected for key '{key}'")


def benchmark_explainers() -> Iterable[BenchmarkResult]:
    """Yield timing comparisons for both classification and regression explainers."""

    rng = np.random.default_rng(42)

    cls_explainer, x_cls = _make_classification_explainer(random_state=int(rng.integers(0, 10_000)))
    reg_explainer, x_reg = _make_regression_explainer(random_state=int(rng.integers(0, 10_000)))

    modern_cls = cls_explainer.explain(x_cls, _use_plugin=False)
    legacy_cls = legacy_explain(cls_explainer, x_cls)
    _assert_equivalent(modern_cls, legacy_cls)

    modern_reg = reg_explainer.explain(x_reg, _use_plugin=False)
    legacy_reg = legacy_explain(reg_explainer, x_reg)
    _assert_equivalent(modern_reg, legacy_reg)

    cls_modern_time = _time_call(lambda: cls_explainer.explain(x_cls, _use_plugin=False))
    cls_legacy_time = _time_call(lambda: legacy_explain(cls_explainer, x_cls))
    reg_modern_time = _time_call(lambda: reg_explainer.explain(x_reg, _use_plugin=False))
    reg_legacy_time = _time_call(lambda: legacy_explain(reg_explainer, x_reg))

    yield BenchmarkResult(
        "classification",
        modern_time=cls_modern_time,
        legacy_time=cls_legacy_time,
        ratio=cls_legacy_time / cls_modern_time if cls_modern_time else float("inf"),
    )
    yield BenchmarkResult(
        "regression",
        modern_time=reg_modern_time,
        legacy_time=reg_legacy_time,
        ratio=reg_legacy_time / reg_modern_time if reg_modern_time else float("inf"),
    )


def main() -> None:
    """Entry-point printing benchmark measurements as formatted JSON."""

    results = list(benchmark_explainers())
    output = {result.scenario: result.as_dict() for result in results}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
