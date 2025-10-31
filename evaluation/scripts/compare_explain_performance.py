"""Benchmark the optimised explain API variants against the legacy path.

The script builds lightweight classification and regression explainers and
compares the following execution strategies for both
``CalibratedExplainer.explain_factual`` and
``CalibratedExplainer.explore_alternatives``:

* The preserved legacy implementation from
  :mod:`calibrated_explanations.core._legacy_explain`.
* The modern, cache-free implementation shipped in ``CalibratedExplainer``.
* The modern implementation with calibrator caching enabled.
* The modern implementation with the parallel executor enabled.
* The modern implementation with both caching and parallelisation enabled.

Each measurement validates the produced explanations against the legacy output
to guard against behavioural drift.  The resulting timings are emitted as JSON
containing the absolute runtime and the speed-ups relative to the legacy and
modern baselines.

Run from the repository root::

    PYTHONPATH=. python evaluation/scripts/compare_explain_performance.py
"""

from __future__ import annotations

import json
import copy
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core._legacy_explain import explain as legacy_explain
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.perf import CacheConfig, CalibratorCache, ParallelConfig, ParallelExecutor


def _install_pickling_shim() -> None:
    """Ensure ``CalibratedExplainer`` instances remain deepcopy-friendly."""

    def __getstate__(self: CalibratedExplainer) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_perf_cache"] = None
        state["_perf_parallel"] = None
        return state

    def __setstate__(self: CalibratedExplainer, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self: CalibratedExplainer, memo: Dict[int, Any]) -> CalibratedExplainer:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = dict(self.__dict__)
        state["_perf_cache"] = None
        state["_perf_parallel"] = None
        result.__dict__.update(state)
        return result

    setattr(CalibratedExplainer, "__getstate__", __getstate__)  # type: ignore[attr-defined]
    setattr(CalibratedExplainer, "__setstate__", __setstate__)  # type: ignore[attr-defined]
    setattr(CalibratedExplainer, "__deepcopy__", __deepcopy__)  # type: ignore[attr-defined]


_install_pickling_shim()

# ---------------------------------------------------------------------------
# Global evaluation parameters
# ---------------------------------------------------------------------------

RNG_SEED: int = 42

CLASSIFICATION_SAMPLES: int = 2000
CLASSIFICATION_FEATURES: int = 64
CLASSIFICATION_INFORMATIVE: int = 16
CLASSIFICATION_REDUNDANT: int = 0
CLASSIFICATION_ESTIMATORS: int = 10

REGRESSION_SAMPLES: int = 2000
REGRESSION_FEATURES: int = 64
REGRESSION_INFORMATIVE: int = 16
REGRESSION_NOISE: float = 0.2
REGRESSION_ESTIMATORS: int = 10

CALIBRATION_SIZE: int = 500
TEST_SIZE: int = 100

TIMING_REPEAT: int = 10
TIMING_WARMUP: int = 1

CACHE_ENABLED: bool = True
CACHE_NAMESPACE: str = "benchmark"
CACHE_VERSION: str = "v1"
CACHE_MAX_ITEMS: int = 2048
CACHE_MAX_BYTES: int | None = 64 * 1024 * 1024
CACHE_TTL_SECONDS: float | None = None

PARALLEL_ENABLED: bool = True
# Strategy can be "threads", "processes", or "joblib" (when installed).
# Threads are generally the safest default, while processes can help with
# CPU-bound workloads at the cost of higher start-up overhead.  Joblib
# delegates to its own smart chunking and can saturate all cores.
PARALLEL_STRATEGY: Literal["auto", "sequential", "joblib", "threads", "processes"] = "threads"
# Tune worker counts per backend:
#   - threads: try 4-8 workers on laptops, or up to 5x CPU cores for servers.
#   - processes: match the physical core count (``os.cpu_count()``) for best results.
#   - joblib: ``-1`` uses all available cores, or pass an explicit integer.
PARALLEL_WORKERS: int | None = 6
# For instance-level granularity we typically parallelise over examples.
# Keep the batch size small (e.g. 1-4) to maximise concurrency, but bump this
# up for feature-level work to avoid overhead.
PARALLEL_MIN_BATCH: int = 4
# ``feature`` maintains the historical behaviour.  Switch to ``instance`` to
# parallelise explanation calls for each row.
PARALLEL_GRANULARITY: Literal["feature", "instance"] = "instance"

EXPLANATION_APIS: Tuple[str, ...] = ("explain_factual", "explore_alternatives")
VARIANT_ORDER: Tuple[str, ...] = (
    "legacy",
    "modern",
    "cached",
    "parallel",
    "cache_parallel",
)


def _hash_for_cache(part: Any) -> Any:
    """Produce a hashable representation for cache key components."""

    if part is None or isinstance(part, (str, bytes, int, float, bool)):
        return part
    if isinstance(part, np.ndarray):
        return (
            "nd",
            tuple(part.shape),
            str(part.dtype),
            hashlib.sha1(part.view(np.uint8)).hexdigest(),
        )
    if isinstance(part, (list, tuple, set, frozenset)):
        return tuple(_hash_for_cache(item) for item in part)
    if isinstance(part, dict):
        return tuple(sorted((key, _hash_for_cache(value)) for key, value in part.items()))
    return ("repr", repr(part))


class _HashingCalibratorCache(CalibratorCache[Any]):
    """Cache variant normalising composite parts before hashing."""

    def get(self, *, stage: str, parts: Iterable[Any]) -> Any:
        hashed_parts = tuple(_hash_for_cache(part) for part in parts)
        return super().get(stage=stage, parts=hashed_parts)

    def set(self, *, stage: str, parts: Iterable[Any], value: Any) -> None:
        hashed_parts = tuple(_hash_for_cache(part) for part in parts)
        super().set(stage=stage, parts=hashed_parts, value=value)


@dataclass
class BenchmarkResult:
    """Container describing benchmark measurements for one scenario/mode."""

    scenario: str
    operation: str
    timings: Dict[str, float]

    def as_dict(self) -> Dict[str, Mapping[str, float]]:
        """Return timings with speed-ups relative to legacy and modern baselines."""

        if "legacy" not in self.timings or "modern" not in self.timings:
            raise ValueError("Timings must include both 'legacy' and 'modern' baselines")
        legacy_time = self.timings["legacy"]
        modern_time = self.timings["modern"]
        output: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        for variant in VARIANT_ORDER:
            if variant not in self.timings:
                continue
            time_taken = self.timings[variant]
            entry = {
                "time": time_taken,
                "speedup_vs_legacy": (legacy_time / time_taken) if time_taken else float("inf"),
            }
            if variant == "modern":
                entry["speedup_vs_modern"] = 1.0
            else:
                entry["speedup_vs_modern"] = (modern_time / time_taken) if time_taken else float("inf")
            output[variant] = entry
        return output


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


@dataclass
class ScenarioSetup:
    """Pre-computed data required to instantiate explainers for a scenario."""

    name: str
    learner: Any
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    kwargs: Dict[str, Any]


def _make_classification_setup(random_state: int) -> ScenarioSetup:
    """Prepare model, calibration data, and metadata for classification."""

    x, y = make_classification(
        n_samples=CLASSIFICATION_SAMPLES,
        n_features=CLASSIFICATION_FEATURES,
        n_informative=CLASSIFICATION_INFORMATIVE,
        n_redundant=CLASSIFICATION_REDUNDANT,
        random_state=random_state,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split_dataset(
        x, y, calibration=CALIBRATION_SIZE, test=TEST_SIZE
    )
    model = RandomForestClassifier(n_estimators=CLASSIFICATION_ESTIMATORS, random_state=random_state)
    model.fit(x_train, y_train)
    feature_names = [f"f{i}" for i in range(x.shape[1])]
    kwargs = {
        "mode": "classification",
        "feature_names": feature_names,
        "categorical_features": [],
        "fast": False,
        "suppress_crepes_errors": True,
    }
    return ScenarioSetup("classification", model, x_cal, y_cal, x_test, kwargs)


def _make_regression_setup(random_state: int) -> ScenarioSetup:
    """Prepare model, calibration data, and metadata for regression."""

    x, y = make_regression(
        n_samples=REGRESSION_SAMPLES,
        n_features=REGRESSION_FEATURES,
        n_informative=REGRESSION_INFORMATIVE,
        noise=REGRESSION_NOISE,
        random_state=random_state,
    )
    x_train, y_train, x_cal, y_cal, x_test, _ = _split_dataset(
        x, y, calibration=CALIBRATION_SIZE, test=TEST_SIZE
    )
    model = RandomForestRegressor(n_estimators=REGRESSION_ESTIMATORS, random_state=random_state)
    model.fit(x_train, y_train)
    feature_names = [f"r{i}" for i in range(x.shape[1])]
    kwargs = {
        "mode": "regression",
        "feature_names": feature_names,
        "categorical_features": [],
        "fast": False,
        "suppress_crepes_errors": True,
    }
    return ScenarioSetup("regression", model, x_cal, y_cal, x_test, kwargs)


def _build_perf_cache() -> CalibratorCache[Any] | None:
    """Create a calibrator cache when caching is enabled."""

    if not CACHE_ENABLED:
        return None
    cfg = CacheConfig(
        enabled=True,
        namespace=CACHE_NAMESPACE,
        version=CACHE_VERSION,
        max_items=CACHE_MAX_ITEMS,
        max_bytes=CACHE_MAX_BYTES,
        ttl_seconds=CACHE_TTL_SECONDS,
    )
    return _HashingCalibratorCache(cfg)


def _build_parallel_executor(cache: CalibratorCache[Any] | None = None) -> ParallelExecutor | None:
    """Create a parallel executor when parallelisation is enabled."""

    if not PARALLEL_ENABLED:
        return None
    cfg = ParallelConfig(
        enabled=True,
        strategy=PARALLEL_STRATEGY,
        max_workers=PARALLEL_WORKERS,
        min_batch_size=PARALLEL_MIN_BATCH,
        granularity=PARALLEL_GRANULARITY,
    )
    return ParallelExecutor(cfg, cache=cache)


def _instantiate_explainer(
    setup: ScenarioSetup,
    *,
    enable_cache: bool,
    enable_parallel: bool,
) -> CalibratedExplainer:
    """Create a new explainer instance with the requested performance knobs."""

    cache = _build_perf_cache() if enable_cache else None
    parallel = _build_parallel_executor(cache if enable_parallel else None) if enable_parallel else None
    explainer = CalibratedExplainer(
        setup.learner,
        setup.x_cal,
        setup.y_cal,
        perf_cache=cache,
        perf_parallel=parallel,
        **setup.kwargs,
    )
    explainer.set_discretizer(None)
    return explainer


def _time_call(fn: Callable[[], None], *, repeat: int = TIMING_REPEAT, warmup: int = TIMING_WARMUP) -> float:
    """Return the average runtime of ``fn`` over ``repeat`` executions with warm-up."""

    for _ in range(max(0, warmup)):
        fn()
    timings: List[float] = []
    for _ in range(max(1, repeat)):
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


def _resolve_discretizer(mode: str, operation: str) -> str:
    """Return the discretizer name associated with the requested operation."""

    is_regression = "regression" in mode
    if operation == "explain_factual":
        return "binaryRegressor" if is_regression else "binaryEntropy"
    if operation == "explore_alternatives":
        return "regressor" if is_regression else "entropy"
    raise ValueError(f"Unknown operation '{operation}'")


def _benchmark_operation(setup: ScenarioSetup, operation: str) -> BenchmarkResult:
    """Benchmark a single explanation API across runtime variants."""

    timings: Dict[str, float] = {}

    # Legacy baseline
    legacy_explainer = _instantiate_explainer(
        setup, enable_cache=False, enable_parallel=False
    )
    legacy_explainer.set_discretizer(_resolve_discretizer(setup.kwargs["mode"], operation))
    legacy_callable = lambda: legacy_explain(legacy_explainer, setup.x_test)
    legacy_reference = legacy_callable()
    timings["legacy"] = _time_call(legacy_callable)

    variant_configs = {
        "modern": {"enable_cache": False, "enable_parallel": False},
        "cached": {"enable_cache": True, "enable_parallel": False},
        "parallel": {"enable_cache": False, "enable_parallel": True},
        "cache_parallel": {"enable_cache": True, "enable_parallel": True},
    }

    for variant, config in variant_configs.items():
        explainer = _instantiate_explainer(
            setup,
            enable_cache=config["enable_cache"],
            enable_parallel=config["enable_parallel"],
        )
        method = getattr(explainer, operation)
        callable_fn = lambda method=method: method(setup.x_test, _use_plugin=False)
        variant_result = callable_fn()
        _assert_equivalent(variant_result, legacy_reference)
        timings[variant] = _time_call(callable_fn)

    return BenchmarkResult(setup.name, operation, timings)


def benchmark_explainers() -> Iterable[BenchmarkResult]:
    """Yield timing comparisons for classification and regression explainers."""

    rng = np.random.default_rng(RNG_SEED)

    classification_setup = _make_classification_setup(
        random_state=int(rng.integers(0, 10_000))
    )
    regression_setup = _make_regression_setup(random_state=int(rng.integers(0, 10_000)))

    for setup in (classification_setup, regression_setup):
        for operation in EXPLANATION_APIS:
            yield _benchmark_operation(setup, operation)


def main() -> None:
    """Entry-point printing benchmark measurements as formatted JSON."""

    results = list(benchmark_explainers())
    output: Dict[str, Dict[str, Mapping[str, float]]] = {}
    for result in results:
        output.setdefault(result.scenario, {})[result.operation] = result.as_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
