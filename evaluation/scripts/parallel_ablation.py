"""Ablation study comparing parallel execution options for modern explainers.

The benchmark mirrors :mod:`evaluation.scripts.compare_explain_performance` but
focuses exclusively on the modern explainer implementation while sweeping
through a variety of parallel execution strategies.  Multiple dataset shapes
are exercised to demonstrate how instance counts and feature dimensionality
influence the impact of each configuration tweak.

Run from the repository root::

    PYTHONPATH=./src:. python evaluation/scripts/parallel_ablation.py

The script prints JSON with timing measurements and speed-ups relative to the
sequential baseline for every dataset/operation combination.
Pass ``--output path/to/results.json`` to persist the JSON to disk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


# ---------------------------------------------------------------------------
# Global evaluation parameters
# ---------------------------------------------------------------------------

RNG_SEED: int = 41

# Keep timings reasonably stable without making the benchmark prohibitively
# slow.  Warm-up iterations help avoid start-up noise (especially for process
# pools) while the repeat count smooths out fluctuations.
TIMING_REPEAT: int = 1
TIMING_WARMUP: int = 1

# When True, print simple progress messages showing the current task and
# parallel variant before each evaluation iteration.
VERBOSE: bool = True

EXPLANATION_APIS: Tuple[str, ...] = ("explain_factual", "explore_alternatives")


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
# Dataset setup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Definition of a synthetic dataset used for the ablation."""

    name: str
    problem: Literal["classification", "regression"]
    samples: int
    features: int
    informative: int
    redundant: int
    estimators: int
    calibration: int
    test: int
    noise: float = 0.2


@dataclass
class ScenarioSetup:
    """Pre-computed data required to instantiate explainers for a scenario."""

    spec: DatasetSpec
    learner: Any
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    kwargs: Dict[str, Any]

COMPACT = DatasetSpec(
    name="compact",
    problem="both",
    samples=600,
    features=10,
    informative=5,
    redundant=2,
    estimators=10,
    calibration=150,
    test=100,
    noise=0
)

WIDE = DatasetSpec(
    name="Wide",
    problem="both",
    samples=3500,
    features=200,
    informative=50,
    redundant=30,
    estimators=10,
    calibration=1000,
    test=500,
    noise=0.3
)

BATCHES_PER_WORKER = 4

DATASET_SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="classification_compact",
        problem="classification",
        samples=COMPACT.samples,
        features=COMPACT.features,
        informative=COMPACT.informative,
        redundant=COMPACT.redundant,
        estimators=COMPACT.estimators,
        calibration=COMPACT.calibration,
        test=COMPACT.test,
    ),
    DatasetSpec(
        name="classification_many_features",
        problem="classification",
        samples=COMPACT.samples,
        features=WIDE.features,
        informative=WIDE.informative,
        redundant=WIDE.redundant,
        estimators=COMPACT.estimators,
        calibration=COMPACT.calibration,
        test=COMPACT.test,
    ),
    DatasetSpec(
        name="classification_many_instances",
        problem="classification",
        samples=WIDE.samples,
        features=COMPACT.features,
        informative=COMPACT.informative,
        redundant=COMPACT.redundant,
        estimators=COMPACT.estimators,
        calibration=WIDE.calibration,
        test=WIDE.test,
    ),
    DatasetSpec(
        name="regression_compact",
        problem="regression",
        samples=COMPACT.samples,
        features=COMPACT.features,
        informative=COMPACT.informative,
        redundant=COMPACT.redundant,
        estimators=COMPACT.estimators,
        calibration=COMPACT.calibration,
        test=COMPACT.test,
        noise=COMPACT.noise,
    ),
    DatasetSpec(
        name="regression_many_features",
        problem="regression",
        samples=COMPACT.samples,
        features=WIDE.features,
        informative=WIDE.informative,
        redundant=WIDE.redundant,
        estimators=COMPACT.estimators,
        calibration=COMPACT.calibration,
        test=COMPACT.test,
        noise=WIDE.noise,
    ),
    DatasetSpec(
        name="regression_many_instances",
        problem="regression",
        samples=WIDE.samples,
        features=COMPACT.features,
        informative=COMPACT.informative,
        redundant=COMPACT.redundant,
        estimators=COMPACT.estimators,
        calibration=WIDE.calibration,
        test=WIDE.test,
        noise=COMPACT.noise,
    ),
)


def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    calibration: int,
    test: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _build_setup(spec: DatasetSpec, *, random_state: int) -> ScenarioSetup:
    """Prepare model, calibration data, and metadata for a dataset specification."""

    if spec.problem == "classification":
        x, y = make_classification(
            n_samples=spec.samples,
            n_features=spec.features,
            n_informative=spec.informative,
            n_redundant=spec.redundant,
            random_state=random_state,
        )
        learner = RandomForestClassifier(n_estimators=spec.estimators, random_state=random_state)
    else:
        x, y = make_regression(
            n_samples=spec.samples,
            n_features=spec.features,
            n_informative=spec.informative,
            noise=spec.noise,
            random_state=random_state,
        )
        learner = RandomForestRegressor(n_estimators=spec.estimators, random_state=random_state)

    x_train, y_train, x_cal, y_cal, x_test, _ = _split_dataset(
        x, y, calibration=spec.calibration, test=spec.test
    )
    learner.fit(x_train, y_train)
    feature_names = [f"f{i}" for i in range(x.shape[1])]
    kwargs = {
        "mode": spec.problem,
        "feature_names": feature_names,
        "categorical_features": [],
        "fast": False,
        "suppress_crepes_errors": True,
    }
    return ScenarioSetup(spec, learner, x_cal, y_cal, x_test, kwargs)


# ---------------------------------------------------------------------------
# Parallel configuration variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelVariant:
    """Description of a parallel configuration under test."""

    name: str
    config: ParallelConfig
    description: str


def _build_parallel_variants() -> Tuple[ParallelVariant, ...]:
    """Return the collection of parallel configurations to benchmark."""
    # Detect CPU counts: prefer psutil for physical cores; fall back to os.cpu_count()
    import os

    try:  # pragma: no cover - optional dependency
        import psutil  # type: ignore

        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:  # pragma: no cover - psutil optional
        phy = os.cpu_count() or 1
        # Heuristic: assume hyperthreading if even and > 1
        physical_cores = phy // 2 if phy > 1 and phy % 2 == 0 else phy

    logical_cores = os.cpu_count() or 1

    variants: List[ParallelVariant] = []

    # Baseline sequential
    variants.append(
        ParallelVariant(
            name="sequential_baseline",
            config=ParallelConfig(enabled=False, strategy="sequential"),
            description="Modern implementation without parallel executor",
        )
    )

    # Helper to add paired instance/feature variants for a strategy and worker count
    def _add_pair(prefix: str, strategy: str, max_workers: int | None, instance_min: int, feature_min: int):
        variants.append(
            ParallelVariant(
                name=f"{prefix}_instance",
                config=ParallelConfig(
                    enabled=True,
                    strategy=strategy,
                    max_workers=max_workers,
                    min_batch_size=instance_min,
                    granularity="instance",
                ),
                description=f"{strategy} pool with {max_workers!s} workers (instance granularity)",
            )
        )
        variants.append(
            ParallelVariant(
                name=f"{prefix}_feature",
                config=ParallelConfig(
                    enabled=True,
                    strategy=strategy,
                    max_workers=max_workers,
                    min_batch_size=feature_min,
                    granularity="feature",
                ),
                description=f"{strategy} pool with {max_workers!s} workers (feature granularity)",
            )
        )

    # Threads: evaluate both physical and logical worker counts equally
    _add_pair(
        "threads_physical_small",
        "threads",
        physical_cores,
        instance_min=int(COMPACT.test/physical_cores/BATCHES_PER_WORKER),
        feature_min=int(COMPACT.features/physical_cores)
    )
    _add_pair(
        "threads_logical_small",
        "threads",
        logical_cores,
        instance_min=int(COMPACT.test/logical_cores/BATCHES_PER_WORKER),
        feature_min=int(COMPACT.features/logical_cores)
    )
    _add_pair(
        "threads_physical_large",
        "threads",
        physical_cores,
        instance_min=int(WIDE.test/physical_cores/BATCHES_PER_WORKER),
        feature_min=int(WIDE.features/physical_cores)
    )
    _add_pair(
        "threads_logical_large",
        "threads",
        logical_cores,
        instance_min=int(WIDE.test/logical_cores/BATCHES_PER_WORKER),
        feature_min=int(WIDE.features/logical_cores)
    )

    # Processes: prefer physical cores for process pools
    _add_pair(
        "processes_physical_small",
        "processes",
        physical_cores,
        instance_min=int(COMPACT.test/physical_cores/BATCHES_PER_WORKER),
        feature_min=int(COMPACT.features/physical_cores)
    )
    _add_pair(
        "processes_physical_large",
        "processes",
        physical_cores,
        instance_min=int(WIDE.test/physical_cores/BATCHES_PER_WORKER),
        feature_min=int(WIDE.features/physical_cores)
    )

    try:  # pragma: no cover - optional dependency
        import joblib  # noqa: F401
    except Exception:  # pragma: no cover - joblib remains optional
        joblib_available = False
    else:
        joblib_available = True

    if joblib_available:
        # Provide both instance- and feature-oriented joblib variants to give equal
        # weight to both granularities in the ablation.
        _add_pair(
            "joblib_small",
            "joblib",
            -1,
            instance_min=int(COMPACT.test/logical_cores/BATCHES_PER_WORKER),
            feature_min=int(COMPACT.features/logical_cores)
        )
        _add_pair(
            "joblib_large",
            "joblib",
            -1,
            instance_min=int(WIDE.test/logical_cores/BATCHES_PER_WORKER),
            feature_min=int(WIDE.features/logical_cores)
        )

    return tuple(variants)


PARALLEL_VARIANTS: Tuple[ParallelVariant, ...] = _build_parallel_variants()


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------


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


def _assert_equivalent(modern_payload: Mapping[str, np.ndarray], reference_payload: Mapping[str, np.ndarray]) -> None:
    """Validate that payloads match the sequential baseline output."""

    for key in reference_payload:
        if not np.allclose(modern_payload[key], reference_payload[key], atol=1e-8, rtol=1e-7):
            raise AssertionError(f"Mismatch detected for key '{key}'")


def _instantiate_explainer(setup: ScenarioSetup, *, parallel: ParallelExecutor | None) -> CalibratedExplainer:
    """Create a fresh explainer instance with the requested parallel executor."""

    kwargs = dict(setup.kwargs)
    
    # Force parallel plugin if parallel executor is enabled
    if parallel is not None and parallel.config.enabled:
        granularity = parallel.config.granularity
        overrides = {}
        if granularity == "feature":
            overrides["factual"] = "core.explanation.factual.feature_parallel"
            overrides["alternative"] = "core.explanation.alternative.feature_parallel"
        elif granularity == "instance":
            overrides["factual"] = "core.explanation.factual.instance_parallel"
            overrides["alternative"] = "core.explanation.alternative.instance_parallel"
        
        if overrides:
            kwargs["explanation_plugin_overrides"] = overrides

    explainer = CalibratedExplainer(
        setup.learner,
        setup.x_cal,
        setup.y_cal,
        perf_parallel=parallel,
        **kwargs,
    )
    explainer.set_discretizer(None)
    return explainer


def _invoke_operation(explainer: CalibratedExplainer, operation: str, x_test: np.ndarray):
    """Execute an explanation API and return the resulting collection."""

    method = getattr(explainer, operation)
    return method(x_test, _use_plugin=False)


def _benchmark_variant(
    setup: ScenarioSetup,
    operation: str,
    variant: ParallelVariant,
    *,
    baseline_payload: Mapping[str, np.ndarray],
) -> Tuple[float, Mapping[str, np.ndarray]]:
    """Measure a variant and return its runtime along with the payload."""

    parallel = ParallelExecutor(variant.config) if variant.config.enabled else None
    explainer = _instantiate_explainer(setup, parallel=parallel)
    callable_fn = lambda: _invoke_operation(explainer, operation, setup.x_test)
    result = callable_fn()
    payload = _collect_payload(result)
    _assert_equivalent(payload, baseline_payload)
    timing = _time_call(callable_fn)
    return timing, payload


def _benchmark_scenario(setup: ScenarioSetup, operation: str) -> Mapping[str, Any]:
    """Return timing information for all parallel variants in a scenario."""

    if VERBOSE:
        print(f"Running baseline for {setup.spec.name}.{operation}...", flush=True)

    baseline_variant = PARALLEL_VARIANTS[0]
    baseline_parallel = ParallelExecutor(baseline_variant.config) if baseline_variant.config.enabled else None
    baseline_explainer = _instantiate_explainer(setup, parallel=baseline_parallel)
    baseline_callable = lambda: _invoke_operation(baseline_explainer, operation, setup.x_test)
    baseline_result = baseline_callable()
    baseline_payload = _collect_payload(baseline_result)
    baseline_time = _time_call(baseline_callable)

    if VERBOSE:
        print(f"Baseline finished: {baseline_time:.4f}s", flush=True)

    variant_entries: List[MutableMapping[str, Any]] = []
    for variant in PARALLEL_VARIANTS:
        # Optionally announce the task/variant about to run
        if VERBOSE:
            print(f"Task: {setup.spec.name}.{operation}  Variant: {variant.name}", flush=True)

        if variant is baseline_variant:
            entry = {
                "name": variant.name,
                "description": variant.description,
                "config": {
                    "enabled": variant.config.enabled,
                    "strategy": variant.config.strategy,
                    "max_workers": variant.config.max_workers,
                    "min_batch_size": variant.config.min_batch_size,
                    "granularity": variant.config.granularity,
                },
                "time": baseline_time,
                "speedup_vs_baseline": 1.0,
            }
            variant_entries.append(entry)
            continue

        timing, _ = _benchmark_variant(
            setup,
            operation,
            variant,
            baseline_payload=baseline_payload,
        )
        entry = {
            "name": variant.name,
            "description": variant.description,
            "config": {
                "enabled": variant.config.enabled,
                "strategy": variant.config.strategy,
                "max_workers": variant.config.max_workers,
                "min_batch_size": variant.config.min_batch_size,
                "granularity": variant.config.granularity,
            },
            "time": timing,
            "speedup_vs_baseline": (baseline_time / timing) if timing else float("inf"),
        }
        variant_entries.append(entry)

    return {
        "scenario": setup.spec.name,
        "operation": operation,
        "dataset": {
            "problem": setup.spec.problem,
            "samples": setup.spec.samples,
            "features": setup.spec.features,
            "calibration": setup.spec.calibration,
            "test": setup.spec.test,
        },
        "baseline_time": baseline_time,
        "variants": variant_entries,
    }


def benchmark_parallel_options() -> Iterable[Mapping[str, Any]]:
    """Yield timing comparisons for all dataset scenarios and operations."""

    rng = np.random.default_rng(RNG_SEED)
    for spec in DATASET_SPECS:
        if VERBOSE:
            print(f"--- Starting scenario: {spec.name} ---", flush=True)
            print(f"Building setup for {spec.name}...", flush=True)
        
        setup = _build_setup(spec, random_state=int(rng.integers(0, 10_000)))
        
        if VERBOSE:
            print(f"Setup built for {spec.name}. Running benchmarks...", flush=True)

        for operation in EXPLANATION_APIS:
            yield _benchmark_scenario(setup, operation)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the parallel ablation benchmark and emit JSON results."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON results. Parent directories are created as needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry-point printing ablation measurements as formatted JSON."""

    args = _parse_args(argv)
    
    metadata = {
        "timing_repeat": TIMING_REPEAT,
        "timing_warmup": TIMING_WARMUP,
        "parallel_variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "config": {
                    "enabled": variant.config.enabled,
                    "strategy": variant.config.strategy,
                    "max_workers": variant.config.max_workers,
                    "min_batch_size": variant.config.min_batch_size,
                    "granularity": variant.config.granularity,
                },
            }
            for variant in PARALLEL_VARIANTS
        ],
    }
    
    results = []
    
    # If output file exists, try to load existing results to resume or append?
    # For now, we just overwrite but update continuously.
    
    for result in benchmark_parallel_options():
        results.append(result)
        
        output = {"metadata": metadata, "results": results}
        serialized = json.dumps(output, indent=2, sort_keys=True)
        
        if VERBOSE:
            # Print the last result summary to stdout
            print(f"Completed {result['scenario']}.{result['operation']} - Baseline: {result['baseline_time']:.4f}s")

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
            
    # Final print to stdout if no output file, or just as confirmation
    if args.output is None:
        print(serialized)

if __name__ == "__main__":
    main()
