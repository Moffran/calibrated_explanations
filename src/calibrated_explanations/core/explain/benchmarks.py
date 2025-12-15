"""Benchmarking harness for parallel explanation strategies.

This module provides tools to measure and compare the performance of different
parallel execution strategies for generating explanations.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...parallel import ParallelConfig, ParallelExecutor
from ...utils.exceptions import CalibratedError

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    strategy: str
    workers: int
    duration: float
    throughput: float  # instances per second
    metrics: Dict[str, Any]


class ParallelBenchmark:
    """Harness for benchmarking parallel explanation performance."""

    def __init__(self, explainer: Any, data: Any):
        """Initialize benchmark with an explainer and dataset."""
        self.explainer = explainer
        self.data = data
        self.results: List[BenchmarkResult] = []

    def run(
        self, strategies: Optional[List[str]] = None, worker_counts: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """Run benchmarks for specified strategies and worker counts."""
        strategies = strategies or ["sequential", "threads", "processes"]
        worker_counts = worker_counts or [2, 4, 8]

        # Baseline: Sequential
        if "sequential" in strategies:
            self._run_single("sequential", 1)

        for strategy in strategies:
            if strategy == "sequential":
                continue
            for workers in worker_counts:
                self._run_single(strategy, workers)

        return self.results

    def _run_single(self, strategy: str, workers: int) -> None:
        """Execute a single benchmark configuration."""
        logger.info(f"Benchmarking strategy={strategy}, workers={workers}")

        config = ParallelConfig(
            enabled=True,
            strategy=strategy,  # type: ignore
            max_workers=workers,
        )
        executor = ParallelExecutor(config)

        # Inject executor into explainer temporarily
        # We look for where the explainer stores its executor
        original_executor = getattr(self.explainer, "executor", None)
        original_perf_parallel = getattr(self.explainer, "_perf_parallel", None)

        # Prefer setting _perf_parallel if it exists, as that's often the internal one
        if hasattr(self.explainer, "_perf_parallel"):
            self.explainer._perf_parallel = executor
        else:
            self.explainer.executor = executor

        try:
            start = time.perf_counter()

            # Run explanation
            _ = self.explainer.explain(self.data)

            duration = time.perf_counter() - start
            n_instances = len(self.data) if hasattr(self.data, "__len__") else 1
            throughput = n_instances / duration if duration > 0 else 0.0

            metrics = executor.metrics.snapshot()

            self.results.append(
                BenchmarkResult(
                    strategy=strategy,
                    workers=workers,
                    duration=duration,
                    throughput=throughput,
                    metrics=metrics,
                )
            )

        except CalibratedError as e:
            logger.error(f"Benchmark failed for {strategy}/{workers}: {e}")
        finally:
            # Restore original executor
            if hasattr(self.explainer, "_perf_parallel"):
                self.explainer._perf_parallel = original_perf_parallel
            else:
                self.explainer.executor = original_executor

    def report(self) -> str:
        """Generate a text report of benchmark results."""
        lines = ["Parallel Execution Benchmark Results", "=" * 60]
        lines.append(
            f"{'Strategy':<12} | {'Workers':<8} | {'Duration (s)':<12} | {'Throughput (i/s)':<16}"
        )
        lines.append("-" * 60)

        for res in sorted(self.results, key=lambda x: x.duration):
            lines.append(
                f"{res.strategy:<12} | {res.workers:<8} | {res.duration:<12.4f} | {res.throughput:<16.2f}"
            )
        return "\n".join(lines)
