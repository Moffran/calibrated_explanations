# Scripts

## Over-testing

- `over_testing/over_testing_report.py`: summarizes per-test coverage counts (via coverage.py contexts) to flag lines or blocks with unusually high test coverage density.
- `over_testing/over_testing_triage.py`: ranks over-tested files and hotspot lines/blocks from the over-testing report, producing JSON/Markdown triage outputs (optionally maps hotspots to test contexts).
- `over_testing/run_over_testing_pipeline.py`: runs pytest with per-test contexts, then generates over-testing and triage reports in one step.

## Performance

- `perf/micro_bench_perf.py`: quick local micro-benchmark for import time, lightweight map throughput, and explanation timings (`explain_factual` / `explore_alternatives`). Supports benchmarking either the current implementation or the legacy explain path via `--explain-backend`.
- `perf/check_perf_micro.py`: compares two micro-benchmark JSON outputs against `tests/benchmarks/perf_thresholds.json` and fails on import-time and explanation regressions.
- `perf/check_perf_regression.py`: compares baseline and current perf snapshots using threshold rules.
- `perf/collect_baseline.py`: collects benchmark baselines into JSON artifacts.
- `perf/run_micro_benchmarks.py`: helper to run the micro benchmark and emit JSON outputs.
- `perf/stream_benchmark.py`: benchmark for streaming export throughput and memory.

## Quality

- `quality/check_docstring_coverage.py`: reports module/class/function/method docstring coverage so Standard-002 adoption can be monitored. Use `--fail-under` locally when experimenting with stricter thresholds.
- `quality/check_coverage_gates.py`: enforces per-module coverage gates.
- `quality/detect_test_anti_patterns.py`: flags test anti-patterns (ADR-030 enforcement).
- `quality/prune_allowlist.py`: trims allowlists for test scans.
- `quality/check_import_graph.py`: enforces ADR-001 import graph boundaries.
- `quality/find_missing_numpy_imports.py`: detects missing numpy imports in tests.

Example (manual):

```powershell
# Generate baseline/current JSON
python scripts/perf/micro_bench_perf.py --explain-backend legacy > .bench_baseline.json
python scripts/perf/micro_bench_perf.py --explain-backend current > .bench_current.json

# Compare using thresholds
python scripts/perf/check_perf_micro.py .bench_baseline.json .bench_current.json tests/benchmarks/perf_thresholds.json
```

Note: The CI integration can adopt these scripts to gate regressions without impacting user behavior.
