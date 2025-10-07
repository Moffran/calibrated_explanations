# Scripts

- `micro_bench_perf.py`: quick local micro-benchmark for import time and simple map throughput. Prints JSON.
- `check_perf_micro.py`: compares two micro-benchmark JSON outputs against `benchmarks/perf_thresholds.json` and fails on import-time regressions.
- `check_docstring_coverage.py`: reports module/class/function/method docstring coverage so ADR-018 adoption can be monitored. Use `--fail-under` locally when experimenting with stricter thresholds.

Example (manual):

```powershell
# Generate baseline/current JSON
python scripts/micro_bench_perf.py > .bench_baseline.json
python scripts/micro_bench_perf.py > .bench_current.json

# Compare using thresholds
python scripts/check_perf_micro.py .bench_baseline.json .bench_current.json benchmarks/perf_thresholds.json
```

Note: The CI integration can adopt these scripts to gate regressions without impacting user behavior.
