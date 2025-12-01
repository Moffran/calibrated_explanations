# ADR-004 Progression Report - 2025-12-01

## Summary
Implemented key components of the Parallel Execution Improvement Plan (Phases 1, 2, and 3) to satisfy ADR-004 requirements for v0.10.0.

## Changes Implemented

### 1. Configuration Surface (Phase 1)
- Updated `ParallelConfig` to include:
  - `task_size_hint_bytes`: Hint for task payload size to influence strategy selection.
  - `force_serial_on_failure`: Option to enforce serial fallback on execution failure.
- Updated `ParallelConfig.from_env` to parse `task_bytes` and `force_serial` from `CE_PARALLEL` environment variable.

### 2. Executor Context Management (Phase 2)
- Refactored `ParallelExecutor` to support the Context Manager protocol (`__enter__`, `__exit__`).
- Implemented persistent pool management within the context, allowing pool reuse across multiple `map` calls.
- Added `shutdown` logic in `__exit__` to ensure resource cleanup.

### 3. Auto-Strategy Heuristics (Phase 3)
- Enhanced `_auto_strategy` to consider `task_size_hint_bytes`.
- Added heuristic: If `task_size_hint_bytes` > 10MB, prefer `threads` over `processes` to avoid pickling overhead, unless on Windows where `threads` is already default.

### 4. Telemetry (Phase 3/4)
- Added execution duration timing (`duration`) to `parallel_execution` telemetry event.
- Added `workers` count to telemetry event.

### 5. Fallback & Error Handling
- Implemented `force_serial_on_failure` logic in `map`.
- Added logging for pool initialization failures.

## Verification
- Created `tests/repro_parallel.py` to verify:
  - Basic parallel execution.
  - Context manager usage (pool reuse).
  - Environment variable configuration.
- Verified that `ParallelExecutor` correctly falls back to serial or handles errors as configured.

## Next Steps
- **Phase 2.1 (Share heavy payloads)**: Investigate moving immutable arrays to shared memory for `ProcessPoolExecutor` if needed, or rely on `joblib`'s memmapping.
- **Phase 4 (Benchmarking)**: Integrate `evaluation/scripts/parallel_ablation.py` into CI.
- **Documentation**: Update `docs/practitioner/advanced/parallel_execution_playbook.md` with new configuration options and context manager usage.
