Title: Pool-at-Explainer-Init — TDD-First Plan

Purpose
- Implement "pool-at-explainer-init": initialize a warmed worker pool when `CalibratedExplainer` is constructed (optional, opt-in). Use a worker initializer to build a lightweight per-worker explain harness so tasks only pass index/slice data.

Principles
- TDD-first: write focused unit tests first, then implement minimal changes to satisfy them.
- Backward-compatible: default behaviour unchanged unless opt-in flags are set.
- Cross-platform: tests must validate Windows spawn semantics.
- Measure: add simple profiling assertions (reduced pickling/time) in integration checks.

High-level plan (short)
1. Tests: add unit tests asserting that `ProcessPoolExecutor` receives `initializer`/`initargs`, and that worker initializer creates a module-global harness exposing `explain_slice`.
2. Plumbing: add `ParallelConfig.worker_initializer` + `worker_init_args`; update `_process_strategy` to forward initializer/initargs to `ProcessPoolExecutor`.
3. Worker runtime: add `parallel_runtime.worker_init_from_explainer_spec` which builds per-worker harness and exposes `explain_slice`.
4. Task payloads: change instance-parallel tasks to pass only `(start, stop, state)` and rely on harness; keep a local-fallback path.
5. Explainer API: add `initialize_pool()` and `close()` and optional `pool_at_init` opt-in in `CalibratedExplainer`.
6. `CalibratedExplainer` changes: must only be a thin delegator, no logic allowed in the delegator. 
7. Tests & profiling: integration tests for correctness and simple perf checks; cross-platform memory check.

Acceptance criteria
- Unit tests for initializer plumbing pass.
- Integration test shows identical outputs between sequential and initializer-based parallel execution on small dataset.
- CI-friendly tests for Windows spawn semantics (or explicit skip if runner absent).
- Benchmarks show marked reduction in per-task pickling overhead on representative test (documented).

Notes
- Keep initializer args compact/serialisable.
- Document memory trade-offs and require explicit `close()` or context usage for long-running processes.

