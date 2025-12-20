Pool-at-Explainer-Init — Tasks (TDD-first, concise)

Start by writing tests for each numbered task below. Each task lists the minimal verification test(s).

1) Tests: initializer plumbing
- Write `tests/unit/parallel/test_process_executor_initializer.py` with two tests:
  - `should_forward_initializer_to_processpool()` — mock `concurrent.futures.ProcessPoolExecutor` and assert `initializer` and `initargs` are passed.
  - `should_not_change_default_behavior_when_no_initializer()` — run small map with default `ParallelExecutor`.

2) Add `ParallelConfig` fields
- Implement `worker_initializer: Optional[Callable]=None` and `worker_init_args: Optional[Tuple]=None` on `ParallelConfig`.
- Test: `should_store_worker_initializer_in_config()` — simple instantiation assertion.

3) Forward initializer to ProcessPoolExecutor
- Update `_process_strategy` to pass `initializer`/`initargs` when present.
- Test: reuse (1) to verify.

4) Worker runtime initializer
- Add `parallel_runtime.worker_init_from_explainer_spec(serialized_spec)` that sets a module-global harness with `explain_slice(start, stop, state)`.
- Test: `should_construct_worker_harness()` — call initializer in-process and assert harness exists and returns expected values for a toy explainer spec.

5) Task payload refactor (hot path)
- Change `_instance_parallel_task` to accept only `(start, stop, serialized_request_state)` and call `parallel_runtime._worker_harness.explain_slice(...)` if present; otherwise create local fallback.
- Test: `should_run_with_harness_and_without_harness()` — unit tests for both paths.

6) Explainer lifecycle API
- Add `CalibratedExplainer.initialize_pool(n_workers=None, *, pool_at_init: bool=False)` and `close()` plus context manager methods.
- Test: `initialize_pool_creates_pool()` and `close_releases_resources()`.

7) Integration & cross-platform
- Add an integration test `tests/integration/test_initializer_parallel.py` which:
  - Starts a small explainer with `pool_at_init=True` and initializer set to `parallel_runtime.worker_init_from_explainer_spec`.
  - Runs `explain()` sequentially and with the initializer-based parallel path; asserts outputs equal and measures that pickling time is lower (high-level assertion).
- Add Windows spawn semantics test (mark skip if runner absent).

8) Benchmark & doc
- Run `evaluation/chunk_size_ablation.py` and `evaluation/parallel_ablation.py` with the new mode; document results in `docs/improvement/` and add CHANGELOG entry.

Order of work
- Implement and run tests in the exact numeric order above.
- Fix smallest failing test and iterate until all tests pass.

