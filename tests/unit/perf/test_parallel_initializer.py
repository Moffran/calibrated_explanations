import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


def test_process_pool_receives_initializer_and_initargs(monkeypatch):
    cfg = ParallelConfig(enabled=True, strategy="processes", min_batch_size=1)

    # Attach the new opt-in initializer attributes (TDD-first; implementation pending)
    def dummy_initializer(spec):
        return None

    cfg.worker_initializer = dummy_initializer
    cfg.worker_init_args = ("arg1", 123)

    captured = {}

    class RecordingProcessPool:
        def __init__(self, *args, **kwargs):
            captured["initializer"] = kwargs.get("initializer")
            captured["initargs"] = kwargs.get("initargs")

        def shutdown(self, wait=True):
            return None

    # Monkeypatch the ProcessPoolExecutor used by the parallel facade
    import calibrated_explanations.parallel.parallel as perf_parallel

    monkeypatch.setattr(
        perf_parallel, "ProcessPoolExecutor", RecordingProcessPool, raising=False
    )

    # Using the context manager will trigger pool initialization path
    with ParallelExecutor(cfg) as executor:
        assert executor is not None

    # Expect the pool to be constructed with initializer/initargs (TDD: will be implemented)
    assert captured.get("initializer") is dummy_initializer
    assert captured.get("initargs") == ("arg1", 123)


def test_worker_initializer_creates_explain_slice_harness():
    # The explain runtime should expose a worker initializer that, when invoked
    # in a worker process, installs a module-global `explain_slice` callable.
    import calibrated_explanations.core.explain.parallel_runtime as pr_mod

    worker_init = getattr(pr_mod, "worker_init_from_explainer_spec", None)
    assert worker_init is not None, "Expected worker_init_from_explainer_spec to be present"

    # Simulate worker init with a compact explainer spec
    fake_spec = {"dummy": True}
    worker_init(fake_spec)

    # After initialization, a module-level `explain_slice` should be available
    assert hasattr(pr_mod, "explain_slice")
    assert callable(getattr(pr_mod, "explain_slice"))
