import importlib
import warnings

from calibrated_explanations.parallel import parallel as canonical


def test_perf_parallel_shim_warns_and_forwards(monkeypatch):
    monkeypatch.syspath_prepend(".")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        perf_parallel = importlib.reload(importlib.import_module("calibrated_explanations.perf.parallel"))
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    assert perf_parallel.ParallelExecutor is canonical.ParallelExecutor
    assert perf_parallel.ParallelConfig is canonical.ParallelConfig
    assert perf_parallel.ParallelMetrics is canonical.ParallelMetrics
