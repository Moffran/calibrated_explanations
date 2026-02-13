import importlib
import warnings

from calibrated_explanations.parallel import parallel as canonical


def test_perf_parallel_shim_warns_and_forwards(monkeypatch):
    monkeypatch.syspath_prepend(".")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        perf_parallel = importlib.reload(
            importlib.import_module("calibrated_explanations.perf.parallel")
        )
    assert perf_parallel.ParallelExecutor.__name__ == canonical.ParallelExecutor.__name__
    assert perf_parallel.ParallelExecutor.__module__ == canonical.ParallelExecutor.__module__
    assert perf_parallel.ParallelConfig.__name__ == canonical.ParallelConfig.__name__
    assert perf_parallel.ParallelConfig.__module__ == canonical.ParallelConfig.__module__
    assert perf_parallel.ParallelMetrics.__name__ == canonical.ParallelMetrics.__name__
    assert perf_parallel.ParallelMetrics.__module__ == canonical.ParallelMetrics.__module__
