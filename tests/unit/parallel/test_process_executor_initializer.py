from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


def test_should_forward_initializer_to_processpool(monkeypatch):
    cfg = ParallelConfig(enabled=True, strategy="processes", min_batch_size=1)

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

    import calibrated_explanations.parallel.parallel as perf_parallel

    monkeypatch.setattr(perf_parallel, "ProcessPoolExecutor", RecordingProcessPool, raising=False)

    with ParallelExecutor(cfg) as executor:
        assert executor is not None

    assert captured.get("initializer") is dummy_initializer
    assert captured.get("initargs") == ("arg1", 123)


def test_should_not_change_default_behavior_when_no_initializer(monkeypatch):
    cfg = ParallelConfig(enabled=True, strategy="processes", min_batch_size=1)

    captured = {}

    class RecordingProcessPool:
        def __init__(self, *args, **kwargs):
            captured["initializer"] = kwargs.get("initializer")
            captured["initargs"] = kwargs.get("initargs")
            self._max_workers = kwargs.get("max_workers", 1)

        def map(self, fn, items, chunksize=None):
            return list(map(fn, items))

        def shutdown(self, wait=True):
            return None

    import calibrated_explanations.parallel.parallel as perf_parallel

    monkeypatch.setattr(perf_parallel, "ProcessPoolExecutor", RecordingProcessPool, raising=False)

    with ParallelExecutor(cfg) as executor:
        results = executor.map(lambda x: x + 1, [1, 2, 3])

    assert results == [2, 3, 4]
    assert captured.get("initializer") is None
