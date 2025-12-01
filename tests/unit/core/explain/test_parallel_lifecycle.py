
import os
import pytest
import time
from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor

def square(x):
    return x * x

def slow_square(x):
    time.sleep(0.01)
    return x * x

def nested_task(x):
    # Simulate nested parallelism by creating a new executor
    # Note: Passing the original executor is hard due to pickling
    config = ParallelConfig(enabled=True, strategy="threads", max_workers=2)
    executor = ParallelExecutor(config)
    return sum(executor.map(square, range(x)))

class TestParallelLifecycle:
    
    @pytest.mark.parametrize("strategy", ["threads", "processes", "joblib"])
    def test_strategies(self, strategy):
        if strategy == "joblib":
            pytest.importorskip("joblib")
            
        config = ParallelConfig(enabled=True, strategy=strategy, max_workers=2, min_batch_size=1)
        executor = ParallelExecutor(config)
        
        results = executor.map(square, range(10))
        assert results == [x*x for x in range(10)]

    def test_context_manager(self):
        config = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
        
        with ParallelExecutor(config) as executor:
            results1 = executor.map(square, range(5))
            results2 = executor.map(square, range(5, 10))
            
        assert results1 == [x*x for x in range(5)]
        assert results2 == [x*x for x in range(5, 10)]

    def test_chunking(self):
        # Verify chunksize parameter is accepted
        config = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
        executor = ParallelExecutor(config)
        
        # We can't easily verify chunking happened without mocking, but we can verify it runs
        results = executor.map(square, range(10), chunksize=2)
        assert results == [x*x for x in range(10)]

    def test_force_serial_on_failure(self):
        config = ParallelConfig(enabled=True, strategy="threads", force_serial_on_failure=True, min_batch_size=1)
        executor = ParallelExecutor(config)
        
        # Inject a failure in strategy resolution to trigger fallback
        # We can mock _resolve_strategy or just rely on the fact that if we pass invalid strategy it might fail?
        # But ParallelConfig validates strategy enum? No, it's a string.
        
        # Let's mock _resolve_strategy
        original_resolve = executor._resolve_strategy
        
        def failing_resolve():
            print("Failing resolve called")
            raise RuntimeError("Simulated failure")
            
        executor._resolve_strategy = failing_resolve
        
        print("Calling map")
        results = executor.map(square, range(5))
        print(f"Results: {results}")
        print(f"Metrics: {executor.metrics}")
        
        assert results == [x*x for x in range(5)]
        assert executor.metrics.fallbacks == 1
        
        executor._resolve_strategy = original_resolve

    @pytest.mark.skipif(os.name == "nt", reason="Nested processes on Windows are slow/complex")
    def test_nested_parallelism_threads(self):
        # Threads inside threads should work
        config = ParallelConfig(enabled=True, strategy="threads", max_workers=2)
        executor = ParallelExecutor(config)
        
        results = executor.map(nested_task, [1, 2, 3])
        # nested_task(1) -> sum([0]) = 0
        # nested_task(2) -> sum([0, 1]) = 1
        # nested_task(3) -> sum([0, 1, 4]) = 5
        assert results == [0, 1, 5]

