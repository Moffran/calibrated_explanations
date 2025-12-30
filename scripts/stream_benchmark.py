"""Simple streaming benchmark utility.

Measures elapsed time and peak Python memory for serialising N synthetic
explanations into the streaming API. Intended as a lightweight check rather
than a full CI-grade benchmark.

Usage:
    python scripts/stream_benchmark.py --n 10000 --chunk 256 --format jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tracemalloc
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, "src")

from calibrated_explanations.explanations import CalibratedExplanations


class DummyCalibratedExplainer:
    def __init__(self, num_features=10):
        self.feature_names = [f"f{j}" for j in range(num_features)]
        self.class_labels = {1: "class_one", 0: "class_zero"}
        self.num_features = num_features
        self.mode = "classification"
        self.y_cal = [0.0, 1.0]
        self.x_cal = [[0.0] * num_features, [1.0] * num_features]
        self.categorical_features = []
        self.categorical_labels = []
        self.feature_values = []


class DummyExplanation:
    def __init__(self, index, x, predict, interval, feature_weights):
        self.index = index
        self.x = x
        self.predict = predict
        self.prediction_interval = interval
        self.feature_weights = feature_weights
        self.rules = {"ensured": [f"feature_{i} > {random.random()}" for i in range(5)]}


def make_calibrated_explanations(n: int) -> CalibratedExplanations:
    dummy_explainer = DummyCalibratedExplainer()
    x = np.array([[random.random() for _ in range(10)] for _ in range(n)])
    bins = ["bin0"] * n
    collection = CalibratedExplanations(dummy_explainer, x, 0.5, bins)
    collection.explanations = [
        DummyExplanation(
            i, x[i], predict=random.random(), interval=(random.random(), random.random()),
            feature_weights=[random.random() for _ in range(10)]
        )
        for i in range(n)
    ]
    collection.low_high_percentiles = (10.0, 90.0)
    return collection


def run(n: int, chunk: int, fmt: str):
    print(f"Creating {n} dummy explanations...", file=sys.stderr)
    collection = make_calibrated_explanations(n)

    tracemalloc.start()
    start = time.time()

    chunks = list(collection.to_json_stream(chunk_size=chunk, format=fmt))

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    elapsed = time.time() - start

    # Output the telemetry from the stream
    telemetry_chunk = chunks[-1]
    telemetry = json.loads(telemetry_chunk)

    results = {
        "n": n,
        "chunk_size": chunk,
        "format": fmt,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
        "telemetry": telemetry
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--format", choices=("jsonl", "chunked"), default="jsonl")
    args = p.parse_args()
    run(args.n, args.chunk, args.format)
