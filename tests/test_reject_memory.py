"""Memory-oriented tests for reject wrappers."""

from __future__ import annotations

import ctypes
import gc
import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectCalibratedExplanations


def current_rss_bytes() -> int:
    """Return resident memory for the current process in bytes."""
    if os.name == "nt":
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi")
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        get_process_memory_info.restype = ctypes.c_int
        ok = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            pytest.skip("Unable to read process RSS on Windows")
        return int(counters.WorkingSetSize)

    # POSIX fallback
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        pytest.skip("Unable to read process RSS on this platform")
    # Linux reports KiB, macOS reports bytes.
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def test_calibration_set_reused_not_duplicated():
    rng_x = np.random.RandomState(0)
    rng_y = np.random.RandomState(1)
    x = rng_x.randn(2000, 20)
    y = rng_y.randint(0, 2, size=len(x))

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:1200], y[:1200])
    wrapper.calibrate(x[1200:], y[1200:], seed=0)

    base = wrapper.explain_factual(x[1200:1210], reject_policy=RejectPolicy.FLAG)
    wrapped = RejectCalibratedExplanations.from_collection(
        base,
        base.metadata_full(),
        RejectPolicy.FLAG,
        base.rejected,
    )

    assert wrapped.calibrated_explainer is base.calibrated_explainer
    assert id(wrapped.calibrated_explainer.x_cal) == id(
        base.calibrated_explainer.x_cal
    ) or np.shares_memory(
        wrapped.calibrated_explainer.x_cal,
        base.calibrated_explainer.x_cal,
    )

    for name in ("x_cal", "_X_cal", "scaled_x_cal", "fast_x_cal", "scaled_y_cal"):
        assert not hasattr(wrapped, name)


def test_should_stabilize_rss_after_warmup_when_reject_flow_repeats():
    """should_keep_process_rss_within_threshold_after_repeated_reject_flow_calls."""
    rng_x = np.random.RandomState(12)
    rng_y = np.random.RandomState(21)
    x = rng_x.randn(500, 8)
    y = rng_y.randint(0, 2, size=len(x))

    model = RandomForestClassifier(n_estimators=6, random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:300], y[:300])
    wrapper.calibrate(x[300:], y[300:], seed=0)
    query = x[300:302]

    rss_samples: list[int] = []
    for iteration in range(200):
        if iteration % 2 == 0:
            _ = wrapper.explain_factual(query, reject_policy=RejectPolicy.FLAG)
        else:
            _ = wrapper.predict(query)
        if iteration % 10 == 0:
            gc.collect()
        rss_samples.append(current_rss_bytes())

    steady_state = rss_samples[50:]
    baseline = steady_state[0]
    peak = max(steady_state)
    abs_growth = peak - baseline
    rel_growth = abs_growth / max(1, baseline)

    assert abs_growth <= 64 * 1024 * 1024
    assert rel_growth <= 0.10
