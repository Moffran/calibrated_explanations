from __future__ import annotations

from tests.helpers.doc_utils import run_quickstart_classification


def test_tune_runtime_performance_snippet(enable_fallbacks):
    from calibrated_explanations import WrapCalibratedExplainer
    from calibrated_explanations.api.config import ExplainerBuilder

    context = run_quickstart_classification()
    model = context.explainer.learner
    builder = ExplainerBuilder(model)
    config = builder.perf_cache(
        True,
        max_items=256,
        max_bytes=8 * 1024 * 1024,
        namespace="service-a",
        version="v2",
        ttl=600,
    ).build_config()
    explainer = WrapCalibratedExplainer._from_config(config)

    config_parallel = (
        builder.perf_parallel(True, backend="threads", workers=4, min_batch=8)
        .perf_cache(True)
        .build_config()
    )
    explainer_parallel = WrapCalibratedExplainer._from_config(config_parallel)

    assert explainer
    assert explainer_parallel
