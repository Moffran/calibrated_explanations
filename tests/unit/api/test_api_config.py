from calibrated_explanations.api.config import ExplainerBuilder


def test_builder_defaults_and_fluent():
    b = ExplainerBuilder(model=object())
    cfg = b.task("classification").low_high_percentiles((10, 90)).build_config()
    assert cfg.task == "classification"
    assert cfg.low_high_percentiles == (10, 90)
    # perf factory can be None but must exist as attribute
    assert hasattr(cfg, "_perf_factory")
