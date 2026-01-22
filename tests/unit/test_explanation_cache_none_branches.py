from calibrated_explanations.cache.explanation_cache import ExplanationCacheFacade


def test_explanation_cache_facade_none_branches_no_errors():
    facade = ExplanationCacheFacade(None)
    # when cache is None many methods should be safe no-ops or return None
    assert facade.get_calibration_summaries(explainer_id="x", x_cal_hash="h") is None
    assert facade.get_feature_names_cache(explainer_id="x") is None

    # setters should be no-ops
    facade.set_calibration_summaries(
        explainer_id="x",
        x_cal_hash="h",
        categorical_counts={0: {}},
        numeric_sorted={0: []},
    )
    facade.set_feature_names_cache(explainer_id="x", feature_names=("a",))

    # invalidate/reset should be safe when cache is None
    facade.invalidate_all()
    facade.reset_version("v")
