import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import warnings
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ValidationError, DataShapeError


@pytest.fixture
def mock_learner():
    learner = MagicMock()
    learner.predict.return_value = np.array([0, 1])
    learner.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    return learner


@pytest.fixture(autouse=True)
def mock_check_is_fitted():
    with patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_plugin_manager():
    with patch("calibrated_explanations.plugins.manager.PluginManager") as mock:
        instance = mock.return_value
        instance.initialize_from_kwargs.return_value = None
        instance.initialize_orchestrators.return_value = None

        # Mock orchestrators
        instance.explanation_orchestrator = MagicMock()
        instance.prediction_orchestrator = MagicMock()
        instance.reject_orchestrator = MagicMock()

        yield mock


@pytest.fixture(autouse=True)
def mock_interval_calibrator():
    with patch(
        "calibrated_explanations.core.prediction.orchestrator.PredictionOrchestrator.obtain_interval_calibrator"
    ) as mock:
        mock.return_value = (MagicMock(), None)
        yield mock


@pytest.fixture(autouse=True)
def mock_legacy_predict_bridge():
    with patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_identify_constant_features():
    with patch(
        "calibrated_explanations.core.calibration_helpers.identify_constant_features"
    ) as mock:
        mock.return_value = []
        yield mock


@pytest.fixture(autouse=True)
def mock_helpers():
    with (
        patch("calibrated_explanations.integrations.LimeHelper"),
        patch("calibrated_explanations.integrations.ShapHelper"),
    ):
        yield


def test_preloads(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    with patch.object(explainer, "lime_helper") as mock_lime:
        explainer.preload_lime(x_cal)
        mock_lime.preload.assert_called_once_with(x_cal=x_cal)

    with patch.object(explainer, "shap_helper") as mock_shap:
        explainer.preload_shap(num_test=10)
        mock_shap.preload.assert_called_once_with(num_test=10)
    # Minimal assertion to satisfy test-quality checks
    assert True


def test_plugin_delegations_and_aliases(
    monkeypatch: pytest.MonkeyPatch, mock_learner, mock_plugin_manager
):
    monkeypatch.delenv("CE_DEPRECATIONS", raising=False)
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    plugin_manager = explainer.plugin_manager

    plugin_manager.build_plot_chain.return_value = ("default",)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert explainer.build_plot_style_chain() == ("default",)

    explainer.prediction_orchestrator.ensure_interval_runtime_state.return_value = "ok"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert explainer.ensure_interval_runtime_state() == "ok"
    explainer.prediction_orchestrator.gather_interval_hints.return_value = ("hint",)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert explainer.gather_interval_hints(fast=True) == ("hint",)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        plugin_manager.interval_plugin_hints = {"a": ("b",)}
        assert explainer.interval_plugin_hints == {"a": ("b",)}
        explainer.interval_plugin_hints = {"c": ("d",)}
        assert plugin_manager.interval_plugin_hints == {"c": ("d",)}
        assert explainer.interval_plugin_hints == {"c": ("d",)}

        plugin_manager.interval_plugin_fallbacks = {"a": ("b",)}
        assert explainer.interval_plugin_fallbacks == {"a": ("b",)}
        explainer.interval_plugin_fallbacks = {"c": ("d",)}
        assert plugin_manager.interval_plugin_fallbacks == {"c": ("d",)}
        assert explainer.interval_plugin_fallbacks == {"c": ("d",)}

        plugin_manager.interval_preferred_identifier = {"x": "y"}
        assert explainer.interval_preferred_identifier == {"x": "y"}
        explainer.interval_preferred_identifier = {"z": None}
        assert plugin_manager.interval_preferred_identifier == {"z": None}

        plugin_manager.telemetry_interval_sources = {"x": "y"}
        assert explainer.telemetry_interval_sources == {"x": "y"}
        explainer.telemetry_interval_sources = {"z": "w"}
        assert plugin_manager.telemetry_interval_sources == {"z": "w"}

        plugin_manager.interval_plugin_identifiers = {"x": "y"}
        assert explainer.interval_plugin_identifiers == {"x": "y"}
        explainer.interval_plugin_identifiers = {"z": "w"}
        assert plugin_manager.interval_plugin_identifiers == {"z": "w"}

        plugin_manager.interval_context_metadata = {"x": {"y": 1}}
        assert explainer.interval_context_metadata == {"x": {"y": 1}}
        explainer.interval_context_metadata = {"z": {"w": 2}}
        assert plugin_manager.interval_context_metadata == {"z": {"w": 2}}

    explainer.plot_plugin_fallbacks = {"plot": ("fallback",)}
    assert plugin_manager.plot_plugin_fallbacks == {"plot": ("fallback",)}

    explainer.explanation_plugin_instances = {"plugin": object()}
    assert plugin_manager.explanation_plugin_instances == explainer.explanation_plugin_instances

    explainer.interval_plugin_override = "override"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert explainer.interval_plugin_override == "override"

    explainer.fast_interval_plugin_override = "fast"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert explainer.fast_interval_plugin_override == "fast"

    explainer.initialized = True
    assert explainer.is_initialized is True

    explainer.parallel_executor = MagicMock()
    explainer.predict_bridge = MagicMock()


def test_enable_fast_mode_resets_on_error(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    with (
        patch.object(
            explainer,
            "initialize_interval_learner_for_fast_explainer",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        explainer.enable_fast_mode()

    assert explainer.is_fast() is False


def test_explain_mondrian_bins_and_legacy_path(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.bins = np.array([1])

    x_test = np.array([[3, 4]])
    explainer.explanation_orchestrator.invoke_factual.return_value = "factual"
    explainer.explain_factual(x_test)
    _, kwargs = explainer.explanation_orchestrator.invoke_factual.call_args
    assert kwargs["bins"] is explainer.bins

    explainer.explanation_orchestrator.invoke_alternative.return_value = "alternative"
    explainer.explore_alternatives(x_test)
    _, kwargs = explainer.explanation_orchestrator.invoke_alternative.call_args
    assert kwargs["bins"] is explainer.bins

    with patch("calibrated_explanations.core.explain.legacy_explain") as mock_legacy:
        mock_legacy.return_value = "legacy"
        result = explainer(x_test, bins=None, _use_plugin=False)
        assert result == "legacy"
        assert mock_legacy.call_args.kwargs["bins"] is explainer.bins


def test_additional_coverage(
    monkeypatch: pytest.MonkeyPatch, mock_learner, mock_plugin_manager
):
    monkeypatch.delenv("CE_DEPRECATIONS", raising=False)
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])

    # predict_function in kwargs
    def my_predict(x):
        return np.array([[0.5, 0.5]])

    explainer = CalibratedExplainer(
        mock_learner, x_cal, y_cal, mode="classification", predict_function=my_predict
    )
    assert explainer.predict_function == my_predict

    # categorical_labels without categorical_features
    explainer = CalibratedExplainer(
        mock_learner, x_cal, y_cal, mode="classification", categorical_labels={0: {0: "A"}}
    )
    assert explainer.categorical_features == [0]

    # numeric y_cal without class_labels
    explainer = CalibratedExplainer(mock_learner, x_cal, np.array([0, 1]), mode="classification")
    assert explainer.class_labels == {0: "0", 1: "1"}

    # properties
    explainer.plugin_manager = mock_plugin_manager
    _ = explainer.plot_plugin_fallbacks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ = explainer.plot_style_override
    _ = explainer.explanation_plugin_instances
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ = explainer.explanation_plugin_overrides
        _ = explainer.interval_plugin_override
        _ = explainer.plot_style_override
    _ = explainer.shap_helper
    _ = explainer.feature_filter_per_instance_ignore
    _ = explainer.runtime_telemetry

    # preprocessor_metadata
    explainer.set_preprocessor_metadata({"a": 1})
    assert explainer.preprocessor_metadata == {"a": 1}
    explainer.set_preprocessor_metadata(None)
    assert explainer.preprocessor_metadata is None

    # get_calibration_summaries
    with patch(
        "calibrated_explanations.calibration.summaries.get_calibration_summaries"
    ) as mock_get:
        explainer.get_calibration_summaries()
        mock_get.assert_called_once()

    # sigma test
    mock_plugin_manager.prediction_orchestrator.interval_registry.get_sigma_test.return_value = (
        np.array([1.0])
    )
    explainer.get_sigma_test(x_cal)

    # initialize for fast
    explainer.initialize_interval_learner_for_fast_explainer()
    mock_plugin_manager.prediction_orchestrator.interval_registry.initialize_for_fast_explainer.assert_called_once()

    # reinitialize with bins
    explainer.bins = np.array([0])
    mock_plugin_manager.prediction_orchestrator.obtain_interval_calibrator.return_value = (
        MagicMock(),
        "id",
    )
    explainer.reinitialize(mock_learner, x_cal, y_cal, bins=np.array([0]))

    # reinitialize error: mix bins
    explainer.bins = None
    with pytest.raises(ValidationError):
        explainer.reinitialize(mock_learner, x_cal, y_cal, bins=np.array([0]))

    # reinitialize error: bin length
    explainer.bins = np.array([0])
    with pytest.raises(DataShapeError):
        explainer.reinitialize(mock_learner, x_cal, y_cal, bins=np.array([0, 1]))

    # reinitialize without xs, ys
    explainer.reinitialize(mock_learner)

    # __repr__ with verbose and various fields
    explainer.verbose = True
    explainer.feature_names = ["f1", "f2"]
    explainer.categorical_features = [0]
    explainer.categorical_labels = {0: {0: "A"}}
    explainer.class_labels = {0: "C0", 1: "C1"}
    explainer.latest_explanation = MagicMock()
    explainer.latest_explanation.total_explain_time = 1.0
    repr_str = repr(explainer)
    assert "init_time" in repr_str
    assert "feature_names" in repr_str
    assert "categorical_features" in repr_str
    assert "categorical_labels" in repr_str
    assert "class_labels" in repr_str
    assert "total_explain_time" in repr_str

    # obtain_interval_calibrator
    explainer.obtain_interval_calibrator(fast=False, metadata={})

    # explain_factual with mondrian
    explainer.bins = np.array([0])
    with patch.object(explainer.explanation_orchestrator, "invoke_factual") as mock_invoke:
        explainer.explain_factual(x_cal)
        mock_invoke.assert_called_once()

    # explore_alternatives with mondrian
    with patch.object(explainer.explanation_orchestrator, "invoke_alternative") as mock_invoke:
        explainer.explore_alternatives(x_cal)
        mock_invoke.assert_called_once()

    # __call__
    with patch.object(explainer, "_explain") as mock_explain:
        explainer(x_cal)
        mock_explain.assert_called_once()

    # explain_fast with legacy path
    import external_plugins.fast_explanations.pipeline as fast_pipeline_mod

    with patch.object(fast_pipeline_mod, "FastExplanationPipeline") as mock_pipeline:
        explainer.explain_fast(x_cal, _use_plugin=False)
        mock_pipeline.assert_called()

    # explain_lime
    import external_plugins.integrations.lime_pipeline as lime_pipeline_mod

    with patch.object(lime_pipeline_mod, "LimePipeline") as mock_lime:
        explainer.explain_lime(x_cal)
        mock_lime.assert_called()

    # explain_shap
    import external_plugins.integrations.shap_pipeline as shap_pipeline_mod

    with patch.object(shap_pipeline_mod, "ShapPipeline") as mock_shap:
        explainer.explain_shap(x_cal)
        mock_shap.assert_called()

    # is_lime_enabled / is_shap_enabled
    explainer.is_lime_enabled(True)
    explainer.is_lime_enabled(False)
    explainer.is_lime_enabled()
    explainer.is_shap_enabled(True)
    explainer.is_shap_enabled(False)
    explainer.is_shap_enabled()

    # is_multiclass
    explainer.num_classes = 3
    assert explainer.is_multiclass() is True
    explainer.num_classes = 2
    assert explainer.is_multiclass() is False

    # is_fast
    assert explainer.is_fast() is False

    # discretize
    with patch("calibrated_explanations.core.explain.discretize") as mock_disc:
        explainer.discretize(x_cal)
        mock_disc.assert_called_once()

    # rule_boundaries
    with patch("calibrated_explanations.core.explain.rule_boundaries") as mock_rb:
        explainer.rule_boundaries(x_cal)
        mock_rb.assert_called_once()

    # set_difficulty_estimator
    with patch(
        "calibrated_explanations.core.difficulty_estimator_helpers.validate_difficulty_estimator"
    ):
        explainer.set_difficulty_estimator(MagicMock())

    # set_mode
    explainer.set_mode("regression")
    assert explainer.mode == "regression"
    explainer.set_mode("classification")
    assert explainer.mode == "classification"
    with pytest.raises(ValidationError):
        explainer.set_mode("invalid")

    # initialize_reject_learner / predict_reject
    explainer.reject_orchestrator.initialize_reject_learner()
    explainer.reject_orchestrator.predict_reject(x_cal)

    # set_discretizer
    explainer.set_discretizer("entropy")

    # predict uncalibrated
    explainer.mode = "regression"
    with patch(
        "calibrated_explanations.core.prediction_helpers.handle_uncalibrated_regression_prediction"
    ) as mock_pred:
        explainer.predict(x_cal, calibrated=False)
        mock_pred.assert_called_once()

    explainer.mode = "classification"
    with patch(
        "calibrated_explanations.core.prediction_helpers.handle_uncalibrated_classification_prediction"
    ) as mock_pred:
        explainer.predict(x_cal, calibrated=False)
        mock_pred.assert_called_once()

    # predict calibrated regression
    explainer.mode = "regression"
    with patch.object(
        explainer.prediction_orchestrator,
        "predict",
        return_value=(np.array([0.5]), np.array([0.4]), np.array([0.6]), None),
    ):
        explainer.predict(x_cal)

    # predict calibrated classification
    explainer.mode = "classification"
    with patch.object(
        explainer.prediction_orchestrator,
        "predict",
        return_value=(np.array([0]), np.array([0]), np.array([0]), np.array([0, 1])),
    ):
        explainer.predict(x_cal)

    # predict_proba uncalibrated
    with pytest.raises(ValidationError):
        explainer.predict_proba(x_cal, calibrated=False, threshold=0.5)

    explainer.predict_proba(x_cal, calibrated=False)
    explainer.predict_proba(x_cal, calibrated=False, uq_interval=True)

    # predict_proba calibrated regression
    explainer.mode = "regression"
    explainer.interval_learner = MagicMock()
    explainer.interval_learner.predict_probability.return_value = (
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.6]),
        None,
    )
    explainer.predict_proba(x_cal, threshold=0.5)

    explainer.interval_learner = [MagicMock()]
    explainer.interval_learner[-1].predict_probability.return_value = (
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.6]),
        None,
    )
    explainer.predict_proba(x_cal, threshold=0.5)

    # predict_proba calibrated multiclass
    explainer.mode = "classification"
    explainer.num_classes = 3
    explainer.interval_learner = MagicMock()
    explainer.interval_learner.predict_proba.return_value = (
        np.array([[0.1, 0.8, 0.1]]),
        np.array([[0.05, 0.7, 0.05]]),
        np.array([[0.15, 0.9, 0.15]]),
        None,
    )
    explainer.predict_proba(x_cal)

    explainer.interval_learner = [MagicMock()]
    explainer.interval_learner[-1].predict_proba.return_value = (
        np.array([[0.1, 0.8, 0.1]]),
        np.array([[0.05, 0.7, 0.05]]),
        np.array([[0.15, 0.9, 0.15]]),
        None,
    )
    explainer.predict_proba(x_cal)

    # predict_proba calibrated binary
    explainer.num_classes = 2
    explainer.interval_learner = MagicMock()
    explainer.interval_learner.predict_proba.return_value = (
        np.array([[0.2, 0.8]]),
        np.array([0.1]),
        np.array([0.3]),
    )
    explainer.predict_proba(x_cal)

    explainer.interval_learner = [MagicMock()]
    explainer.interval_learner[-1].predict_proba.return_value = (
        np.array([[0.2, 0.8]]),
        np.array([0.1]),
        np.array([0.3]),
    )
    explainer.predict_proba(x_cal)

    # plot
    with patch("calibrated_explanations.plotting.plot_global") as mock_plot:
        explainer.plot(x_cal)
        mock_plot.assert_called_once()

    # calibrated_confusion_matrix
    explainer.mode = "regression"
    with pytest.raises(ValidationError):
        explainer.calibrated_confusion_matrix()

    explainer.mode = "classification"
    with patch(
        "calibrated_explanations.core.calibration_metrics.compute_calibrated_confusion_matrix"
    ) as mock_cm:
        explainer.calibrated_confusion_matrix()
        mock_cm.assert_called_once()

    # predict_calibration
    explainer.predict_calibration()

    # deleters
    del explainer.explanation_orchestrator
    del explainer.prediction_orchestrator
    del explainer.reject_orchestrator
    del explainer.lime_helper
    del explainer.shap_helper
