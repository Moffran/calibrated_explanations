import pytest
import warnings
from types import MappingProxyType
from unittest.mock import MagicMock, patch
import numpy as np
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.utils.exceptions import NotFittedError
from calibrated_explanations.utils.exceptions import (
    DataShapeError,
    ConfigurationError,
    ValidationError,
)
from calibrated_explanations.plugins.intervals import (
    IntervalCalibratorContext,
    RegressionIntervalCalibrator,
    ClassificationIntervalCalibrator,
)
from calibrated_explanations.utils import exceptions as core_exceptions


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.plugin_manager = MagicMock()
    explainer.perf_cache = None
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.is_fast.return_value = False
    # Use a simple attribute for initialized to allow toggling in tests
    explainer.initialized = True
    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"
    ) as mock_registry:
        orchestrator = PredictionOrchestrator(mock_explainer)
        orchestrator.interval_registry = mock_registry.return_value
        return orchestrator


def test_init(mock_explainer):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"
    ) as mock_registry:
        orchestrator = PredictionOrchestrator(mock_explainer)
        assert orchestrator.explainer == mock_explainer
        mock_registry.assert_called_once_with(mock_explainer)


def test_initialize_chains(orchestrator, mock_explainer):
    orchestrator.initialize_chains()
    mock_explainer.plugin_manager.initialize_chains.assert_called_once()


def test_predict_delegates(orchestrator):
    x = np.array([[1, 2]])
    with patch.object(orchestrator, "_predict") as mock_predict:
        orchestrator.predict(x, threshold=0.5)
        mock_predict.assert_called_once()
        args, kwargs = mock_predict.call_args
        assert args[0] is x
        assert kwargs["threshold"] == 0.5


def test_predict_caching(orchestrator, mock_explainer):
    x = np.array([[1, 2]])
    mock_cache = MagicMock()
    mock_cache.enabled = True
    mock_cache.get.return_value = "cached_result"
    mock_explainer.perf_cache = mock_cache

    result = orchestrator.predict(x)
    assert result == "cached_result"
    mock_cache.get.assert_called_once()


def test_predict_caching_miss(orchestrator, mock_explainer):
    x = np.array([[1, 2]])
    mock_cache = MagicMock()
    mock_cache.enabled = True
    mock_cache.get.return_value = None
    mock_explainer.perf_cache = mock_cache

    with patch.object(orchestrator, "predict_impl") as mock_impl:
        mock_impl.return_value = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)

        result = orchestrator.predict(x)

        mock_cache.get.assert_called_once()
        mock_impl.assert_called_once()
        mock_cache.set.assert_called_once()
        assert result == mock_impl.return_value


def test_validate_prediction_result_valid(orchestrator):
    # predict, low, high, classes
    result = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        orchestrator.validate_prediction_result(result)
        assert len(record) == 0


def test_validate_prediction_result_invalid_low_high(orchestrator):
    # low > high
    result = (np.array([0.5]), np.array([0.7]), np.array([0.6]), None)
    with pytest.warns(UserWarning, match="Prediction interval invariant violated"):
        orchestrator.validate_prediction_result(result)


def test_validate_prediction_result_invalid_predict(orchestrator):
    # predict > high
    result = (np.array([0.8]), np.array([0.4]), np.array([0.6]), None)
    with pytest.warns(UserWarning, match="Prediction invariant violated"):
        orchestrator.validate_prediction_result(result)


def testpredict_impl_not_fitted(orchestrator, mock_explainer):
    mock_explainer.initialized = False
    with pytest.raises(NotFittedError):
        orchestrator.predict(np.array([[1]]))


def testpredict_impl_binary_classification(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = False
    mock_explainer.is_fast.return_value = False

    mock_learner = MagicMock()
    mock_learner.predict_proba.return_value = (
        np.array([[0.1, 0.9]]),  # predict
        np.array([[0.0, 0.8]]),  # low
        np.array([[0.2, 1.0]]),  # high
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x)

    assert np.allclose(predict, [0.9])
    assert np.allclose(low, [[0.0, 0.8]])
    assert np.allclose(high, [[0.2, 1.0]])
    assert classes is None


def testpredict_impl_multiclass_classification(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = True
    mock_explainer.is_fast.return_value = False

    mock_learner = MagicMock()
    mock_learner.predict_proba.return_value = (
        np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]]),  # predict
        np.array([[0.0, 0.1, 0.6], [0.7, 0.0, 0.0], [0.2, 0.2, 0.3]]),  # low
        np.array([[0.2, 0.3, 0.8], [0.9, 0.2, 0.2], [0.4, 0.4, 0.5]]),  # high
        np.array([2, 0, 2]),  # classes
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2], [3, 4], [5, 6]])
    predict, low, high, classes = orchestrator.predict(x)

    assert len(predict) == 3
    assert len(low) == 3
    assert len(high) == 3
    assert np.allclose(classes, [2, 0, 2])


def testpredict_impl_regression(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False

    mock_learner = MagicMock()
    mock_learner.predict_uncertainty.return_value = (
        np.array([0.5]),  # predict
        np.array([0.4]),  # low
        np.array([0.6]),  # high
        None,
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x)

    assert np.allclose(predict, [0.5])
    assert np.allclose(low, [0.4])
    assert np.allclose(high, [0.6])
    assert classes is None


def testpredict_impl_regression_probabilistic(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False

    mock_learner = MagicMock()
    mock_learner.predict_probability.return_value = (
        np.array([0.8]),  # predict (prob)
        np.array([0.7]),  # low
        np.array([0.9]),  # high
        None,
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x, threshold=0.5)

    assert np.allclose(predict, [0.8])
    assert classes is None


def testpredict_impl_fast_binary(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = False
    mock_explainer.is_fast.return_value = True
    mock_explainer.num_features = 0

    mock_learner = MagicMock()
    mock_learner.__getitem__.return_value.predict_proba.return_value = (
        np.array([[0.1, 0.9]]),  # predict
        np.array([[0.0, 0.8]]),  # low
        np.array([[0.2, 1.0]]),  # high
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x)

    assert np.allclose(predict, [0.9])
    mock_learner.__getitem__.assert_called_with(0)


def testpredict_impl_fast_multiclass(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = True
    mock_explainer.is_fast.return_value = True
    mock_explainer.num_features = 0

    mock_learner = MagicMock()
    mock_learner.__getitem__.return_value.predict_proba.return_value = (
        np.array([[0.1, 0.2, 0.7]]),  # predict
        np.array([[0.0, 0.1, 0.6]]),  # low
        np.array([[0.2, 0.3, 0.8]]),  # high
        np.array([2]),  # classes
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x)

    assert len(predict) == 1
    assert np.allclose(classes, [2])
    mock_learner.__getitem__.assert_called_with(0)


def testpredict_impl_regression_invalid_percentiles(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False

    from calibrated_explanations.utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="low percentile must be smaller"):
        orchestrator.predict(np.array([[1]]), low_high_percentiles=(95, 5))

    with pytest.raises(ValidationError, match="percentiles must be between 0 and 100"):
        orchestrator.predict(np.array([[1]]), low_high_percentiles=(-10, 110))


def testpredict_impl_regression_crepes_error(orchestrator, mock_explainer, enable_fallbacks):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = True

    mock_learner = MagicMock()
    mock_learner.predict_uncertainty.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.warns(UserWarning, match="crepes produced an unexpected result"):
        predict, low, high, classes = orchestrator.predict(x)

    assert np.allclose(predict, [0])
    assert np.allclose(low, [0])
    assert np.allclose(high, [0])


def testensure_interval_runtime_state(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_hints = None
    mock_explainer.plugin_manager.interval_plugin_fallbacks = None
    mock_explainer.plugin_manager.interval_plugin_identifiers = None
    mock_explainer.plugin_manager.telemetry_interval_sources = None
    mock_explainer.plugin_manager.interval_preferred_identifier = None
    mock_explainer.plugin_manager.interval_context_metadata = None

    orchestrator.ensure_interval_runtime_state()

    assert mock_explainer.plugin_manager.interval_plugin_hints == {}
    assert mock_explainer.plugin_manager.interval_plugin_fallbacks == {}
    assert mock_explainer.plugin_manager.interval_plugin_identifiers == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.telemetry_interval_sources == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.interval_preferred_identifier == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.interval_context_metadata == {"default": {}, "fast": {}}


def test_check_interval_runtime_metadata_valid(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.bins = None

    metadata = {
        "name": "test_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
        "fast_compatible": True,
    }

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=False)
    assert error is None


def test_check_interval_runtime_metadata_invalid_mode(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"

    metadata = {
        "name": "test_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
    }

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=False)
    assert "does not support mode 'regression'" in error


def test_check_interval_runtime_metadata_invalid_capability(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"

    metadata = {
        "name": "test_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("other:capability",),
    }

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=False)
    assert "missing capability 'interval:classification'" in error


def test_check_interval_runtime_metadata_fast_incompatible(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"

    metadata = {
        "name": "test_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
        "fast_compatible": False,
    }

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=True)
    assert "not marked fast_compatible" in error


def test_resolve_interval_plugin_override(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = "test_plugin"
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = "test_plugin"
    # Ensure the override is in the fallbacks so it gets tried
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["test_plugin"]}

    with patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor"
    ) as mock_find:
        mock_descriptor = MagicMock()
        mock_descriptor.metadata = {
            "name": "test_plugin",
            "schema_version": 1,
            "modes": ("classification",),
            "capabilities": ("interval:classification",),
        }
        mock_descriptor.trusted = True
        mock_descriptor.plugin = MagicMock()
        mock_find.return_value = mock_descriptor

        mock_explainer.instantiate_plugin.return_value = "instantiated_plugin"

        plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)

        assert plugin == "instantiated_plugin"
        assert identifier == "test_plugin"


def test_resolve_interval_plugin_fallback(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = None
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    # If preferred identifier is set, failure raises ConfigurationError.
    # To test fallback, we must not have a strict preference that matches the failing plugin.
    mock_explainer.plugin_manager.interval_preferred_identifier = {}
    # Ensure both are in fallbacks so they are tried in order
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {
        "default": ["default_plugin", "fallback_plugin"]
    }

    with patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor"
    ) as mock_find:
        # First call (default_plugin) returns None (not found)
        # Second call (fallback_plugin) returns descriptor

        def side_effect(identifier):
            if identifier == "default_plugin":
                return None
            if identifier == "fallback_plugin":
                mock_descriptor = MagicMock()
                mock_descriptor.metadata = {
                    "name": "fallback_plugin",
                    "schema_version": 1,
                    "modes": ("classification",),
                    "capabilities": ("interval:classification",),
                }
                mock_descriptor.trusted = True
                mock_descriptor.plugin = MagicMock()
                return mock_descriptor
            return None

        mock_find.side_effect = side_effect

        mock_explainer.instantiate_plugin.return_value = "instantiated_fallback"

        # Mock find_interval_plugin and find_interval_plugin_trusted to return None
        with (
            patch(
                "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin",
                return_value=None,
            ),
            patch(
                "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted",
                return_value=None,
            ),
        ):
            plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)

            assert plugin == "instantiated_fallback"
            assert identifier == "fallback_plugin"


def testpredict_impl_fast_regression(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = True
    mock_explainer.num_features = 0

    mock_learner = MagicMock()
    mock_learner.__getitem__.return_value.predict_uncertainty.return_value = (
        np.array([0.5]),  # predict
        np.array([0.4]),  # low
        np.array([0.6]),  # high
        None,
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict_impl(x)

    assert np.allclose(predict, [0.5])
    assert np.allclose(low, [0.4])
    assert np.allclose(high, [0.6])
    assert classes is None
    mock_learner.__getitem__.assert_called_with(0)


def testpredict_impl_fast_regression_probabilistic(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = True
    mock_explainer.num_features = 0

    mock_learner = MagicMock()
    mock_learner.__getitem__.return_value.predict_probability.return_value = (
        np.array([0.8]),  # predict (prob)
        np.array([0.7]),  # low
        np.array([0.9]),  # high
        None,
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict_impl(x, threshold=0.5)

    assert np.allclose(predict, [0.8])
    assert classes is None
    mock_learner.__getitem__.assert_called_with(0)


def testpredict_impl_fast_specific_feature(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = False
    mock_explainer.is_fast.return_value = True
    mock_explainer.num_features = 5  # Default if feature is None

    mock_learner = MagicMock()
    mock_learner.__getitem__.return_value.predict_proba.return_value = (
        np.array([[0.1, 0.9]]),
        np.array([[0.0, 0.8]]),
        np.array([[0.2, 1.0]]),
    )
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    # Pass explicit feature index
    predict, low, high, classes = orchestrator.predict_impl(x, feature=2)

    mock_learner.__getitem__.assert_called_with(2)


def test_resolve_interval_plugin_object_override(orchestrator, mock_explainer):
    # Test when coerce_plugin_override returns a plugin instance directly
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "direct_plugin"}
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = mock_plugin
    mock_explainer.plugin_manager.interval_plugin_override = mock_plugin

    plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)

    assert plugin == mock_plugin
    assert identifier == "direct_plugin"


def test_resolve_interval_plugin_with_hints(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = None
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.interval_preferred_identifier = {}
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["fallback"]}

    hints = ("hinted_plugin",)

    with patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor"
    ) as mock_find:
        # hinted_plugin found
        mock_descriptor = MagicMock()
        mock_descriptor.metadata = {
            "name": "hinted_plugin",
            "schema_version": 1,
            "modes": ("classification",),
            "capabilities": ("interval:classification",),
        }
        mock_descriptor.trusted = True
        mock_descriptor.plugin = MagicMock()

        mock_find.side_effect = lambda id: mock_descriptor if id == "hinted_plugin" else None
        mock_explainer.instantiate_plugin.return_value = "instantiated_hint"

        plugin, identifier = orchestrator.resolve_interval_plugin(fast=False, hints=hints)

        assert identifier == "hinted_plugin"


def test_resolve_interval_plugin_denied(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = None
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.interval_preferred_identifier = {"default": "denied_plugin"}
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["denied_plugin"]}

    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
            return_value=True,
        ),
        pytest.raises(core_exceptions.ConfigurationError, match="denied via CE_DENY_PLUGIN"),
    ):
        orchestrator.resolve_interval_plugin(fast=False)


def test_resolve_interval_plugin_not_registered_preferred(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = None
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.interval_preferred_identifier = {"default": "missing_plugin"}
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["missing_plugin"]}

    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
            return_value=None,
        ),
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin",
            return_value=None,
        ),
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted",
            return_value=None,
        ),
        pytest.raises(core_exceptions.ConfigurationError, match="not registered"),
    ):
        orchestrator.resolve_interval_plugin(fast=False)


def test_resolve_interval_plugin_metadata_error_preferred(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.fast_interval_plugin_override = None
    mock_explainer.plugin_manager.interval_plugin_override = None
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = None
    mock_explainer.plugin_manager.interval_preferred_identifier = {"default": "bad_metadata_plugin"}
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["bad_metadata_plugin"]}

    mock_descriptor = MagicMock()
    # Missing modes
    mock_descriptor.metadata = {"name": "bad_metadata_plugin", "schema_version": 1}
    mock_descriptor.trusted = True
    mock_descriptor.plugin = MagicMock()

    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
            return_value=mock_descriptor,
        ),
        pytest.raises(core_exceptions.ConfigurationError, match="missing modes declaration"),
    ):
        orchestrator.resolve_interval_plugin(fast=False)


def test_obtain_interval_calibrator_success(orchestrator, mock_explainer):
    # Test obtain_interval_calibrator calling resolve_interval_plugin and creating calibrator
    mock_explainer.plugin_manager.interval_plugin_hints = {}

    with (
        patch.object(orchestrator, "resolve_interval_plugin") as mock_resolve,
        patch.object(orchestrator, "build_interval_context") as mock_build,
        patch.object(orchestrator, "capture_interval_calibrators") as mock_capture,
    ):
        mock_plugin = MagicMock()
        mock_plugin.plugin_meta = {"trusted": False}  # Mark as untrusted to skip validation
        mock_plugin.create.return_value = "calibrator_instance"
        mock_resolve.return_value = (mock_plugin, "test_plugin")

        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_build.return_value = mock_context

        calibrator, identifier = orchestrator.obtain_interval_calibrator(fast=False, metadata={})

        assert calibrator == "calibrator_instance"
        assert identifier == "test_plugin"
        mock_plugin.create.assert_called_once()
        mock_capture.assert_called_once()


def test_obtain_interval_calibrator_creation_failure(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_hints = {}

    with (
        patch.object(orchestrator, "resolve_interval_plugin") as mock_resolve,
        patch.object(orchestrator, "build_interval_context") as mock_build,
    ):
        mock_plugin = MagicMock()
        mock_plugin.create.side_effect = ValueError("Creation failed")
        mock_resolve.return_value = (mock_plugin, "test_plugin")

        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_build.return_value = mock_context

        with pytest.raises(
            core_exceptions.ConfigurationError, match="Interval plugin execution failed"
        ):
            orchestrator.obtain_interval_calibrator(fast=False, metadata={})


def test_build_interval_context(orchestrator, mock_explainer):
    mock_explainer.x_cal = "x_cal"
    mock_explainer.y_cal = "y_cal"
    mock_explainer.bins = "bins"
    mock_explainer.difficulty_estimator = "diff_est"
    mock_explainer.mode = "classification"
    mock_explainer.categorical_features = [1, 2]
    mock_explainer.num_features = 10
    mock_explainer.learner = "learner"
    mock_explainer.plugin_manager.interval_context_metadata = {"default": {"stored": "meta"}}

    # Mock private attributes for noise config
    mock_explainer.noise_type = "noise"
    mock_explainer.scale_factor = 0.1
    mock_explainer.severity = 0.5
    mock_explainer.seed = 42
    mock_explainer.rng = "rng"

    context = orchestrator.build_interval_context(fast=False, metadata={"new": "meta"})

    assert context.learner == "learner"
    assert context.calibration_splits == (("x_cal", "y_cal"),)
    assert context.bins == {"calibration": "bins"}
    assert context.difficulty == {"estimator": "diff_est"}
    assert context.fast_flags == {"fast": False}

    meta = context.metadata
    assert meta["stored"] == "meta"
    assert meta["new"] == "meta"
    assert meta["task"] == "classification"
    assert meta["categorical_features"] == (1, 2)
    assert meta["noise_config"]["noise_type"] == "noise"


def test_build_interval_context_fast(orchestrator, mock_explainer):
    mock_explainer.x_cal = "x_cal"
    mock_explainer.y_cal = "y_cal"
    mock_explainer.bins = None
    mock_explainer.difficulty_estimator = None
    mock_explainer.mode = "classification"
    mock_explainer.categorical_features = []
    mock_explainer.num_features = 10
    mock_explainer.learner = "learner"
    mock_explainer.plugin_manager.interval_context_metadata = {"fast": {}}
    mock_explainer.interval_learner = ["learner1", "learner2"]

    context = orchestrator.build_interval_context(fast=True, metadata={})

    assert context.fast_flags == {"fast": True}
    assert context.metadata["existing_fast_calibrators"] == ("learner1", "learner2")


def testpredict_impl_regression_crepes_error_reraise(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = False

    mock_learner = MagicMock()
    mock_learner.predict_uncertainty.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.raises(ValueError, match="crepes error"):
        orchestrator.predict_impl(x)


def testpredict_impl_regression_probabilistic_crepes_error_suppress(
    orchestrator, mock_explainer, enable_fallbacks
):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = True

    mock_learner = MagicMock()
    mock_learner.predict_probability.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.warns(UserWarning, match="crepes produced an unexpected result"):
        predict, low, high, classes = orchestrator.predict_impl(x, threshold=0.5)

    assert np.allclose(predict, [0])


def testpredict_impl_regression_probabilistic_crepes_error_reraise(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = False

    mock_learner = MagicMock()
    mock_learner.predict_probability.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    # It raises DataShapeError wrapping the original error
    with pytest.raises(DataShapeError, match="Error while computing prediction intervals"):
        orchestrator.predict_impl(x, threshold=0.5)


def test_capture_interval_calibrators(orchestrator):
    context = MagicMock()
    context.plugin_state = {}

    # Fast mode
    orchestrator.capture_interval_calibrators(context=context, calibrator=["c1", "c2"], fast=True)
    assert context.plugin_state["fast_calibrators"] == ("c1", "c2")

    # Default mode
    context.plugin_state = {}
    orchestrator.capture_interval_calibrators(context=context, calibrator="c1", fast=False)
    assert context.plugin_state["calibrator"] == "c1"


def testgather_interval_hints(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_hints = {
        "fast": ("fast_hint",),
        "factual": ("fact_hint", "shared_hint"),
        "alternative": ("alt_hint", "shared_hint"),
    }

    hints_fast = orchestrator.gather_interval_hints(fast=True)
    assert hints_fast == ("fast_hint",)

    hints_default = orchestrator.gather_interval_hints(fast=False)
    assert hints_default == ("fact_hint", "shared_hint", "alt_hint")


def test_predict_no_cache(orchestrator, mock_explainer):
    mock_explainer.perf_cache = None
    x = np.array([[1, 2]])

    with patch.object(orchestrator, "predict_impl") as mock_impl:
        mock_impl.return_value = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)

        result = orchestrator.predict(x)

        mock_impl.assert_called_once()
        assert result == mock_impl.return_value


def test_validate_prediction_result_none(orchestrator):
    result = (None, None, None, None)
    # Should not raise or warn
    orchestrator.validate_prediction_result(result)


def test_validate_prediction_result_empty(orchestrator):
    result = (np.array([]), np.array([]), np.array([]), None)
    orchestrator.validate_prediction_result(result)


def test_check_interval_runtime_metadata_requires_bins(orchestrator, mock_explainer):
    mock_explainer.mode = "classification"
    mock_explainer.bins = None

    metadata = {
        "name": "test_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
        "requires_bins": True,
    }

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=False)
    assert "requires bins" in error


def testpredict_impl_regression_probabilistic_invalid_threshold(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False

    x = np.array([[1, 2]])
    # Threshold length mismatch
    threshold = [0.5, 0.6]  # Length 2, x has 1 sample

    with pytest.raises(AssertionError):
        orchestrator.predict_impl(x, threshold=threshold)


def testpredict_impl_unknown_mode(orchestrator, mock_explainer):
    mock_explainer.mode = "unknown"
    mock_explainer.is_fast.return_value = False

    x = np.array([[1, 2]])
    result = orchestrator.predict_impl(x)
    assert result == (None, None, None, None)


def test_predict_not_fitted(orchestrator, mock_explainer):
    mock_explainer.initialized = False
    with pytest.raises(NotFittedError):
        orchestrator.predict_impl(np.array([[1]]))


def test_predict_regression_invalid_percentiles(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.initialized = True

    # Low > High
    with pytest.raises(ValidationError):
        orchestrator.predict_impl(np.array([[1]]), low_high_percentiles=(95, 5))

    # Out of bounds
    with pytest.raises(ValidationError):
        orchestrator.predict_impl(np.array([[1]]), low_high_percentiles=(-10, 110))


def testensure_interval_runtime_state_missing_attributes(orchestrator, mock_explainer):
    # Set attributes to None to force recreation
    mock_explainer.plugin_manager.interval_plugin_hints = None
    mock_explainer.plugin_manager.interval_plugin_fallbacks = None
    mock_explainer.plugin_manager.interval_plugin_identifiers = None
    mock_explainer.plugin_manager.telemetry_interval_sources = None
    mock_explainer.plugin_manager.interval_preferred_identifier = None
    mock_explainer.plugin_manager.interval_context_metadata = None

    orchestrator.ensure_interval_runtime_state()

    assert mock_explainer.plugin_manager.interval_plugin_hints == {}
    assert mock_explainer.plugin_manager.interval_plugin_fallbacks == {}
    assert mock_explainer.plugin_manager.interval_plugin_identifiers == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.telemetry_interval_sources == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.interval_preferred_identifier == {
        "default": None,
        "fast": None,
    }
    assert mock_explainer.plugin_manager.interval_context_metadata == {"default": {}, "fast": {}}


def testgather_interval_hints_duplicate(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_hints = {
        "fast": ("fast_hint",),
        "factual": ("factual_hint",),
        "alternative": ("alternative_hint",),
    }

    # Fast mode
    hints_fast = orchestrator.gather_interval_hints(fast=True)
    assert hints_fast == ("fast_hint",)

    # Default mode
    hints_default = orchestrator.gather_interval_hints(fast=False)
    assert "factual_hint" in hints_default
    assert "alternative_hint" in hints_default


def test_resolve_interval_plugin_override_object(orchestrator, mock_explainer):
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "custom_plugin"}
    mock_explainer.plugin_manager.interval_plugin_override = mock_plugin
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = mock_plugin

    plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)
    assert plugin == mock_plugin
    assert identifier == "custom_plugin"


def test_resolve_interval_plugin_denied_preferred(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_override = "denied_plugin"
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = "denied_plugin"
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["denied_plugin"]}

    # Mock is_identifier_denied to return True for "denied_plugin"
    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
            return_value=True,
        ),
        pytest.raises(ConfigurationError, match="denied via CE_DENY_PLUGIN"),
    ):
        orchestrator.resolve_interval_plugin(fast=False)


def test_resolve_interval_plugin_metadata_error_preferred_duplicate(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_override = "bad_metadata_plugin"
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = "bad_metadata_plugin"
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["bad_metadata_plugin"]}

    # Mock find_interval_descriptor to return a descriptor with bad metadata
    mock_descriptor = MagicMock()
    mock_descriptor.metadata = {"schema_version": 999}  # Invalid version
    mock_descriptor.trusted = True
    mock_descriptor.plugin = MagicMock()

    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
            return_value=mock_descriptor,
        ),
        pytest.raises(ConfigurationError, match="unsupported interval schema_version"),
    ):
        orchestrator.resolve_interval_plugin(fast=False)


def test_obtain_interval_calibrator_creation_failure_duplicate(orchestrator, mock_explainer):
    mock_explainer.plugin_manager.interval_plugin_override = "failing_plugin"
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = "failing_plugin"
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {"default": ["failing_plugin"]}

    mock_plugin = MagicMock()
    mock_plugin.create.side_effect = RuntimeError("Creation failed")

    # Ensure instantiate_plugin returns our mock_plugin
    mock_explainer.instantiate_plugin.return_value = mock_plugin

    mock_descriptor = MagicMock()
    mock_descriptor.metadata = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
    }
    mock_descriptor.trusted = True
    mock_descriptor.plugin = mock_plugin

    mock_explainer.mode = "regression"

    with (
        patch(
            "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
            return_value=False,
        ),
        patch(
            "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
            return_value=mock_descriptor,
        ),
        patch("calibrated_explanations.core.prediction.orchestrator.ensure_builtin_plugins"),
        pytest.raises(ConfigurationError, match="Interval plugin execution failed"),
    ):
        orchestrator.obtain_interval_calibrator(fast=False, metadata={})


def test_resolve_interval_plugin_override_object_fast(orchestrator, mock_explainer):
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "custom_plugin_fast"}
    mock_explainer.plugin_manager.fast_interval_plugin_override = mock_plugin
    mock_explainer.plugin_manager.coerce_plugin_override.return_value = mock_plugin

    plugin, identifier = orchestrator.resolve_interval_plugin(fast=True)
    assert plugin == mock_plugin
    assert identifier == "custom_plugin_fast"


def test_check_interval_runtime_metadata_fast_incompatible_duplicate(orchestrator, mock_explainer):
    metadata = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
        "fast_compatible": False,
    }
    mock_explainer.mode = "regression"

    error = orchestrator.check_interval_runtime_metadata(metadata, identifier="test", fast=True)
    assert "not marked fast_compatible" in error


def test_validate_interval_calibrator_fast_mode_deep_validation_failure(
    orchestrator, mock_explainer
):
    """Test that validate_interval_calibrator performs deep validation in FAST mode."""
    from calibrated_explanations.calibration.interval_wrappers import FastIntervalCalibrator

    # Create a simple invalid calibrator class
    class InvalidCalibrator:
        pass

    # Create a FastIntervalCalibrator with one invalid item
    fast_calibrator = FastIntervalCalibrator([InvalidCalibrator()])

    context = IntervalCalibratorContext(
        learner=MagicMock(),
        calibration_splits=(MagicMock(),),
        bins={},
        residuals={},
        difficulty={},
        metadata={"task": "classification"},
        fast_flags={"fast": True},
    )

    with pytest.raises(
        ConfigurationError, match="Interval calibrator at index 0.*is non-compliant"
    ):
        orchestrator.validate_interval_calibrator(
            calibrator=fast_calibrator,
            context=context,
            identifier="test_plugin",
            fast=True,
        )


def test_fast_interval_calibrator_protocol_compliance():
    """Test that FastIntervalCalibrator implements the full protocol."""
    from calibrated_explanations.calibration.interval_wrappers import FastIntervalCalibrator

    # Create a mock calibrator that implements the protocol
    mock_calibrator = MagicMock()
    mock_calibrator.predict_proba.return_value = None
    mock_calibrator.predict_probability.return_value = None
    mock_calibrator.predict_uncertainty.return_value = None
    mock_calibrator.is_multiclass.return_value = False
    mock_calibrator.is_mondrian.return_value = False
    mock_calibrator.pre_fit_for_probabilistic.return_value = None
    mock_calibrator.compute_proba_cal.return_value = None
    mock_calibrator.insert_calibration.return_value = None

    fast_calibrator = FastIntervalCalibrator([mock_calibrator])

    # Check isinstance for both protocols
    assert isinstance(fast_calibrator, ClassificationIntervalCalibrator)
    assert isinstance(fast_calibrator, RegressionIntervalCalibrator)


def test_build_interval_context_metadata_immutable(orchestrator, mock_explainer):
    """Test that build_interval_context exposes immutable metadata and plugin_state."""
    mock_explainer.mode = "classification"
    mock_explainer.learner = MagicMock()
    mock_explainer.x_cal = MagicMock()
    mock_explainer.y_cal = MagicMock()
    mock_explainer.bins = {}
    mock_explainer.difficulty_estimator = MagicMock()
    mock_explainer.plugin_manager.interval_context_metadata = {"default": {}}
    mock_explainer.categorical_features = ()
    mock_explainer.num_features = 5
    mock_explainer.seed = None
    mock_explainer.rng = None
    mock_explainer.noise_type = None
    mock_explainer.scale_factor = None
    mock_explainer.severity = None
    context = orchestrator.build_interval_context(fast=False, metadata={})

    assert isinstance(context.metadata, MappingProxyType)

    # Metadata should remain read-only; plugins should use plugin_state.
    with pytest.raises(TypeError):
        context.metadata["new_key"] = "value"  # type: ignore
    context.plugin_state["new_key"] = "value"
    assert context.plugin_state["new_key"] == "value"
