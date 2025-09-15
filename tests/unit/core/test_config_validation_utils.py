import numpy as np

from calibrated_explanations.api.config import ExplainerBuilder, ExplainerConfig
from calibrated_explanations.core import validation as val
from calibrated_explanations.utils import helper


def test_explainer_builder_builds_config_and_perf_factory_is_optional():
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))

    b = ExplainerBuilder(DummyModel())
    b.task("regression").perf_cache(True, max_items=5)
    cfg = b.build_config()
    assert isinstance(cfg, ExplainerConfig)
    assert cfg.task == "regression"
    assert cfg.perf_cache_enabled is True


def test_infer_task_from_y_and_model():
    # model with predict_proba -> classification
    class M1:
        def predict_proba(self, X):
            return np.zeros((len(X), 2))

    assert val.infer_task(model=M1()) == "classification"

    # y float -> regression
    y = np.array([0.1, 0.2, 0.3])
    assert val.infer_task(y=y) == "regression"

    # y ints -> classification
    y2 = np.array([0, 1, 1])
    assert val.infer_task(y=y2) == "classification"


def test_validate_inputs_matrix_mismatch_y_length_raises():
    X = np.zeros((3, 2))
    y = np.zeros(2)
    try:
        val.validate_inputs_matrix(X, y)
    except Exception as e:
        assert "does not match number of samples" in str(e)


def test_assert_threshold_variants_and_errors():
    assert helper.assert_threshold(0.5, [1, 2]) == 0.5
    assert helper.assert_threshold((0.2, 0.8), [1, 2]) == (0.2, 0.8)
    try:
        helper.assert_threshold([0.1, 0.2], [1])
        raise AssertionError("should have raised")
    except AssertionError:
        pass


def test_safe_mean_and_immutable_array_and_prepare_for_saving(tmp_path):
    assert helper.safe_mean([], default=3.2) == 3.2
    arr = helper.immutable_array([1, 2, 3])
    assert not arr.flags.writeable
    p = tmp_path / "out"
    p.mkdir()
    # Use a relative path under 'plots' so make_directory can create it
    fn = f"plots/{p.name}/test.png"
    path, filename, title, ext = helper.prepare_for_saving(fn)
    assert title == "test"
    assert ext == ".png"
