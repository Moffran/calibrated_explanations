import pytest
import numpy as np
from calibrated_explanations.explanations.explanation import CalibratedExplanation
from unittest.mock import MagicMock

class ConcreteExplanation(CalibratedExplanation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __repr__(self): return ""
    def _check_preconditions(self): pass
    def get_rules(self): return []
    def _is_lesser(self, i): return False
    def add_conjunctions(self, *args): pass
    def build_rules_payload(self): return {}
    def plot(self, *args, **kwargs): pass

def test_explanation_init_variants():
    mock_explainer = MagicMock()
    mock_explainer.y_cal = np.array([0, 1])
    mock_ce = MagicMock()
    mock_ce.calibrated_explainer = mock_explainer
    
    # Test with __full_probabilities__
    prediction = {
        "predict": [0.5], 
        "low": [0.4], 
        "high": [0.6],
        "__full_probabilities__": np.array([[0.5, 0.5]])
    }
    exp = ConcreteExplanation(
        calibrated_explanations=mock_ce,
        index=0,
        x=np.array([[1]]),
        binned={},
        feature_weights={},
        feature_predict={},
        prediction=prediction,
        y_threshold=None
    )
    assert "__full_probabilities__" in exp.prediction
    assert np.array_equal(exp.prediction["__full_probabilities__"], prediction["__full_probabilities__"])

    # Test with y_threshold as list
    exp2 = ConcreteExplanation(
        calibrated_explanations=mock_ce,
        index=0,
        x=np.array([[1]]),
        binned={},
        feature_weights={},
        feature_predict={},
        prediction=prediction,
        y_threshold=[0.5, 0.6]
    )
    assert exp2.y_threshold == 0.5

def test_explanation_y_minmax_categorical():
    from pandas import Categorical
    mock_explainer = MagicMock()
    mock_explainer.y_cal = Categorical([0, 1])
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    mock_ce.calibrated_explainer = mock_explainer
    
    prediction = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    exp = ConcreteExplanation(
        calibrated_explanations=mock_ce,
        index=0,
        x=np.array([[1]]),
        binned={},
        feature_weights={},
        feature_predict={},
        prediction=prediction,
        y_threshold=None
    )
    assert exp.y_minmax == [0, 0]

def test_explanation_get_explainer_fallbacks():
    mock_explainer = MagicMock()
    
    # Test _get_explainer fallback
    mock_ce_1 = MagicMock(spec=[])
    mock_ce_1._get_explainer = MagicMock(return_value=mock_explainer)
    exp1 = ConcreteExplanation(mock_ce_1, 0, np.array([[1]]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    assert exp1.get_explainer() == mock_explainer
    
    # Test explainer attribute fallback
    mock_ce_2 = MagicMock(spec=[])
    mock_ce_2.explainer = mock_explainer
    exp2 = ConcreteExplanation(mock_ce_2, 0, np.array([[1]]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    assert exp2.get_explainer() == mock_explainer
    
    # Test calibrated_explainer attribute fallback
    mock_ce_3 = MagicMock(spec=[])
    mock_ce_3.calibrated_explainer = mock_explainer
    exp3 = ConcreteExplanation(mock_ce_3, 0, np.array([[1]]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    assert exp3.get_explainer() == mock_explainer

def test_explanation_ignored_features():
    mock_explainer = MagicMock()
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    mock_ce.features_to_ignore = [1, 2]
    
    exp = ConcreteExplanation(mock_ce, 0, np.array([[1]]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    ignored = exp._ignored_features_for_instance()
    assert 1 in ignored
    assert 2 in ignored
    
    # Test with None
    mock_ce.features_to_ignore = None
    ignored = exp._ignored_features_for_instance()
    assert len(ignored) == 0

def test_explanation_add_conjunction_warnings():
    mock_explainer = MagicMock()
    mock_explainer.feature_names = ["a", "b"]
    mock_explainer.categorical_features = [1]
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    
    exp = ConcreteExplanation(mock_ce, 0, np.array([[1, 2]]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    
    # Feature not found
    with pytest.warns(UserWarning, match="Feature c not found"):
        exp.add_new_rule_condition("c", 0.5)
        
    # Categorical feature
    with pytest.warns(UserWarning, match="Alternatives for all categorical features are already included"):
        exp.add_new_rule_condition(1, 0.5)


def test_explanation_validate_prediction_invariant_none():
    mock_explainer = MagicMock()
    mock_explainer.y_cal = np.array([0, 1])
    mock_ce = MagicMock()
    mock_ce.calibrated_explainer = mock_explainer
    
    # Missing 'predict'
    prediction = {"low": [0.4], "high": [0.6]}
    exp = ConcreteExplanation(
        calibrated_explanations=mock_ce,
        index=0,
        x=np.array([[1]]),
        binned={},
        feature_weights={},
        feature_predict={},
        prediction=prediction,
        y_threshold=None
    )
    # Should return early without warning


def test_explanation_define_conditions_no_discretizer():
    mock_explainer = MagicMock()
    mock_explainer.discretizer = None
    mock_explainer.num_features = 1
    mock_explainer.feature_names = ["f1"]
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    mock_ce.calibrated_explainer = mock_explainer
    mock_ce.features_to_ignore = []
    
    exp = ConcreteExplanation(mock_ce, 0, np.array([1.5]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    conds = exp.define_conditions()
    assert conds == ["f1 = 1.5"]


def test_explanation_define_conditions_categorical_index_error():
    mock_explainer = MagicMock()
    mock_explainer.num_features = 1
    mock_explainer.feature_names = ["f1"]
    mock_explainer.categorical_features = [0]
    mock_explainer.categorical_labels = {0: ["a"]} # only one label
    
    mock_discretizer = MagicMock()
    mock_discretizer.discretize.return_value = np.array([1]) # index 1 out of range for ["a"]
    mock_explainer.discretizer = mock_discretizer
    
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    mock_ce.calibrated_explainer = mock_explainer
    mock_ce.features_to_ignore = []
    
    exp = ConcreteExplanation(mock_ce, 0, np.array([1]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    conds = exp.define_conditions()
    assert conds == ["f1 = 1"]


def test_explanation_add_new_rule_condition_variants():
    mock_explainer = MagicMock()
    mock_explainer.feature_names = ["f1"]
    mock_explainer.categorical_features = []
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    
    exp = ConcreteExplanation(mock_ce, 0, np.array([1]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    
    # Feature not found
    with pytest.warns(UserWarning, match="Feature f2 not found"):
        exp.add_new_rule_condition("f2", 0.5)
    
    # Categorical feature
    mock_explainer.categorical_features = [0]
    with pytest.warns(UserWarning, match="Alternatives for all categorical features are already included"):
        exp.add_new_rule_condition(0, 0.5)


def test_predict_conjunction_tuple_empty():
    mock_explainer = MagicMock()
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    exp = ConcreteExplanation(mock_ce, 0, np.array([1]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    
    # Empty rule_value_set
    res = exp._predict_conjunction_tuple([], [], np.array([1]), None, None)
    assert res == (0.0, 0.0, 0.0)
    
    # combo_matrix.size == 0
    res2 = exp._predict_conjunction_tuple([[]], [0], np.array([1]), None, None)
    assert res2 == (0.0, 0.0, 0.0)


def test_predict_conjunction_tuple_with_bins():
    mock_explainer = MagicMock()
    mock_explainer._predict.return_value = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)
    mock_ce = MagicMock()
    mock_ce.get_explainer.return_value = mock_explainer
    mock_ce.low_high_percentiles = (5, 95)
    exp = ConcreteExplanation(mock_ce, 0, np.array([1]), {}, {}, {}, {"predict": [0.5], "low": [0.4], "high": [0.6]})
    
    # With bins as scalar
    res = exp._predict_conjunction_tuple([[1]], [0], np.array([1]), None, None, bins=0)
    assert res == (0.5, 0.4, 0.6)
    
    # With bins as array
    res2 = exp._predict_conjunction_tuple([[1]], [0], np.array([1]), None, None, bins=np.array([0]))
    assert res2 == (0.5, 0.4, 0.6)

