import pytest
import numpy as np
from unittest.mock import MagicMock
from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.core.exceptions import ValidationError

class ConcreteExplanation(CalibratedExplanation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __repr__(self): return ""
    def _check_preconditions(self): pass
    def _get_rules(self): return []
    def _is_lesser(self, i): return False
    def add_conjunctions(self, *args): pass
    def build_rules_payload(self): return {}
    def plot(self, *args, **kwargs): pass

class TestCalibratedExplanationInvariants:
    def setup_method(self):
        self.mock_explainer = MagicMock()
        self.mock_explainer.y_cal = np.array([0, 1])
        self.mock_ce = MagicMock()
        self.mock_ce.calibrated_explainer = self.mock_explainer

    def create_explanation(self, prediction):
        return ConcreteExplanation(
            calibrated_explanations=self.mock_ce,
            index=0,
            x=np.array([[1]]),
            binned={},
            feature_weights={},
            feature_predict={},
            prediction=prediction,
            y_threshold=None
        )

    def test_init_valid_invariant(self):
        prediction = {
            "predict": [0.5],
            "low": [0.4],
            "high": [0.6]
        }
        # Should not raise
        self.create_explanation(prediction)

    def test_init_invalid_low_gt_high(self):
        prediction = {
            "predict": [0.5],
            "low": [0.7],
            "high": [0.6]
        }
        with pytest.raises(ValidationError, match="low > high"):
            self.create_explanation(prediction)

    def test_init_invalid_predict_lt_low(self):
        prediction = {
            "predict": [0.3],
            "low": [0.4],
            "high": [0.6]
        }
        with pytest.raises(ValidationError, match="predict not in"):
            self.create_explanation(prediction)

    def test_init_invalid_predict_gt_high(self):
        prediction = {
            "predict": [0.7],
            "low": [0.4],
            "high": [0.6]
        }
        with pytest.raises(ValidationError, match="predict not in"):
            self.create_explanation(prediction)
