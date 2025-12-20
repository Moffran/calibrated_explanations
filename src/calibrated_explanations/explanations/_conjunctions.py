# src/calibrated_explanations/explanations/_conjunctions.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

class ConjunctionState:
    """Helper class to manage the state of conjunctive rules."""

    def __init__(self, initial_rules: Optional[Dict[str, Any]] = None):
        """Initialize the state with existing rules."""
        if initial_rules is None:
            self.state = {
                "base_predict": [],
                "base_predict_low": [],
                "base_predict_high": [],
                "predict": [],
                "predict_low": [],
                "predict_high": [],
                "weight": [],
                "weight_low": [],
                "weight_high": [],
                "value": [],
                "rule": [],
                "feature": [],
                "sampled_values": [],
                "feature_value": [],
                "is_conjunctive": [],
                "classes": [],
            }
        else:
            self.state = self._clone_payload(initial_rules)
            # Ensure is_conjunctive is present
            if "is_conjunctive" not in self.state:
                self.state["is_conjunctive"] = [False] * len(self.state["rule"])
        self._combination_keys = set()
        if self.state["feature"]:
            for feature in self.state["feature"]:
                key = self._normalize_feature_entry(feature)
                self._combination_keys.add(key)

    def _clone_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    def set_base_prediction(self, predict, low, high):
        """Set the base prediction values."""
        self.state["base_predict"] = [predict]
        self.state["base_predict_low"] = [low]
        self.state["base_predict_high"] = [high]

    def add_rule(self,
                 predict: float,
                 low: float,
                 high: float,
                 base_predict: float,
                 value: str,
                 feature: List[int],
                 sampled_values: List[Any],
                 feature_value: Optional[List[Any]],
                 rule_text: str,
                 is_conjunctive: bool = True,
                 weight: Optional[float] = None,
                 weight_low: Optional[float] = None,
                 weight_high: Optional[float] = None):
        """Add a new conjunctive rule to the state."""
        
        self.state["predict"].append(predict)
        self.state["predict_low"].append(low)
        self.state["predict_high"].append(high)
        
        if weight is not None:
            self.state["weight"].append(weight)
        else:
            self.state["weight"].append(predict - base_predict)
            
        if weight_low is not None:
            self.state["weight_low"].append(weight_low)
        else:
            self.state["weight_low"].append(low - base_predict if low != -np.inf else -np.inf)
            
        if weight_high is not None:
            self.state["weight_high"].append(weight_high)
        else:
            self.state["weight_high"].append(high - base_predict if high != np.inf else np.inf)
        
        self.state["value"].append(value)
        self.state["feature"].append(feature)
        self.state["sampled_values"].append(sampled_values)
        self.state["feature_value"].append(feature_value)
        self.state["rule"].append(rule_text)
        self.state["is_conjunctive"].append(is_conjunctive)
        self._combination_keys.add(self._normalize_feature_entry(feature))

    def get_state(self) -> Dict[str, Any]:
        """Return the current state."""
        return self.state
    
    def get_weights(self) -> np.ndarray:
        """Return weights as numpy array."""
        return np.asarray(self.state["weight"], dtype=float)

    def get_widths(self) -> np.ndarray:
        """Return widths (high - low) as numpy array."""
        return np.asarray(self.state["weight_high"], dtype=float) - np.asarray(
                self.state["weight_low"], dtype=float
            )
    
    def get_feature(self, index: int) -> Any:
        return self.state["feature"][index]
    
    def get_sampled_values(self, index: int) -> Any:
        return self.state["sampled_values"][index]
    
    def get_feature_value(self, index: int) -> Any:
        return self.state["feature_value"][index]
    
    def get_value(self, index: int) -> str:
        return self.state["value"][index]
    
    def get_rule(self, index: int) -> str:
        return self.state["rule"][index]

    def is_conjunctive(self, index: int) -> bool:
        return self.state["is_conjunctive"][index]

    @staticmethod
    def _normalize_feature_entry(feature: Any) -> Tuple[int, ...]:
        if isinstance(feature, (list, tuple, np.ndarray)):
            return tuple(sorted(int(v) for v in np.asarray(feature).ravel()))
        return (int(feature),)

    def has_combination_key(self, key: Union[Tuple[int, ...], Any]) -> bool:
        if not isinstance(key, tuple):
            key = self._normalize_feature_entry(key)
        return key in self._combination_keys

    def register_combination_key(self, key: Union[Tuple[int, ...], Any]) -> None:
        if not isinstance(key, tuple):
            key = self._normalize_feature_entry(key)
        self._combination_keys.add(key)
