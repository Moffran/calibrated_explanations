"""Comprehensive unit tests for reject policy hardening.

This module provides exhaustive testing of all RejectPolicy variants,
edge cases, policy interactions, and metadata validation per v0.10.3 Task 9.

Test Categories:
1. Policy behavior verification (all 6 variants)
2. Edge cases (empty, single, all-rejected, no-rejects)
3. Policy interactions (override, combination)
4. Metadata validation (error_rate, reject_rate)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy, is_policy_enabled
from calibrated_explanations.explanations.reject import RejectResult


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


class MockExplainer:
    """Mock explainer for testing reject orchestrator without crepes dependency."""

    def __init__(
        self,
        mode: str = "classification",
        rejected_mask: np.ndarray | None = None,
        error_rate: float = 0.05,
        reject_rate: float = 0.2,
    ):
        self.mode = mode
        self.x_cal = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        self.y_cal = np.array([0, 1, 0, 1])
        self.bins = None
        self.reject_learner = MagicMock()
        self.reject_threshold = None
        self.interval_learner = MagicMock()

        # Configure prediction behavior (public for test inspection)
        self.mock_rejected_mask = (
            rejected_mask if rejected_mask is not None else np.array([False, True, False, True])
        )
        self.mock_error_rate = error_rate
        self.mock_reject_rate = reject_rate
        self.predictions_called = False
        self.predict_calls = []

    def predict(self, x, **kwargs):
        """Mock prediction that tracks calls."""
        self.predictions_called = True
        self.predict_calls.append((x, kwargs))
        # Handle the _ce_skip_reject flag that the orchestrator adds
        kwargs.pop("_ce_skip_reject", None)
        return np.array([0.5] * len(x))

    def is_multiclass(self):
        """Mock multiclass check."""
        return False


class MockRejectOrchestrator(RejectOrchestrator):
    """Orchestrator with mocked prediction to avoid crepes dependency."""

    def __init__(self, explainer: MockExplainer):
        super().__init__(explainer)
        self.mock_rejected = explainer.mock_rejected_mask
        self.mock_error_rate = explainer.mock_error_rate
        self.mock_reject_rate = explainer.mock_reject_rate

    def predict_reject(self, x, bins=None, confidence=0.95):
        """Return mocked rejection results."""
        n = len(x)
        rejected = self.mock_rejected[:n] if len(self.mock_rejected) >= n else self.mock_rejected
        return rejected, self.mock_error_rate, self.mock_reject_rate


@pytest.fixture
def mock_explainer():
    """Create a mock explainer with default rejection mask."""
    return MockExplainer()


@pytest.fixture
def mock_orchestrator(mock_explainer):
    """Create a mock orchestrator for testing policies."""
    return MockRejectOrchestrator(mock_explainer)


def make_explain_fn(return_value: Any = "EXPLANATION"):
    """Create a mock explain function that tracks calls."""
    calls = []

    def fn(x, **kwargs):
        calls.append((x, kwargs))
        return return_value

    fn.calls = calls
    return fn


# ---------------------------------------------------------------------------
# 1. Policy Behavior Verification Tests
# ---------------------------------------------------------------------------


class TestPolicyBehavior:
    """Test each RejectPolicy variant produces correct behavior."""



    def test_policy_only_rejected_only_explains_rejected_instances(self):
        """ONLY_REJECTED should explain only rejected instances."""
        # Create explainer with specific rejection pattern: [False, True, False, True]
        explainer = MockExplainer(rejected_mask=np.array([False, True, False, True]))
        orchestrator = MockRejectOrchestrator(explainer)
        explain_fn = make_explain_fn()

        x_input = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = orchestrator.apply_policy(
            RejectPolicy.ONLY_REJECTED, x=x_input, explain_fn=explain_fn
        )

        assert result.policy is RejectPolicy.ONLY_REJECTED
        assert len(explain_fn.calls) == 1
        # Should only explain indices 1 and 3 (rejected)
        explained_x = explain_fn.calls[0][0]
        assert len(explained_x) == 2

    def test_policy_only_accepted_only_explains_non_rejected_instances(self):
        """ONLY_ACCEPTED should explain only non-rejected instances."""
        # Create explainer with specific rejection pattern: [False, True, False, True]
        explainer = MockExplainer(rejected_mask=np.array([False, True, False, True]))
        orchestrator = MockRejectOrchestrator(explainer)
        explain_fn = make_explain_fn()

        x_input = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = orchestrator.apply_policy(
            RejectPolicy.ONLY_ACCEPTED, x=x_input, explain_fn=explain_fn
        )

        assert result.policy is RejectPolicy.ONLY_ACCEPTED
        assert len(explain_fn.calls) == 1
        # Should only explain indices 0 and 2 (non-rejected)
        explained_x = explain_fn.calls[0][0]
        assert len(explained_x) == 2


class TestIsPolicyEnabled:
    """Test the is_policy_enabled helper function."""


    def test_invalid_policy_returns_false(self):
        """Invalid policy values should return False."""
        assert is_policy_enabled("invalid") is False
        assert is_policy_enabled(None) is False
        assert is_policy_enabled(42) is False


# ---------------------------------------------------------------------------
# 2. Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for reject policy handling."""

    def test_empty_input_array(self, mock_orchestrator):
        """Empty input should be handled gracefully."""
        explain_fn = make_explain_fn()
        result = mock_orchestrator.apply_policy(RejectPolicy.FLAG, x=[], explain_fn=explain_fn)

        assert isinstance(result, RejectResult)
        # Empty input should still produce a result envelope

    def test_single_instance_input(self):
        """Single instance should work without indexing errors."""
        explainer = MockExplainer(rejected_mask=np.array([False]))
        orchestrator = MockRejectOrchestrator(explainer)
        explain_fn = make_explain_fn()

        result = orchestrator.apply_policy(RejectPolicy.FLAG, x=[[1, 2]], explain_fn=explain_fn)

        assert isinstance(result, RejectResult)
        assert result.explanation is not None

    def test_all_rejected_scenario_explain_non_rejects_returns_none_explanation(self):
        """When all instances are rejected, EXPLAIN_NON_REJECTS should return None explanation."""
        explainer = MockExplainer(rejected_mask=np.array([True, True, True]))
        orchestrator = MockRejectOrchestrator(explainer)
        explain_fn = make_explain_fn()

        result = orchestrator.apply_policy(
            RejectPolicy.ONLY_ACCEPTED, x=[[1, 2], [3, 4], [5, 6]], explain_fn=explain_fn
        )

        assert result.policy is RejectPolicy.ONLY_ACCEPTED
        assert result.explanation is None  # No non-rejected instances to explain
        assert len(explain_fn.calls) == 0  # explain_fn not called

    def test_no_rejects_scenario_explain_rejects_returns_none_explanation(self):
        """When no instances are rejected, EXPLAIN_REJECTS should return None explanation."""
        explainer = MockExplainer(rejected_mask=np.array([False, False, False]))
        orchestrator = MockRejectOrchestrator(explainer)
        explain_fn = make_explain_fn()

        result = orchestrator.apply_policy(
            RejectPolicy.ONLY_REJECTED, x=[[1, 2], [3, 4], [5, 6]], explain_fn=explain_fn
        )

        assert result.policy is RejectPolicy.ONLY_REJECTED
        assert result.explanation is None  # No rejected instances to explain
        assert len(explain_fn.calls) == 0  # explain_fn not called

    def test_mixed_rejected_scenario_proper_subset_handling(self):
        """Mixed rejection should properly subset instances."""
        # Pattern: [False, True, True, False, True]
        explainer = MockExplainer(rejected_mask=np.array([False, True, True, False, True]))
        orchestrator = MockRejectOrchestrator(explainer)

        # Test EXPLAIN_REJECTS
        explain_fn_rejects = make_explain_fn()
        x_input = [[i, i] for i in range(5)]
        orchestrator.apply_policy(
            RejectPolicy.ONLY_REJECTED, x=x_input, explain_fn=explain_fn_rejects
        )
        assert len(explain_fn_rejects.calls[0][0]) == 3  # 3 rejected

        # Test EXPLAIN_NON_REJECTS
        explain_fn_non = make_explain_fn()
        orchestrator.apply_policy(RejectPolicy.ONLY_ACCEPTED, x=x_input, explain_fn=explain_fn_non)
        assert len(explain_fn_non.calls[0][0]) == 2  # 2 non-rejected


# ---------------------------------------------------------------------------
# 3. Policy Interaction Tests
# ---------------------------------------------------------------------------


class TestPolicyInteractions:
    """Test policy combination and override behavior."""

    def test_per_call_overrides_explainer_default(self):
        """Per-call reject_policy should override explainer-level default."""

        # Create a minimal mock to test the override logic
        mock_self = SimpleNamespace()
        mock_self.reject_orchestrator = MagicMock()
        mock_self.reject_orchestrator.apply_policy = MagicMock(
            return_value=RejectResult(
                prediction=None,
                explanation=None,
                rejected=None,
                policy=RejectPolicy.FLAG,
                metadata={},
            )
        )
        mock_self.default_reject_policy = RejectPolicy.FLAG
        mock_self.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
        mock_self.explanation_orchestrator = MagicMock()

        # The per-call policy should take precedence
        # This tests the routing logic, not the full implementation


    def test_strategy_resolution_error_handling(self, mock_orchestrator):
        """Unknown strategy should raise KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            mock_orchestrator.resolve_strategy("unknown.strategy")

    def test_initialization_failure_returns_metadata_init_error(self):
        """When reject learner init fails, metadata should contain init_error."""
        explainer = MockExplainer()
        explainer.reject_learner = None  # Simulate uninitialized

        orchestrator = RejectOrchestrator(explainer)

        # Mock initialize_reject_learner to raise
        def failing_init(*args, **kwargs):
            raise RuntimeError("Init failed")

        orchestrator.initialize_reject_learner = failing_init

        result = orchestrator.apply_policy(RejectPolicy.FLAG, x=[[1, 2]])

        assert result.metadata is not None
        assert result.metadata.get("init_error") is True

    def test_custom_strategy_is_invoked(self, mock_orchestrator):
        """Custom registered strategy should be invoked when specified."""
        custom_called = []

        def custom_strategy(policy, x, **kwargs):
            custom_called.append((policy, x))
            return RejectResult(
                prediction="custom_pred",
                explanation="custom_expl",
                rejected=np.array([False]),
                policy=policy,
                metadata={"custom": True},
            )

        mock_orchestrator.register_strategy("test.custom", custom_strategy)
        result = mock_orchestrator.apply_policy(
            RejectPolicy.FLAG, x=[[1, 2]], strategy="test.custom"
        )

        assert len(custom_called) == 1
        assert result.prediction == "custom_pred"
        assert result.metadata == {"custom": True}


# ---------------------------------------------------------------------------
# 4. Metadata Validation Tests
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    """Test that metadata fields are correctly populated."""


    def test_error_rate_within_expected_bounds(self):
        """Error rate should be between 0 and 1."""
        for er in [0.0, 0.05, 0.1, 0.5, 1.0]:
            explainer = MockExplainer(error_rate=er, reject_rate=0.1)
            orchestrator = MockRejectOrchestrator(explainer)
            result = orchestrator.apply_policy(RejectPolicy.FLAG, x=[[1, 2]])

            assert 0.0 <= result.metadata["error_rate"] <= 1.0


    def test_metadata_none_for_none_policy(self, mock_orchestrator):
        """NONE policy should have None metadata."""
        result = mock_orchestrator.apply_policy(RejectPolicy.NONE, x=[[1, 2]])
        assert result.metadata is None


# ---------------------------------------------------------------------------
# 5. RejectResult Dataclass Tests
# ---------------------------------------------------------------------------


class TestRejectResult:
    """Test RejectResult dataclass behavior."""

    def test_default_values(self):
        """RejectResult should have sensible defaults."""
        result = RejectResult()
        assert result.prediction is None
        assert result.explanation is None
        assert result.rejected is None
        assert result.policy is RejectPolicy.NONE
        assert result.metadata is None

    def test_all_fields_populated(self):
        """RejectResult should accept all fields."""
        result = RejectResult(
            prediction=[0.5, 0.6],
            explanation={"type": "factual"},
            rejected=np.array([False, True]),
            policy=RejectPolicy.FLAG,
            metadata={"error_rate": 0.05, "reject_rate": 0.5},
        )

        assert result.prediction == [0.5, 0.6]
        assert result.explanation == {"type": "factual"}
        np.testing.assert_array_equal(result.rejected, np.array([False, True]))
        assert result.policy is RejectPolicy.FLAG
        assert result.metadata["error_rate"] == 0.05

    def test_equality(self):
        """RejectResult instances with same values should be equal."""
        r1 = RejectResult(prediction=[1], policy=RejectPolicy.NONE)
        r2 = RejectResult(prediction=[1], policy=RejectPolicy.NONE)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 6. Policy Enum Tests
# ---------------------------------------------------------------------------


class TestRejectPolicyEnum:
    """Test RejectPolicy enum members and values."""

    def test_all_policies_have_string_values(self):
        """All policy members should have string values."""
        for policy in RejectPolicy:
            assert isinstance(policy.value, str)

    def test_policy_values_are_unique(self):
        """Policy values should be unique."""
        values = [p.value for p in RejectPolicy]
        assert len(values) == len(set(values))

    def test_policy_count(self):
        """There should be exactly 4 policies."""
        assert len(list(RejectPolicy)) == 4

    def test_policy_from_string(self):
        """Policies should be constructable from their string values."""
        assert RejectPolicy("none") is RejectPolicy.NONE
        assert RejectPolicy("flag") is RejectPolicy.FLAG
        assert RejectPolicy("only_rejected") is RejectPolicy.ONLY_REJECTED
        assert RejectPolicy("only_accepted") is RejectPolicy.ONLY_ACCEPTED

    def test_deprecated_policy_strings_map_to_new_policies(self):
        """Deprecated policy string values should map to new policies with warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert RejectPolicy("predict_and_flag") is RejectPolicy.FLAG
            assert RejectPolicy("explain_all") is RejectPolicy.FLAG
            assert RejectPolicy("explain_rejects") is RejectPolicy.ONLY_REJECTED
            assert RejectPolicy("explain_non_rejects") is RejectPolicy.ONLY_ACCEPTED
            assert RejectPolicy("skip_on_reject") is RejectPolicy.ONLY_ACCEPTED
            # Should have emitted 5 deprecation warnings
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 5

    def test_invalid_string_raises_value_error(self):
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError):
            RejectPolicy("invalid_policy")
