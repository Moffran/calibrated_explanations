import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.logging import get_logging_context

def test_explanation_orchestrator_should_set_logging_context():
    explainer = MagicMock()
    explainer.id = "expl-123"
    explainer.plugin_manager = MagicMock()
    explainer.plugin_manager.explanation_plugin_instances = {}
    
    orchestrator = ExplanationOrchestrator(explainer)
    
    captured_context = []
    
    def mock_resolve(*args, **kwargs):
        captured_context.append(get_logging_context())
        return MagicMock(), "test-plugin"

    with patch.object(ExplanationOrchestrator, "resolve_plugin", side_effect=mock_resolve):
        try:
            # invoke calls ensure_plugin which calls resolve_plugin within a logging_context
            orchestrator.invoke(
                mode="factual",
                x=np.array([[1, 2]]),
                threshold=None,
                low_high_percentiles=(5, 95),
                bins=None,
                features_to_ignore=None
            )
        except Exception:
            pass
                
    assert len(captured_context) > 0
    assert any(ctx.get("explainer_id") == "expl-123" for ctx in captured_context)

def test_prediction_orchestrator_should_set_logging_context():
    explainer = MagicMock()
    explainer.id = "expl-456"
    explainer.plugin_manager = MagicMock()
    
    orchestrator = PredictionOrchestrator(explainer)
    
    captured_context = []
    
    def mock_resolve(*args, **kwargs):
        captured_context.append(get_logging_context())
        return MagicMock(), "test-plugin"
        
    with patch.object(PredictionOrchestrator, "resolve_interval_plugin", side_effect=mock_resolve):
        with patch.object(PredictionOrchestrator, "ensure_interval_runtime_state"):
            with patch.object(PredictionOrchestrator, "gather_interval_hints"):
                try:
                    orchestrator.obtain_interval_calibrator(fast=False, metadata={})
                except Exception:
                    pass
        
    assert len(captured_context) > 0
    assert any(ctx.get("explainer_id") == "expl-456" for ctx in captured_context)

def test_prediction_orchestrator_should_set_plugin_identifier_context():
    explainer = MagicMock()
    explainer.id = "expl-789"
    explainer.plugin_manager = MagicMock()
    
    orchestrator = PredictionOrchestrator(explainer)
    
    captured_context = []
    
    plugin = MagicMock()
    def mock_create(*args, **kwargs):
        captured_context.append(get_logging_context())
        return MagicMock()
    plugin.create.side_effect = mock_create
    
    with patch.object(PredictionOrchestrator, "resolve_interval_plugin", return_value=(plugin, "resolved-plugin")):
        with patch.object(PredictionOrchestrator, "ensure_interval_runtime_state"):
            with patch.object(PredictionOrchestrator, "gather_interval_hints"):
                with patch.object(PredictionOrchestrator, "build_interval_context"):
                    try:
                        orchestrator.obtain_interval_calibrator(fast=False, metadata={})
                    except Exception:
                        pass
                        
    assert len(captured_context) > 0
    assert any(ctx.get("plugin_identifier") == "resolved-plugin" for ctx in captured_context)
