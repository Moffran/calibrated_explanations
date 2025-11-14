"""Prediction orchestration for CalibratedExplainer.

This module provides the PredictionOrchestrator class which coordinates
prediction pipeline execution, including interval calibration, difficulty
estimation, and uncertainty quantification.

Part of Phase 1b: Delegate Prediction Orchestration (ADR-001, ADR-004).
"""

from .orchestrator import PredictionOrchestrator

__all__ = [
    "PredictionOrchestrator",
]
