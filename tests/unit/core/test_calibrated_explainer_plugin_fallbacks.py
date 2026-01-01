"""Unit tests for CalibratedExplainer fallback paths.

These tests exercise defensive branches that are primarily used when a
fully initialized :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer`
instance is not available (for example, in legacy usage or during error
handling). The goal is to ensure the fallback logic remains functional and
boost coverage for branches not hit by higher-level integration tests.
"""

from __future__ import annotations
