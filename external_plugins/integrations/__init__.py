"""Integration pipelines for LIME, SHAP, and other explanation methods.

These modules provide delegated execution for third-party explanation methods
integrated with calibrated explanations.
"""

from __future__ import annotations

from .lime_pipeline import LimePipeline

__all__ = ["LimePipeline"]
