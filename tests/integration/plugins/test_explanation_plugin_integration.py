# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

from tests.helpers.explainer_utils import (
    make_explainer_from_dataset,
    make_multiclass_explainer_from_dataset,
    make_regression_explainer_from_dataset,
    assert_explanation_collections_equal,
)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_explanations.core.exceptions import ConfigurationError
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
    register_explanation_plugin,
    unregister,
)


class RegressionOnlyFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.regression_only_factual",
        "tasks": ("regression",),
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:regression",
        ],
    }


class IncompatibleFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.incompatible_factual",
        "fallbacks": ("core.explanation.factual",),
        "trust": False,
    }

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return False


class FutureSchemaFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.future_schema_factual",
        "schema_version": 0,
    }


@pytest.fixture(autouse=True)
def restore_registry():
    clear_explanation_plugins()
    ensure_builtin_plugins()
    yield
    clear_explanation_plugins()
    ensure_builtin_plugins()


def test_factual_fallback_dependency_propagation(monkeypatch, binary_dataset):
    plugin = IncompatibleFactualPlugin()
    identifier = "tests.integration.incompatible_factual"
    register_explanation_plugin(identifier, plugin)

    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", identifier)

    try:
        explainer, x_test = make_explainer_from_dataset(binary_dataset)
        explainer.explain_factual(x_test[:2])
        assert plugin._context is None

        chain = explainer._explanation_plugin_fallbacks["factual"]
        assert chain[0] == identifier
        assert "core.explanation.factual" in chain

        context = explainer._explanation_contexts["factual"]
        assert context.interval_settings["dependencies"] == ("core.interval.legacy",)
        fallbacks = context.plot_settings["fallbacks"]
        assert "legacy" in fallbacks
        assert "plot_spec.default" in fallbacks
    finally:
        unregister(plugin)


def test_missing_override_identifier_errors(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, factual_plugin="tests.integration.missing"
    )

    with pytest.raises(ConfigurationError, match="missing: not registered"):
        explainer.explain_factual(x_test[:1])


def test_schema_version_override_error(binary_dataset):
    plugin = FutureSchemaFactualPlugin()
    explainer, x_test = make_explainer_from_dataset(binary_dataset, factual_plugin=plugin)

    with pytest.raises(ConfigurationError, match="schema_version"):
        explainer.explain_factual(x_test[:1])


def test_factual_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(binary_dataset)
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_sequential_factual_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, factual_plugin="core.explanation.factual.sequential"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_instance_parallel_factual_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, factual_plugin="core.explanation.factual.instance_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_feature_parallel_factual_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, factual_plugin="core.explanation.factual.feature_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_alternative_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(binary_dataset)
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_sequential_alternative_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, alternative_plugin="core.explanation.alternative.sequential"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_instance_parallel_alternative_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, alternative_plugin="core.explanation.alternative.instance_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_feature_parallel_alternative_explanations_match_legacy(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, alternative_plugin="core.explanation.alternative.feature_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


# =====================================================================
# Multiclass Classification Tests
# =====================================================================


def test_multiclass_factual_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(multiclass_dataset)
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_sequential_factual_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, factual_plugin="core.explanation.factual.sequential"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_instance_parallel_factual_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, factual_plugin="core.explanation.factual.instance_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_feature_parallel_factual_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, factual_plugin="core.explanation.factual.feature_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_alternative_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(multiclass_dataset)
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_sequential_alternative_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, alternative_plugin="core.explanation.alternative.sequential"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_instance_parallel_alternative_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, alternative_plugin="core.explanation.alternative.instance_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_multiclass_feature_parallel_alternative_explanations_match_legacy(multiclass_dataset):
    explainer, x_test = make_multiclass_explainer_from_dataset(
        multiclass_dataset, alternative_plugin="core.explanation.alternative.feature_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


# =====================================================================
# Regression Tests
# =====================================================================


def test_regression_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(regression_dataset)
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_sequential_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.sequential"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_instance_parallel_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.instance_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_feature_parallel_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.feature_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_alternative_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(regression_dataset)
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_sequential_alternative_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.sequential"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_instance_parallel_alternative_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.instance_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


def test_regression_feature_parallel_alternative_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.feature_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3])

    assert_explanation_collections_equal(plugin_result, legacy)


# =====================================================================
# Probabilistic Regression Tests (with threshold)
# =====================================================================


def test_probabilistic_regression_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(regression_dataset)
    legacy = explainer.explain_factual(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_sequential_factual_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.sequential"
    )
    legacy = explainer.explain_factual(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_instance_parallel_factual_explanations_match_legacy(
    regression_dataset,
):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.instance_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_feature_parallel_factual_explanations_match_legacy(
    regression_dataset,
):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, factual_plugin="core.explanation.factual.feature_parallel"
    )
    legacy = explainer.explain_factual(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explain_factual(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_alternative_explanations_match_legacy(regression_dataset):
    explainer, x_test = make_regression_explainer_from_dataset(regression_dataset)
    legacy = explainer.explore_alternatives(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_sequential_alternative_explanations_match_legacy(
    regression_dataset,
):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.sequential"
    )
    legacy = explainer.explore_alternatives(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_instance_parallel_alternative_explanations_match_legacy(
    regression_dataset,
):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.instance_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)


def test_probabilistic_regression_feature_parallel_alternative_explanations_match_legacy(
    regression_dataset,
):
    explainer, x_test = make_regression_explainer_from_dataset(
        regression_dataset, alternative_plugin="core.explanation.alternative.feature_parallel"
    )
    legacy = explainer.explore_alternatives(x_test[:3], threshold=0.5, _use_plugin=False)
    plugin_result = explainer.explore_alternatives(x_test[:3], threshold=0.5)

    assert_explanation_collections_equal(plugin_result, legacy)
