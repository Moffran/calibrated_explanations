# CE Skill Trigger Phrase Catalog

This reference preserves important legacy trigger phrases that were removed from
frontmatter descriptions during strict normalization.

Use this catalog when refining routing logic or deciding between overlapping
skills.

## CE pipeline and explanations

### `ce-pipeline-builder`
- "how do I use CE"
- "set up a CE pipeline"
- "fit and calibrate"
- "WrapCalibratedExplainer"

### `ce-factual-explain`
- "explain factual"
- "factual rules for prediction"
- "what rules justify this prediction"
- "explain_guarded_factual"

### `ce-alternatives-explore`
- "explore alternatives"
- "counterfactual explanations"
- "what-if analysis"
- "recourse"
- "super_explanations / counter_explanations / ensured_explanations"

### `ce-explain-interact`
- "to_narrative"
- "add conjunctions"
- "filter_top / filter_features / filter_rule_sizes"
- "inspect explanation output"

### `ce-calibrated-predict`
- "predict_proba calibrated"
- "uq_interval"
- "prediction with bounds"
- "calibrated probability"

### `ce-regression-intervals`
- "low_high_percentiles"
- "threshold regression"
- "probabilistic regression"
- "DifficultyEstimator"

### `ce-mondrian-conditional`
- "conditional calibration"
- "MondrianCategorizer"
- "group-specific calibration"
- "protected attribute"

### `ce-reject-policy`
- "predict_reject"
- "RejectPolicy"
- "defer / abstain / selective prediction"
- "RejectResult"

## Plugins and visualization

### `ce-plugin-scaffold`
- "new calibrator plugin"
- "new explanation plugin"
- "new plot plugin"
- "plugin_meta"

### `ce-plugin-audit`
- "plugin registry compliance"
- "plugin trust model"
- "plugin_api_version"

### `ce-plotspec-author`
- "new plot kind"
- "PlotSpec"
- "validate_plotspec"

### `ce-plot-review`
- "ADR-023 matplotlib exemption"
- "lazy plot import"
- "plot conformance"

## Quality, testing, and governance

### `ce-test-author`
- "write tests for"
- "add test coverage"
- "TDD for this module"

### `ce-test-audit`
- "find test anti-patterns"
- "improve test quality"
- "over-testing"

### `ce-code-review`
- "check my PR"
- "ADR conformance"
- "coding standards check"

### `ce-deprecation`
- "deprecate parameter"
- "DeprecationWarning"
- "CE_DEPRECATIONS=error"
- "mitigation guide"

### `ce-serialization-audit` / `ce-serializer-impl`
- "save_state / load_state"
- "to_primitive / from_primitive"
- "schema_version"
- "round-trip serialization"

## RTD and skill operations

### `ce-rtd-auditor`
- "audit docs"
- "RTD quality"
- "toctree/link correctness"

### `ce-rtd-writer`
- "write docs page"
- "restructure RTD content"
- "audience-specific docs update"

### `ce-skill-audit`
- "audit skills"
- "SKILL.md quality"
- "trigger precision review"

### `ce-skill-creator`
- "create new skill"
- "refactor poor skill"
- "add assets/references/scripts for skill"
