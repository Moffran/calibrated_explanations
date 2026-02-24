# Agent skill catalogue

This ReadTheDocs page lists every repository skill under `.claude/skills/`.
Use it as the human-readable index, and use each linked `SKILL.md` as the
execution contract.

> Source of truth for inventory: filesystem directories under `.claude/skills/`.
> Synchronization is enforced by `ce-skill-registry-sync`.

| Skill | Purpose | Link |
|---|---|---|
| `ce-adr-author` | Author or update ADR documents. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-adr-author/SKILL.md) |
| `ce-adr-consult` | Find and apply relevant ADRs before implementation. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-adr-consult/SKILL.md) |
| `ce-alternatives-explore` | Generate alternative and counterfactual explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-alternatives-explore/SKILL.md) |
| `ce-calibrated-predict` | Produce calibrated predictions without explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-calibrated-predict/SKILL.md) |
| `ce-classification` | Calibrated Explanations for binary and multiclass tasks. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-classification/SKILL.md) |
| `ce-code-quality-auditor` | Identify quality risks and anti-patterns per ADR-030. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-code-quality-auditor/SKILL.md) |
| `ce-code-review` | Perform CE-focused code review and risk checks. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-code-review/SKILL.md) |
| `ce-deadcode-hunter` | Identify and clean up unreachable or non-contributing code. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-deadcode-hunter/SKILL.md) |
| `ce-deprecation` | Apply ADR-011 compliant deprecation patterns. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-deprecation/SKILL.md) |
| `ce-devils-advocate` | Rigorously review agent proposals for risks and blind spots. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-devils-advocate/SKILL.md) |
| `ce-docstring-author` | Write or repair numpy-style docstrings. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-docstring-author/SKILL.md) |
| `ce-explain-interact` | Post-process, narrate, filter, and plot explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-explain-interact/SKILL.md) |
| `ce-factual-explain` | Generate factual calibrated explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-factual-explain/SKILL.md) |
| `ce-fallback-impl` | Implement visible fallback behavior in production code. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-fallback-impl/SKILL.md) |
| `ce-fallback-test` | Validate fallback warnings and test coverage behavior. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-fallback-test/SKILL.md) |
| `ce-logging-observability` | Manage logging, governance, and audit context (ADR-028). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-logging-observability/SKILL.md) |
| `ce-modality-extension` | Extend CE to new data modalities safely. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-modality-extension/SKILL.md) |
| `ce-mondrian-conditional` | Configure conditional and Mondrian calibration. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-mondrian-conditional/SKILL.md) |
| `ce-notebook-audit` | Audit notebooks for API and policy compliance. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-notebook-audit/SKILL.md) |
| `ce-onboard` | Prime new sessions with CE-first context and routing. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-onboard/SKILL.md) |
| `ce-payload-governance` | Manage and validate explanation payloads (ADR-005). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-payload-governance/SKILL.md) |
| `ce-pipeline-builder` | Build CE-first fit/calibrate/explain pipelines. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-pipeline-builder/SKILL.md) |
| `ce-plot-review` | Review visualization and plotting quality. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plot-review/SKILL.md) |
| `ce-plotspec-author` | Author or evolve PlotSpec-based visual outputs. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plotspec-author/SKILL.md) |
| `ce-plugin-audit` | Audit plugin conformance and boundary rules. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plugin-audit/SKILL.md) |
| `ce-plugin-scaffold` | Scaffold new plugins from CE contracts. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plugin-scaffold/SKILL.md) |
| `ce-regression-intervals` | Workflows for conformal/probabilistic regression intervals. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-regression-intervals/SKILL.md) |
| `ce-reject-policy` | Configure and validate reject/defer policy behavior. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-reject-policy/SKILL.md) |
| `ce-release-check` | Select and validate release tasks and gates. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-release-check/SKILL.md) |
| `ce-serialization-audit` | Audit serialization behavior and test coverage. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-serialization-audit/SKILL.md) |
| `ce-serializer-impl` | Implement serialization and persistence support. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-serializer-impl/SKILL.md) |
| `ce-skill-registry-sync` | Keep all skill registries synchronized. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-skill-registry-sync/SKILL.md) |
| `ce-test-audit` | Audit tests against ADR-030 and test standards. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-audit/SKILL.md) |
| `ce-test-author` | Author high-signal CE tests. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-author/SKILL.md) |
| `ce-test-creator` | Design high-efficiency tests to close coverage gaps accurately. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-creator/SKILL.md) |
| `ce-test-pruning-expert` | Identify and remove redundant or low-value tests (ADR-030). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-pruning-expert/SKILL.md) |
| `ce-test-quality-method` | Coordinate the full Test Quality Method (ADR-030). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-quality-method/SKILL.md) |
