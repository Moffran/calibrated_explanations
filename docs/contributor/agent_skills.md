# Agent skill catalogue

This ReadTheDocs page lists every repository skill under `.claude/skills/`.
Use it as the human-readable index, and use each linked `SKILL.md` as the
execution contract.

> Source of truth for inventory: filesystem directories under `.claude/skills/`.
> Synchronization is enforced by the repository skill-registry workflow.

## 1) Practitioner Workflows

### 1.1 Pipeline Setup and Core Prediction

| Skill | Purpose | Link |
|---|---|---|
| `ce-pipeline-builder` | Build CE-first fit/calibrate/explain pipelines. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-pipeline-builder/SKILL.md) |
| `ce-calibrated-predict` | Produce calibrated predictions without explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-calibrated-predict/SKILL.md) |
| `ce-classification` | Calibrated Explanations for binary and multiclass tasks. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-classification/SKILL.md) |
| `ce-data-preparation` | Validate and preprocess input data for CE pipelines. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-data-preparation/SKILL.md) |

### 1.2 Explanation Generation and Interpretation

| Skill | Purpose | Link |
|---|---|---|
| `ce-factual-explain` | Generate factual calibrated explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-factual-explain/SKILL.md) |
| `ce-alternatives-explore` | Generate alternative and counterfactual explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-alternatives-explore/SKILL.md) |
| `ce-explain-interact` | Post-process, narrate, filter, and plot explanations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-explain-interact/SKILL.md) |

### 1.3 Uncertainty and Decision Policies

| Skill | Purpose | Link |
|---|---|---|
| `ce-regression-intervals` | Workflows for conformal/probabilistic regression intervals. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-regression-intervals/SKILL.md) |
| `ce-mondrian-conditional` | Configure conditional and Mondrian calibration. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-mondrian-conditional/SKILL.md) |
| `ce-reject-policy` | Configure and validate reject/defer policy behavior. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-reject-policy/SKILL.md) |

### 1.4 Integrations and Performance

| Skill | Purpose | Link |
|---|---|---|
| `ce-integration-compare` | Guide CE integration with SHAP and LIME. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-integration-compare/SKILL.md) |
| `ce-performance-tuning` | Configure caching, parallelism, and batch-size tuning. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-performance-tuning/SKILL.md) |

## 2) Contributor Implementation and Extensibility

### 2.1 Plugin and Modality Engineering

| Skill | Purpose | Link |
|---|---|---|
| `ce-plugin-scaffold` | Scaffold new plugins from CE contracts. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plugin-scaffold/SKILL.md) |
| `ce-plugin-audit` | Audit plugin conformance and boundary rules. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plugin-audit/SKILL.md) |
| `ce-modality-extension` | Extend CE to new data modalities safely. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-modality-extension/SKILL.md) |

### 2.2 Runtime Behavior, Persistence, and Payload Contracts

| Skill | Purpose | Link |
|---|---|---|
| `ce-fallback-impl` | Implement visible fallback behavior in production code. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-fallback-impl/SKILL.md) |
| `ce-serializer-impl` | Implement serialization and persistence support. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-serializer-impl/SKILL.md) |
| `ce-serialization-audit` | Audit serialization behavior and test coverage. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-serialization-audit/SKILL.md) |
| `ce-payload-governance` | Manage and validate explanation payloads (ADR-005). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-payload-governance/SKILL.md) |
| `ce-logging-observability` | Manage logging, governance, and audit context (ADR-028). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-logging-observability/SKILL.md) |

### 2.3 Visualization Engineering

| Skill | Purpose | Link |
|---|---|---|
| `ce-plotspec-author` | Author or evolve PlotSpec-based visual outputs. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plotspec-author/SKILL.md) |
| `ce-plot-review` | Review visualization and plotting quality. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-plot-review/SKILL.md) |

## 3) Quality, Testing, and Risk Control

### 3.1 Test Authoring and Coverage Validation

| Skill | Purpose | Link |
|---|---|---|
| `ce-test-author` | Author high-signal CE tests. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-author/SKILL.md) |
| `ce-fallback-test` | Validate fallback warnings and test coverage behavior. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-fallback-test/SKILL.md) |
| `ce-test-audit` | Audit tests against ADR-030 and test standards. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-audit/SKILL.md) |

### 3.2 Test Suite Optimization and Remediation

| Skill | Purpose | Link |
|---|---|---|
| `ce-test-creator` | Design high-efficiency tests to close coverage gaps accurately. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-creator/SKILL.md) |
| `ce-test-pruning-expert` | Identify and remove redundant or low-value tests (ADR-030). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-pruning-expert/SKILL.md) |
| `ce-test-quality-method` | Coordinate the full Test Quality Method (ADR-030). | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-test-quality-method/SKILL.md) |

### 3.3 Code and Risk Review

| Skill | Purpose | Link |
|---|---|---|
| `ce-code-review` | Perform CE-focused code review and risk checks. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-code-review/SKILL.md) |
| `ce-code-quality-auditor` | Identify quality risks and anti-patterns per ADR-030. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-code-quality-auditor/SKILL.md) |
| `ce-deadcode-hunter` | Identify and clean up unreachable or non-contributing code. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-deadcode-hunter/SKILL.md) |
| `ce-devils-advocate` | Rigorously review agent proposals for risks and blind spots. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-devils-advocate/SKILL.md) |

## 4) Governance, Documentation, and Skill Operations

### 4.1 Architecture and Lifecycle Governance

| Skill | Purpose | Link |
|---|---|---|
| `ce-adr-consult` | Find and apply relevant ADRs before implementation. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-adr-consult/SKILL.md) |
| `ce-adr-gap-analyzer` | Analyze ADR compliance by verifying implementation and RTD against ADR intent. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-adr-gap-analyzer/SKILL.md) |
| `ce-adr-author` | Author or update ADR documents. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-adr-author/SKILL.md) |
| `ce-deprecation` | Apply ADR-011 compliant deprecation patterns. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-deprecation/SKILL.md) |
| `ce-release-check` | Select and validate release tasks and gates. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-release-check/SKILL.md) |
| `ce-release-planner` | Analyze release plan and produce vX.Y.Z implementation plans. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-release-planner/SKILL.md) |
| `ce-release-task` | Identify, implement, and verify individual release tasks. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-release-task/SKILL.md) |
| `ce-release-finalize` | Execute the PyPI release checklist. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-release-finalize/SKILL.md) |
| `ce-regulatory-compliance` | Map CE capabilities to EU AI Act, GDPR, and liability directive obligations. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-regulatory-compliance/SKILL.md) |
| `ce-standards-gap-analyzer` | Analyze STD compliance by verifying implementation and RTD against STD intent. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-standards-gap-analyzer/SKILL.md) |

### 4.2 Documentation Authoring and Audit

| Skill | Purpose | Link |
|---|---|---|
| `ce-docstring-author` | Write or repair numpy-style docstrings. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-docstring-author/SKILL.md) |
| `ce-notebook-audit` | Audit notebooks for API and policy compliance. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-notebook-audit/SKILL.md) |
| `ce-rtd-auditor` | Audit ReadTheDocs content for structure, accuracy, and governance alignment. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-rtd-auditor/SKILL.md) |
| `ce-rtd-writer` | Write or revise ReadTheDocs pages with audience-aware structure. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-rtd-writer/SKILL.md) |

### 4.3 Skill System Operations

| Skill | Purpose | Link |
|---|---|---|
| `ce-onboard` | Prime new sessions with CE-first context and routing. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-onboard/SKILL.md) |
| `ce-skill-audit` | Audit `.claude/skills` for Claude skill-authoring compliance and drift. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-skill-audit/SKILL.md) |
| `ce-skill-creator` | Create or refactor skills with templates, assets, and quality guardrails. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-skill-creator/SKILL.md) |
| `ce-skill-registry-sync` | Keep all skill registries synchronized. | [SKILL.md](https://github.com/Moffran/calibrated_explanations/blob/main/.claude/skills/ce-skill-registry-sync/SKILL.md) |
