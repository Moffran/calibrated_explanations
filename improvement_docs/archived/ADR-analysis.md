# ADR-001

## 1

Severity scale
Score	Meaning	When to use
4 – Critical	Blocks the architectural goal or directly contradicts the ADR.	Missing mandated package boundaries, hard cross-layer coupling.
3 – High	Major misalignment that risks regressions or complicates follow-through phases.	Consolidated modules that should be split, broad API exposure.
2 – Medium	Partial implementation; needs work but not immediately blocking.	Naming drift, functionality located in interim modules.
1 – Low	Minor clean-ups or documentation clarifications.	Cosmetic or bookkeeping adjustments.
Gap summary
Gap	ADR expectation	Current state	Severity
Calibration layer still lives in core	Calibration code isolated under calibrated_explanations.calibration.	Calibration helpers and Venn-Abers live inside core, keeping tight coupling with explainer orchestration.	4
Core imports downstream siblings	Core should only speak through domain models / explicit interfaces; calibration must not import explanations.	core.calibrated_explainer imports explanations, plugins, perf, and API helpers directly.	4
Cache & parallel boundaries not split	Separate cache and parallel packages (ADR phases 1A/2).	Both concerns combined under perf, with shared factory and exports.	3
Schema validation module missing	Dedicated schema package with JSON schema + helpers.	Validation helper implemented in serialization.py; only raw JSON file under schemas/.	2
Public API consolidation incomplete	__init__ should only re-export stable, high-level entry points.	Package init eagerly exposes explanation classes, discretizers, and viz namespace through lazy getattr.	2
Additional top-level namespaces lack ADR coverage	ADR lists core/calibration/explanations/cache/parallel/schema/plugins/viz/utils.	Additional packages (api, legacy, perf, plotting.py, serialization.py) persist without documented boundary justification.	2
Detailed findings
1. Calibration layer still embedded in core (Severity 4)
ADR-001 calls for a distinct calibrated_explanations.calibration package to house calibration algorithms and conformal predictors, separating them from core orchestration.
In the codebase, calibration utilities such as calibration_helpers.py and venn_abers.py remain under core/, and are treated as part of the same module tree as the explainer entry point.
This blocks the intended structural decoupling and prevents Phase 1A’s mechanical move from completing, because calibration logic still shares state and imports with core orchestration.

Recommended next steps

Extract calibration algorithms and helpers into a dedicated calibration/ package, providing a thin interface consumed by core through domain models.

Introduce compatibility shims so existing imports remain functional during migration.

2. Cross-talk from core into siblings (Severity 4)
The ADR forbids sibling cross-talk except through core models and explicit interfaces, and specifically states that calibration code must not import explanation modules.
core/calibrated_explainer.py currently imports explanation collections, plugin registries, performance utilities, and API param validators directly, creating a dense dependency hub that violates the intended layering.
Until this is untangled, the module remains a choke-point for refactors, and the “no cross-talk” policy is unenforceable.

Recommended next steps

Define explicit interfaces in core that depend only on domain models, and have downstream packages (e.g., plugins, explanations) register implementations via dependency inversion.

Move parameter validation helpers to the appropriate boundary (likely core or calibration) or expose them through a sanctioned interface to avoid direct API imports.

3. Cache & parallel split not realized (Severity 3)
ADR-001 establishes cache and parallel as independent top-level internal packages, anticipating later feature-flagged adoption.
The repository instead groups these concerns under perf/, exporting both cache and parallel primitives from a shared namespace and factory.
This divergence makes it harder to toggle or evolve the cache and parallel layers independently, delaying Phase 2+ readiness.

Recommended next steps

Split perf into cache/ and parallel/ packages, retaining a thin façade for backwards compatibility.

Update factories and configuration helpers to compose the two packages instead of bundling them implicitly.

4. Schema validation co-located with serialization (Severity 2)
The ADR requires a schema package that isolates JSON schema definitions and validation helpers to prevent heavy imports for consumers that do not need validation.
Currently, schema validation resides in serialization.py, which loads the JSON schema via importlib.resources and performs validation when jsonschema is available.
Although the raw schema file lives under schemas/, there is no dedicated package boundary, so downstream code must import serialization logic even when it only needs validation helpers.

Recommended next steps

Promote schemas/ to a proper package (e.g., schema/__init__.py) containing validation utilities.

Refactor serialization to depend on the schema helper via the new boundary, keeping optional validation lightweight.

5. Public API consolidation incomplete (Severity 2)
ADR-001 asks for calibrated_explanations.__init__ to re-export only stable, high-level entry points, treating everything else as private.
The current __init__ lazily exposes explanation classes, discretizers, and the viz namespace, enabling from calibrated_explanations import AlternativeExplanation despite those symbols not being part of __all__.
This effectively maintains a broad surface area, risking accidental API commitments.

Recommended next steps

Restrict lazy getters to the officially supported façade (e.g., CalibratedExplainer, configuration entry points), and require consumers to import explanations and discretizers from their dedicated packages.

Document the narrowed API and provide guidance for migration.

6. Additional namespaces outside ADR scope (Severity 2)
Several top-level packages (api, legacy, perf, standalone plotting.py, serialization.py) remain without explicit coverage in ADR-001, suggesting either undocumented scope creep or missing follow-up ADRs.
While some may be transitional (e.g., legacy), their presence should be reconciled with the architectural boundary map to avoid confusion for new contributors.

Recommended next steps

Document each additional namespace in either ADR-001 (via amendments) or subsequent ADRs, clarifying their role.

Where possible, fold legacy or redundant modules into the approved packages to simplify navigation.

Observed progress
Expected namespaces such as core, explanations, plugins, utils, and viz exist, indicating that the mechanical package split has started.

Utility modules remain free of upward dependencies, adhering to the “utils must not depend on higher-level packages” rule (e.g., utils.helper only touches standard libs and numpy/pandas).

Addressing the high-severity gaps (calibration extraction and cross-talk elimination) should be prioritized to unlock subsequent migration phases and enforce the architectural boundaries defined in ADR-001.

## 2

Overview
I reviewed ADR-001 and compared its prescribed architectural boundaries with the current calibrated_explanations codebase. Findings are grouped by gap, each tagged with a severity score on a 1–5 scale (5 = critical, 1 = informational).

Severity scale
5 – Critical: Directly contradicts an accepted ADR outcome and blocks intended modularity.

4 – High: Major erosion of the boundary that risks regressions or future refactors.

3 – Medium: Noticeable drift from the design that should be corrected soon.

2 – Low: Minor divergence; track for cleanup.

1 – Informational: Observed but not immediately harmful.

Gap analysis
Area	Severity	Details & Evidence	Suggested action
Dedicated calibration package never materialized	5	ADR-001 requires calibration algorithms to live in calibrated_explanations.calibration, separate from core. In practice, the CalibratedExplainer class, its calibration workflow, and associated state live inside calibrated_explanations.core, confirming that calibration logic was not split out.	Stand up a calibrated_explanations/calibration package, move the calibration routines out of core, and leave core focused on domain models/interfaces.
Calibration code imports explanations module	4	ADR-001 forbids sibling cross-talk, specifically stating that calibration code must not import explanation modules. CalibratedExplainer imports AlternativeExplanations/CalibratedExplanations directly and even keeps explanation instances on the class, breaching that rule.	Introduce explicit interfaces/domain models that explanations can consume so core stays decoupled from explanations.
Cache & parallel layers merged under perf	3	ADR-001 calls for distinct cache and parallel packages. Instead, both concerns live in a single perf namespace, and core imports them together via calibrated_explanations.perf.	Split perf into cache and parallel packages (keeping thin compatibility shims if needed) to match the ADR boundary map.
Schema boundary not isolated	3	The ADR expects a schema package hosting JSON schema helpers. Currently, schema assets live as raw JSON under schemas/, and validation logic resides in the top-level serialization.py, making schema utilities effectively part of serialization rather than an isolated layer.	Create a Python package (calibrated_explanations/schema) that owns schema loading/validation and have serialization depend on it, aligning with the intended separation.
Public API exports exceed “stable entry points only”	3	ADR-001 asks for a narrow __init__ that re-exports only stable factories/dataclasses/high-level run functions. The current __init__.py eagerly re-exports visualization, explanation classes, discretizers, and more, broadening the public surface beyond that guidance.	Trim the exports to just the sanctioned entry points (e.g., wrappers/factories) and move convenience re-exports behind documented public modules instead.
Additional observations (low severity)
Utilities remain largely isolated from higher-level packages, which aligns with ADR intent, so no immediate action is required.

No commands were executed, so there are no test results to report.

# ADR-002

## 1

Severity scale
High – Blocks ADR intent or leaves major user-facing inconsistency/risk.

Medium – Partial compliance; core behavior works but ADR outcomes are weakened.

Low – Cosmetic/documentation misalignment with minimal functional impact.

Compliance snapshot
The ADR-specified exception taxonomy is present in core/exceptions.py, including the details payload hook.

Validation helpers, canonicalization utilities, and tests exist, and strict mypy overrides target the new modules, aligning with the Phase 1B rollout plan.

Gap analysis
Area	Gap & impact	Severity	Evidence
Exception replacement	Many core paths still raise raw ValueError/RuntimeError, so callers cannot rely on the new taxonomy (e.g., interval regressor, Venn-Abers, explanation models, helper utilities).	High
Validation API contract	validate_inputs is implemented as a generic non-null checker instead of the richer signature promised by the ADR (x, y, task inference knobs, etc.), so downstream code cannot adopt the documented contract.	High
Optional validate(...) helper	The ADR calls for a reusable validate(condition, exc_cls, message, *, details=None) helper; no such utility is exposed, forcing ad-hoc checks throughout the codebase.	Medium
explain_exception & debugging hooks	No implementation of the proposed explain_exception(e) or “last validation errors” debug surface exists, so developers still lack the standardized human-readable reporting channel.	Medium
Parameter combination guardrails	validate_param_combination is a no-op, leaving callers without the “basic consistency checks” envisioned for Phase 1B.	Medium
Legacy fit-check behavior	utils.helper.check_is_fitted still raises RuntimeError, undermining the unified NotFittedError pathway and confusing callers who expect ADR-compliant errors.	Medium
Recommended focus
Prioritize replacing remaining ValueError/RuntimeError raises with the ADR taxonomy, starting in high-traffic prediction and explanation paths.

Expand validate_inputs (or introduce the richer function under a new name) to honor the ADR contract, and add targeted tests.

Add the optional validate helper plus explain_exception/debug surfaces to encourage consistent usage patterns.

Flesh out validate_param_combination with the minimal Phase 1B checks (e.g., mutual exclusivity, threshold sanity) to deliver on ADR promises.

## 2

Severity scale
Score	Meaning
3 (High)	Directly violates the ADR’s Phase 1B contract or blocks downstream tooling. Needs prioritised remediation.
2 (Medium)	Partially fulfils the ADR but leaves important benefits unrealised; should be scheduled soon.
1 (Low)	Nice-to-have improvements or hygiene that can follow once higher-severity gaps close.
What’s already in place
The agreed exception taxonomy lives in core/exceptions.py, including the CalibratedError base class with optional details, and is backed by unit tests to protect the hierarchy and payload contract.

Core validation helpers and strict mypy configuration exist for the targeted modules, satisfying the ADR’s “implement core/validation.py” and “wire mypy strict” deliverables.

Parameter canonicalisation helpers (canonicalize_kwargs, warn_on_aliases) are implemented, exercised by tests, and wired into public prediction surfaces of CalibratedExplainer and the wrapper layer.

Gap analysis
Area needing attention	Severity	Evidence & impact	Suggested next actions
Legacy ValueError usage across core/plugin modules	3	ADR-002 calls for replacing generic exceptions in core paths with the structured taxonomy. Yet critical components such as interval_regressor, Venn-Abers helpers, and the plugin registry still raise bare ValueError, denying callers the promised semantics and error codes.	Inventory the remaining ValueError/RuntimeError raises in core and plugins, map each to the closest ADR taxonomy member (or introduce new subclasses if truly required), and update call sites/tests accordingly. Prioritise paths exercised by public APIs and plugin validation to restore ADR conformance.
Validation API mismatch and limited reuse	2	The ADR specifies a rich validate_inputs(x, y=None, task="auto", allow_nan=False, require_y=False, n_features=None, class_labels=None, check_finite=True) signature plus helper reuse at entry points. The current implementation exposes a generic *args/**kwargs variant without the contract, and adds a separate validate_inputs_matrix to cover part of the behaviour. Fit-state checks are also duplicated manually in WrapCalibratedExplainer instead of using validate_fit_state, undercutting the centralised guard rail the ADR promised.	Align validate_inputs with the ADR signature (possibly delegating to validate_inputs_matrix) and refactor wrapper/core entry points to call the shared helpers (e.g., swap _assert_fitted for validate_fit_state). Extend helper coverage to parameters like class_labels per the ADR spec.
Structured error payloads & surfacing	2	Although CalibratedError accepts details, no runtime path populates them, and the optional validate(condition, exc_cls, …)/explain_exception(e) helpers from the ADR are still absent. This limits machine-readable diagnostics and the envisioned debug-report integration.	Introduce a lightweight validate(...) helper that fills details (e.g., error codes, offending dimensions) and add an explain_exception utility that formats those details for logs/support tooling. Retrofit high-value validations (matrix checks, plugin metadata) to populate details.
Wrapper-level alias handling diverges from canonicalisation contract	1	canonicalize_kwargs explicitly preserves aliases while adding canonical keys, but _normalize_public_kwargs in WrapCalibratedExplainer drops alias keys after warning, so downstream code never sees the canonical duplicates that other callers expect. This is subtle today but risks behavioural drift once Phase 2 expands alias coverage.	Replace the wrapper’s bespoke normalisation with canonicalize_kwargs (possibly followed by a filter) so the behaviour matches ADR guidance everywhere. Document the preferred pattern for future entry points to avoid reintroducing divergence.
Logging/debug surfacing	1	The ADR encourages attaching validation summaries to debug reports, but no hook currently collects or surfaces them, and exceptions raised in prediction paths still rely solely on string messages.	Once structured details are wired, add an optional accumulator (e.g., on the wrapper) that records the latest validation failures for support logs or telemetry. This can ride alongside the explain_exception helper.
These items, ordered by severity, provide a roadmap to closing the remaining ADR-002 gaps while protecting the work already completed.

# ADR-003

## 1

Perform an in-depth gap analysis for ADR-003 and track any areas that need attention based on severity (using an appropriate metric).
Copy

Version 1

Version 2
Severity scale
High – Spec deviation risks incorrect behaviour or undermines ADR-003’s intent; should be addressed before shipping opt-in cache.

Medium – Important misalignment that weakens operability, observability, or rollout guidance but is unlikely to break correctness immediately.

Low – Minor polish gaps or documentation nits; defer until higher-severity items are resolved.

ADR-003 gap analysis
Severity	Area needing attention	ADR-003 expectation	Current state / gap	Evidence
High	Automatic invalidation & user flush	Version tags must change automatically with library/strategy revisions and provide an explicit flush hook.	CacheConfig exposes only a manually supplied version; nothing derives from __version__, no STRATEGY_REV registry, and the only reset is the fork-specific hook, leaving stale data after upgrades.
High	Cache coverage of required artifacts	Cache should avoid recomputing calibration summaries, predictions, and attribution tensors for repeated explain runs.	Only the _predict bridge is wired into CalibratorCache; no other stages register cache keys, so calibration summaries and explanation tensors still recompute every call.
Medium	Telemetry integration	Emit hit/miss/eviction metrics through the existing telemetry hook so staging can validate impact.	Cache counters exist, but the runtime telemetry payload never includes them, and the optional perf_telemetry callback isn’t bridged into CalibratedExplainer.runtime_telemetry, leaving operators without built-in observability.
Medium	Backend implementation fidelity	ADR mandates cachetools with Pympler sizing for the default backend.	Implementation is a bespoke OrderedDict LRU that intentionally avoids external dependencies, so we lose the proven eviction semantics and precise sizing ADR-003 assumed.
Medium	Release collateral & STRATEGY_REV discipline	Update README/plugins docs, add STRATEGY_REV appendix, and ship migration guidance.	How-to and plugin docs exist, but README lacks cache coverage, no STRATEGY_REV registry is present, and there’s no enterprise migration guidance, leaving rollout playbooks incomplete.
Additional observations
The opt-in posture, environment overrides, TTLs, and telemetry counters themselves align with ADR intent.

Unit tests exercise key normalization, TTL, memory limits, and telemetry callbacks, satisfying ADR testing guidance, but there is no smoke benchmark baked into CI yet (manual script lives under evaluation/scripts).

No automated tests were run for this analysis.

## 2

Severity scale
5 – Critical: Blocks ADR intent or risks incorrect/stale results in production.

4 – High: High-likelihood correctness or governance issue that must be addressed before broad rollout.

3 – Medium: Important for completeness or operator guidance; ship before GA but not an immediate blocker.

2 – Low: Nice-to-have alignment or polish; track but defer behind higher-severity items.

1 – Info: Observational notes without required action.

ADR-003 gap analysis
Requirement area	Status vs. ADR	Severity	Notes
Automatic invalidation (version_tag bumps, STRATEGY revisions, user flush)	Cache config exposes a static version string and forksafe_reset, but there is no linkage to package __version__, no STRATEGY revision registry, and no explicit flush API for operators, so cached payloads will persist across library upgrades contrary to the ADR’s invalidation mandate.	4	ADR calls for automatic bumps and a manual flush hook. The implementation keeps a fixed default (version="v1") and only exposes forksafe_reset() with no flush method or version automation.
STRATEGY_REV governance links	ADR directs maintainers to record STRATEGY revision identifiers and wire them into the release checklist, but the checklist currently lacks any STRATEGY_REV tasks or references.	3	Governance doc omits the STRATEGY discipline the ADR requires, leaving no procedural backstop for cache invalidation changes.
Cache coverage breadth (calibration, attribution artefacts)	Runtime currently wraps only _predict with the new cache, so calibration statistics and alternative explanation artefacts highlighted in the ADR remain uncached, limiting the performance win and cross-stage determinism.	3	ADR context targets calibration intervals and attribution tensors in addition to predictions. The implementation narrows usage to the prediction path (stage="predict").
Rollout documentation (README / namespace policy)	README and top-level entry points still lack cache configuration tables or namespace taxonomy guidance, so operators cannot discover the opt-in flow without diving into deeper docs, contrary to ADR rollout expectations.	3	ADR requires README/docs updates describing opt-in behaviour and namespace policy. The README’s onboarding sections omit any cache discussion or CE_CACHE reference.
Hashing & memory estimator parity with ADR spec	Implementation delivers stable hashing and byte budgeting, but it diverges from the specified blake2 + cachetools/pympler stack, falling back to sha256 and a constant 256-byte estimate for opaque objects. This may under/over-evict relative to ADR expectations.	2	ADR explicitly names blake2 and a cachetools+pympler combination. The code substitutes sha256 and a bespoke estimator with a static fallback.
Miss/eviction observability	Cache falls back to recomputation on misses and oversize entries but only emits telemetry callbacks; it never logs warnings as the ADR’s failure-mode guidance specifies, reducing operator visibility when the cache silently underperforms.	2	ADR asks for warnings and debug logs on misses/evictions. The current get/set paths simply return defaults or skip stores without any logging beyond optional telemetry events.
Additional observations
Core cache mechanics (namespaced keys, TTL, metrics counters, telemetry callback, env overrides) are implemented and exercised by unit tests, aligning with the ADR’s structural requirements.

Operator how-to documentation exists under the foundations guide, but until README and governance artefacts reference it, discovery remains limited.

Addressing the high-severity gaps (automatic invalidation and STRATEGY governance) should be prioritised before treating ADR-003 as fully satisfied; medium- and low-severity items can then close the remaining compliance delta.

# ADR-004

## 1

Approach
I reviewed ADR-004 to extract the required behaviors and compared them with the current implementation, documentation, and tests for the parallel execution stack. Evidence is cited for both the ADR expectations and the observed code/doc behavior.

Severity scale
Critical – blocks ADR intent or risks breaking core user flows.

High – major deviation that undermines promised capabilities or reliability.

Medium – important enhancement or safeguard missing, but core flow still works.

Low – polish, clarity, or follow-through gaps without significant risk.

Gap analysis
Area	ADR expectation	Current state	Severity	Notes
Strategy override & workload heuristics	ADR calls for user-supplied strategies, CE_PARALLEL “off” override, and workload-aware auto selection (task cost, data size, hints).	ParallelExecutor only switches on a string strategy; no hook for user objects and “auto” merely checks OS, CPU count, and joblib presence.	High	Heuristic gaps risk poor backend choice on CPU-bound vs I/O workloads.
Context management & cancellation	Unified API should expose map plus context manager semantics and graceful shutdown/cancellation.	Executor offers map but lacks __enter__, __exit__, or cancellation helpers; strategies rely on with internally.	Medium	Users cannot scope executor lifetime or cancel outstanding work as promised.
Configuration surface completeness	Config must expose max_workers, preferred, task_size_hint_bytes, force_serial_on_failure.	ParallelConfig includes max_workers, min_batch_size, and granularity but omits the additional knobs.	Medium	Missing fields prevent tuning heuristics and fallback policy per ADR.
Telemetry coverage	ADR expects timings, queue wait, worker utilisation, and aligned metric names flowing into shared telemetry.	Metrics track only counts (submitted, completed, fallbacks, failures); _emit just forwards events with no timing data.	High	Lacking telemetry blocks validation of performance impact and resource usage.
Resource guardrails & cgroup awareness	Should respect container quotas, avoid CI oversubscription, and expose CPU-detection safeguards.	Thread/process strategies simply scale by logical CPU count; no cgroup detection or CI guardrails.	High	Risk of runaway workers on constrained hosts or CI where ADR expected guardrails.
Cache compatibility lifecycle	After fork/spawn, cache must call forksafe_reset() to avoid stale state.	Process strategy resets the cache once before launching the pool; no worker initializer or post-fork hook.	Medium	Single pre-fork reset may miss child-process cache reuse after spawn.
Documentation rollout	Docs should include strategy matrix, env-var matrix, and troubleshooting tips per platform.	How-to guide explains enabling toggles but lacks platform-specific troubleshooting or matrix tables.	Medium	Users lack guidance for macOS/Windows spawn pitfalls highlighted in ADR.
Testing & benchmarking	ADR demands unit tests across strategies, telemetry, failure fallback, plus an integration benchmark.	Unit tests cover strategies, env overrides, and fallback behavior, but no benchmark or spawn lifecycle coverage exists.	Medium	Without integration benchmarking, it is hard to demonstrate ADR performance goals.
Additional observations
The fallback behavior specified by ADR (auto serial on failure with telemetry) is implemented and exercised in tests.

Documentation already instructs how to toggle CE_PARALLEL and highlights that the cache is reset when forking, aligning with part of the rollout guidance.

Recommended next steps
Expand ParallelConfig and ParallelExecutor to cover ADR-required heuristics, lifecycle management, and telemetry.

Implement cgroup-aware worker limits and CI safeguards.

Enhance documentation with troubleshooting matrices and provide integration benchmarks to validate gains.

No code changes were made.

## 2

Severity scale
Critical (5): Blocks ADR intent or risks incorrect behaviour by default.

Major (4): Substantially reduces ADR value or robustness in common scenarios.

Moderate (3): Noticeable deviation with limited scope or clear workarounds.

Minor (2): Cosmetic or low-risk divergence; address when convenient.

Informational (1): Already compliant or no action required.

Compliance snapshot
Most structural pieces of ADR-004 exist (executor facade, strategy stubs, cache reset hook), but workload-aware selection, rich telemetry, operational guardrails, and rollout collateral remain largely unimplemented. The highest-risk items concentrate around automatic backend choice, observability, and documentation expectations.

Gap detail
ADR expectation	Current state & evidence	Gap / risk	Severity
Workload-informed auto strategy: inspect task cost, data size, hints before choosing threads/processes/joblib (ADR-004 §Decision bullets 25–33).	_auto_strategy only checks OS, CPU count, and joblib availability; map ignores task hints and just compares to min_batch_size (no payload heuristics, no GIL hints).	Auto selection cannot distinguish CPU- vs IO-bound workloads, so users get suboptimal or hazardous backends unless they hard-code strategy.	Critical (5)
Allow explicit strategy objects and context-managed execution with cancellation semantics (ADR-004 §Decision lines 27 & 34).	ParallelExecutor only accepts a ParallelConfig; there is no way to inject a custom strategy object, nor __enter__/__exit__ context manager or cancellation hooks—only a bare map method and partial wrappers exist.	Limits extensibility promised by ADR and prevents safe lifetime management of pools (no deterministic shutdown beyond with inside strategies).	Major (4)
Configuration surface should expose task_size_hint_bytes and force_serial_on_failure toggles (ADR-004 §Decision line 36).	ParallelConfig only includes enabled, strategy, max_workers, min_batch_size, granularity, and telemetry callback; no size hint or force-serial flag is defined or parsed from CE_PARALLEL.	Lacking knobs to bias heuristics toward serial execution for large payloads or to permanently downgrade after errors.	Major (4)
Telemetry should capture task timings, queue wait, worker utilisation, fallbacks, and integrate with global hooks (ADR-004 §Decision & Operational clarifications lines 38 & 51).	ParallelMetrics only tracks counts (submitted, completed, fallbacks, failures) and ParallelExecutor._emit just forwards payloads; no timing or utilisation metrics are gathered, and no link to the broader telemetry contract beyond a raw callback.	Observability goals unmet; cannot validate gains or monitor regressions, undermining rollout safety.	Critical (5)
Graceful degradation should emit structured warnings on fallback (ADR-004 Operational clarifications line 48).	Fallback path increments counters and emits a telemetry event but never logs or warns; users get silent downgrades unless they register a callback.	Failures are invisible in default setups, making diagnosing backend issues difficult.	Major (4)
Resource guardrails: honour cgroup quotas, avoid CI oversubscription, provide heuristics (ADR-004 Operational clarifications line 50).	Worker counts default to CPU count (threads *5 cap 32) with no cgroup detection or CI safeguards; no environment toggles beyond explicit worker limit parsing.	Risk of overloading constrained environments, especially CI or containerized deployments.	Major (4)
Telemetry integration with cache metrics pipeline (ADR-004 Operational clarifications line 51).	Telemetry callback is shared with cache via PerfFactory, but metrics snapshot lacks worker_utilisation_pct and related fields; nothing forwards counts to cache telemetry structure beyond user-supplied callback.	Partial compliance; downstream consumers cannot ingest consistent payloads.	Moderate (3)
Testing: cover each strategy path, fork/spawn lifecycle, telemetry emission, failure fallback; provide integration benchmark demonstrating throughput improvement (ADR-004 Operational clarifications line 52).	Unit tests exercise strategy methods, cache reset, telemetry callback, and failure fallback with mocks (but no real fork/spawn). A micro-benchmark script exists but is manual, without assertions showing parallel win.	Need spawn-mode coverage and automated benchmark validation to satisfy ADR mandate.	Moderate (3)
Documentation: configuration docs, environment matrix, troubleshooting (macOS/Windows spawn), upgrade guide for plugin interactions, heuristics appendix (ADR-004 Documentation & rollout lines 56–58).	Current “tune runtime performance” guide explains enabling toggles and env var syntax but lacks platform-specific troubleshooting or heuristic appendix; no upgrade snippet covering plugin-provided executors. Release notes mention opt-in state but not detailed matrix.	Users lack guidance for spawn quirks, heuristics tuning, and plugin coexistence required by ADR.	Major (4)
API contract preservation (keep public signatures stable) and cache compatibility (ADR-004 Operational clarifications lines 43–49).	Wrap explainer API untouched and process strategy resets cache via forksafe_reset(), meeting compatibility requirement.	No action needed; validated through integration parity test.	Informational (1)
Recommended next actions (ordered by impact)
Implement workload-aware strategy selection leveraging task metadata and payload hints; wire ParallelConfig extensions to drive heuristics.

Expand telemetry capture (timings, utilisation) and surface structured warnings/logs on fallback while respecting verbosity controls.

Introduce context manager / cancellation support and accept injectable strategy objects to honour pluggability guarantees.

Add resource guardrails (cgroup-aware worker caps, CI defaults) and config toggles (force_serial_on_failure, size hints).

Broaden test matrix (spawn mode, joblib availability, telemetry assertions) and automate throughput benchmarks.

Update documentation with platform troubleshooting, env matrix, heuristic appendix, and plugin interoperability guidance.

These steps will close the critical gaps and align the implementation and collateral with ADR-004’s intent.

# ADR-005

## 1

ADR-005 commitments (for reference)
ADR-005 mandates a versioned explanation envelope containing schema_version, type, generator, meta, and a payload whose structure depends on the enumerated type list (feature_attribution, interval, global_importance, calibration_diagnostics). It also calls for a validate_explanation(obj) helper, JSON Schema drafts per type under schema/v1/*.json, and CI-backed validation using jsonschema.

Severity scale
High – Blocks ADR intent or creates breaking ambiguity for consumers.

Medium – Partial implementation that risks drift or misleads downstream tooling.

Low – Documentation or hygiene issues that do not immediately threaten interoperability.

Gap analysis
Requirement (ADR-005)	Current status in repo	Severity	Notes
Envelope with schema_version, type, generator, meta, nested payload	to_json emits flat payloads (task, prediction, rules, metadata, etc.) with an optional schema_version but no type, generator, meta, or nested payload wrapper.	High	Downstream consumers cannot rely on the envelope contract, undermining interoperability and versioning promises.
Enumerated type field & per-type payload schemas (schema/v1/*.json)	No code paths populate a type key, and only a single monolithic schema file (explanation_schema_v1.json) exists under schemas/.	High	Without typed payload schemas, new explanation kinds cannot evolve safely; consumers also lack the discriminant ADR calls for.
generator metadata with parameters_hash for provenance	Serialization output never includes generator or any hash; repository contains no implementation for parameters_hash.	High	Missing provenance makes reproducibility (a stated ADR goal) impossible to audit.
validate_explanation(obj) performing semantic checks + CI validation	Helper is named validate_payload and bails out silently when jsonschema is unavailable; tests skip validation if the dependency or schema load fails.	Medium	Validation is neither enforced nor aligned with the ADR’s semantics; optional skips weaken CI guarantees.
Schema documentation aligned with actual fields	Docs still describe rule_weight / rule_prediction, but the schema & serializer expose weight / prediction. They also point to an old test path (tests/test_serialization_and_quick.py).	Medium	Mismatched naming confuses integrators and suggests stale guidance; outdated references hinder discoverability of validation examples.
schema_version always present and semver-governed	Serializer allows callers to drop schema_version, and tests assert that omission is supported.	Medium	Optional version undermines the semver guardrail ADR-005 was meant to provide.
Published schema docs covering envelope & payload separation	Documentation focuses on the legacy flat payload; it does not mention envelope keys (type, generator, meta, nested payload).	Low	Documentation drifts from the architectural decision, increasing adoption risk once the envelope is implemented.
Additional observations
Golden fixtures and round-trip tests only exercise the legacy flat structure, so even after implementing the envelope, new tests/fixtures will be required to avoid regressions.

Plugin metadata today uses integer schema_version flags unrelated to ADR-005’s semver string, signaling broader alignment work once the envelope is restored (not yet classified above, but worth tracking when planning remediation).

Addressing the high-severity items (envelope structure, type discrimination, generator provenance) should be the top priority to unlock ADR-005 compliance; the medium- and low-severity issues can then be resolved to prevent future drift and ensure documentation/tests remain trustworthy.

## 2

Overview
Architectural Decision Record ADR-005 mandates a versioned JSON envelope for explanations, including envelope metadata, per-type payload schemas, and validation tooling. Current implementation and documentation diverge substantially from this contract, resulting in multiple high-severity compliance gaps that threaten downstream interoperability.

Severity Scale
Critical (C) – Breaks ADR’s core contract or blocks consumers.

High (H) – Major risk; requires prompt remediation.

Medium (M) – Important but not immediately blocking.

Low (L) – Minor inconsistency or documentation drift.

Gap Analysis
Area	ADR Expectation	Current State	Severity	Notes
Envelope structure	Top-level object must wrap schema_version, type, generator, meta, and a nested payload block.	Serializer emits flattened fields (task, prediction, rules, etc.) with optional schema_version; no envelope, type, generator, or meta fields.	C	Consumers built to ADR cannot parse current output; migration plan stalled.
Schema version handling	schema_version mandatory and independent from library version.	Field is optional (include_version=False path) and defaults to 1.0.0; schema does not require it.	H	Optional version undermines compatibility guarantees and evolution strategy.
Type registry	Enumerated type values to discriminate payload semantics.	No type property in serializer, schema, or tests.	C	Prevents multi-type tooling; contradicts ADR contract.
Generator provenance	Envelope must embed generator metadata with reproducibility hash.	No generator block emitted or validated.	H	Blocks traceability and reproducibility promised by ADR.
Meta vs. metadata	ADR specifies meta block for dataset info.	Implementation exposes metadata field (optional) without ADR-defined structure.	M	Naming mismatch and missing required fields hamper consumer alignment.
Payload partitioning	ADR expects nested payload for type-specific data.	Type-specific values placed at top level; schema/doc reinforce flat layout.	C	Breaks contract; no clear boundary between envelope metadata and payload.
Validation API	Provide validate_explanation(obj) covering structural and semantic checks.	Implementation exports validate_payload performing schema-only validation when jsonschema is available; no semantic rules.	H	API name/signature diverges; semantic invariants unimplemented.
Schema coverage	Ship JSON Schema drafts per explanation type.	Single generic schema file; no per-type definitions or directory structure described in ADR.	H	Cannot validate other explanation types once introduced.
CI fixtures & tests	CI should validate fixtures against schema to ensure round-trips.	Tests cover current flat structure and skip schema load test; no envelope fixtures or multi-type coverage.	M	Existing tests entrench non-compliant shape and miss ADR scenarios.
Documentation alignment	Docs in docs/schema/ should describe ADR envelope contract.	Documentation mirrors flattened payload, omits envelope fields, and renames keys (e.g., rule_weight).	H	External consumers will implement incompatible readers/writers.
Areas Needing Immediate Attention
Reintroduce ADR-compliant envelope with required metadata and nested payload, updating serializer, schema, tests, and docs accordingly.

Enforce schema_version and type semantics to regain forward/backward compatibility guarantees.

Implement generator/meta sections and per-type schema files to restore traceability and validation coverage.

Rename and extend validation API to validate_explanation (or provide alias) with semantic checks mandated by ADR.

Realign documentation and tests to the approved contract to avoid further drift.

No code changes or tests were executed during this analysis.

# ADR-006

## 1

Severity scale
High – Undermines ADR-006’s security or trust guarantees and can let unreviewed code run automatically.

Medium – Partially implements ADR intent but leaves notable risk or operational friction.

Low – Cosmetic or diagnostic gaps; ADR goals are met but developer/operator ergonomics suffer.

Current coverage highlights
Entry-point discovery defers to the calibrated_explanations.plugins group and skips untrusted plugins by default, warning operators to set CE_TRUST_PLUGIN or invoke trust_plugin as required by the ADR.

Metadata validation enforces the ADR’s required keys (schema_version, name, version, provider, capabilities) and accepts optional checksums while normalising trust flags.

Operators can snapshot registered plugins (optionally filtering for trusted ones) through list_plugins(include_untrusted=True), matching the diagnostics requirement.

Gaps requiring attention
Area	Gap summary	Evidence	Severity	Recommended next steps
Trust gating for third-party plugins	Any plugin that ships metadata with trusted=True is automatically trusted on registration, so a third-party package can bypass the explicit CE_TRUST_PLUGIN/trust_plugin gate that ADR-006 demands for external code.	High	Distinguish “built-in” trust (e.g., via an allowlist of package providers or signed metadata) from third-party metadata, and ignore/override trusted=True unless the package is explicitly trusted by the operator.
Denylist enforcement coverage	CE_DENY_PLUGIN is only consulted during explainer resolution and CLI display; registration and entry-point loading happily import and trust denied identifiers, so the variable cannot prevent execution or implicit trust when other call sites fetch plugins directly.	Medium	Apply is_identifier_denied (or an equivalent name-based check) during entry-point discovery and descriptor registration, short-circuiting before trust metadata is applied or the plugin is returned to callers. Emit an explicit warning when a denied plugin is skipped.
Diagnostics for skipped entry-point plugins	Because untrusted entry-point plugins are skipped entirely, list_plugins(include_untrusted=True) never surfaces their metadata for audit even though ADR-006 calls out diagnostics for untrusted entries.	Low	Persist minimal metadata for skipped entry-point plugins (e.g., in a “quarantined” store) so diagnostics can enumerate them without loading code, and extend the CLI to expose that quarantine with trust guidance.

## 2

ADR-006 coverage snapshot
Entry-point and manual registration paths exist. The registry targets the calibrated_explanations.plugins entry-point group and exposes both discovery helpers and direct registration APIs, matching the ADR’s discovery guidance.

Metadata validation enforces the required fields. validate_plugin_meta checks name, version, provider, capabilities, and optional checksum, ensuring registry state carries the ADR-mandated keys.

Trust gating is explicit and opt-in. Environment-driven trust (CE_TRUST_PLUGIN) and programmatic trust_plugin / untrust_plugin mirror the ADR trust flow, while _warn_untrusted_plugin skips untrusted discoveries unless explicitly allowed.

Checksum verification is wired in. When a plugin supplies checksum, _verify_plugin_checksum performs the best-effort SHA256 check envisioned by the ADR.

Gaps requiring attention
Severity: High – Untrusted entry points stay invisible to diagnostics. ADR-006 expects list_plugins(include_untrusted=True) to support auditing, but load_entrypoint_plugins drops every untrusted discovery before registration, so list_plugins can only enumerate plugins that were already accepted. Operators therefore cannot inspect metadata for blocked plugins without re-running discovery in permissive mode, undermining the diagnostic requirement and the trust guardrail.

Severity: Medium – “No sandbox” risk is undocumented for users. The ADR explicitly calls for documenting that plugins execute without isolation, yet the user-facing plugin guidance only lists trust toggles and CLI usage and never warns that discovery/import runs third-party code in-process. The sole warning lives in the internal register docstring, which typical users will not read, leaving the ADR’s documentation promise unmet.

# ADR-007

## 1

ADR-007 Scope (Expectations)
ADR-007 requires a JSON-serializable PlotSpec IR that records the plot’s kind, data, encodings, metadata, and schema version, plus infrastructure such as a renderer registry, per-kind validation, and a backend-selectable render(spec, backend="matplotlib") API.

It also calls for an extensibility story so plugins can register new kinds with default renderer fallbacks, while supporting JSON round-tripping for caching or remote rendering.

Current Implementation Snapshot
PlotSpec exists but is trimmed to title/header/body fields for bar-style panels; there is no explicit kind, encoding map, or version embedded in the dataclass.

A single matplotlib adapter (render function) is provided; it accepts either a dataclass or dict payload and directly renders (or short-circuits) without any backend selection argument.

Serialization helpers add a plotspec_version envelope (1.0.0) and can rebuild the minimal spec, but validation only checks for a body with bars and required bar fields.

The plotting pipeline converts probabilistic bar plots through the builder and adapter, partially realizing Phase C’s migration goal for at least one plot family.

Gap Assessment
Area	ADR Expectation	Current State	Gap	Severity*
PlotSpec structure	Dataclass should capture kind, encodings, data, meta, and version for general plots.	Dataclass only models bar/header panels; no kind, encoding descriptors, or embedded version.	Abstraction can’t generalize beyond current bar plots, blocking multi-backend support.	High (4/5)
Renderer registry & backend selection	Registry of adapters with render(spec, backend="matplotlib") API.	Single matplotlib adapter called directly from plotting code; no registry or backend parameter.	No extensible dispatch, preventing alternative backends.	High (4/5)
Validation by kind	validate_plotspec should enforce required fields per kind.	Validator only checks bar bodies and ignores kinds; header-only or other plot types unsupported.	Structural drift across new plot kinds would go undetected.	Medium-High (3/5)
Plugin extensibility	Plugins register new kinds with default renderer fallbacks.	Plugin layer defines plot builders/renderers but lacks registration hooks for PlotSpec kinds or default adapters; everything assumes matplotlib pipeline.	Ecosystem cannot extend PlotSpec semantics without core changes.	High (4/5)
Multi-backend coverage	Phase C then later add second backend (e.g., Plotly).	No alternative renderer implemented; matplotlib-only path remains.	Multi-backend milestone unmet; testing abstraction robustness impossible.	Medium-High (3/5)
JSON round-tripping	Ability to convert PlotSpec to/from JSON for caching/remote rendering.	Serialization helpers exist but only cover MVP bar schema; dict payloads for other kinds handled ad hoc in adapter.	Works for current plots but not future kinds; risk of divergent schemas.	Medium (2/5)
Headless export utilities	ADR highlights headless export (SVG/PNG bytes) as part of rendering API discussion.	Adapter only supports saving to disk via file paths; no byte-based or headless export abstraction.	Limits documentation/remote rendering scenarios; moderate impact.	Medium (2/5)
*Severity scale: 1 = Low (minor inconvenience), 3 = Medium/High (meaningful risk but workaround exists), 5 = Critical (blocks ADR outcomes). Scores reflect the impact of leaving the gap unresolved; likelihood is implicitly high because implementation paths currently miss these features.

Additional Observations
Phase C migration goals are partially met: probabilistic bar plots flow through builders and the adapter, but other plot families remain legacy-only, so broader adoption tracking is still needed.

The serializer’s narrow focus suggests schema evolution will be brittle without expanding coverage or providing schema definitions for non-bar plots.

Dict shims in the adapter for triangular/global plots indicate a stopgap approach; formalizing these specs would reduce technical debt before adding new backends.

Addressing the high-severity gaps (generalized PlotSpec schema, renderer registry, and plugin extensibility) should be prioritized to unlock ADR-007’s multi-backend vision and prevent the abstraction from calcifying around the initial matplotlib MVP.

## 2

Overview

ADR-007 establishes a backend-agnostic visualization layer centered on a PlotSpec intermediate representation, renderer registry, validation, and JSON serialization so that explanations can target multiple plotting backends.
The current codebase has delivered a working matplotlib pathway plus builders for several plot types, but several architectural promises remain unmet. Severity below follows a High / Medium / Low scale where High blocks ADR goals, Medium risks partial functionality or future extensibility, and Low tracks polish or documentation gaps.

Gap assessment

Area	ADR expectation	Observed gap	Severity	Notes
PlotSpec schema shape	Dataclass should expose kind, data, encodings, meta, and version so adapters can reason about plots uniformly.	The PlotSpec dataclass currently holds only title, figure_size, header, and body, omitting the structural fields the ADR calls out. That prevents adapters from inferring plot intent or version directly from the object.	High	Without schema parity, adding non-bar plots or new backends requires ad-hoc handling rather than a consistent contract.
Backend dispatch API	Provide render(spec, backend="matplotlib", **opts) for backend selection and headless export.	Only matplotlib_adapter.render exists; no module-level dispatcher accepts a backend argument, and the public viz namespace exports the adapter directly, tying callers to matplotlib internals.	High	Lacking a dispatcher blocks the “plug in plotly/headless” intent and forces downstream code to import backend-specific modules.
Renderer registry breadth	Maintain a registry of renderer adapters so multiple backends can be registered and resolved.	The plugin system registers a single PlotSpecDefaultRenderer, which simply forwards to the matplotlib adapter; the default builder also throws if the intent is outside a narrow set, leaving no path to register alternative renderers or declare fallback policies per kind.	High	Extending to additional backends requires refactoring rather than configuration/registration, undermining the ADR’s modularity goal.
Validation coverage	validate_plotspec should enforce required fields per kind.	Validation only checks bar bodies and requires a body, so triangular/global specs (which are represented as dicts) bypass or fail validation. There is no kind-aware branching to guard other payloads.	Medium	New adapters or plot kinds cannot rely on central validation, increasing the chance of runtime mismatches.
Serialized spec completeness	All specs should round-trip through JSON with explicit versioning for caching/remote rendering.	build_triangular_plotspec_dict and build_global_plotspec_dict emit JSON-like dicts but omit the plotspec_version envelope, so they are incompatible with plotspec_from_dict / validate_plotspec and cannot be versioned cleanly.	Medium	Headless export or schema evolution for non-bar plots will be fragile until these dict payloads align with the serializer contract.
What’s already in place

A stable PLOTSPEC_VERSION constant plus plotspec_to_dict / plotspec_from_dict functions exist for the dataclass-based specs, giving a foundation for versioned serialization.

Regression, triangular, and alternative plotting paths now build PlotSpec payloads (dataclass or dict) before handing off to the matplotlib adapter, satisfying the Phase C requirement to route core visuals through the abstraction.

Recommended focus

Extend the PlotSpec dataclass (or introduce a lightweight wrapper) so it exposes the ADR-required structural fields and embeds the version identifier.

Add a render(spec, backend=..., **opts) dispatcher that consults an explicit adapter registry, with the existing matplotlib adapter registered as the default.

Harmonize the serializer/validator contract so every builder (including triangular/global) emits the versioned envelope and validation becomes kind-aware.

Expand the plot plugin registry to allow registering additional adapters without modifying the matplotlib implementation.

These actions will close the remaining High/Medium gaps and unblock future backend work envisioned by ADR-007.

# ADR-008

## 1

ADR-008 scope recap
ADR-008 mandates an internal Explanation domain model with first-class FeatureRule objects carrying task/model metadata, calibration parameters, provenance, and rule-level semantics, while legacy APIs continue to expose the old dict shape through adapters until schema v1 fully replaces it.

Severity scale
Critical – Breaks ADR guarantees or blocks adoption entirely.

High – Major risk to correctness or migration; requires near-term remediation.

Medium – Important gap that could erode future work if unaddressed.

Low – Cosmetic or low-risk follow-up.

Gap analysis
Gap	Evidence	Impact	Severity	Recommendation
Domain model still isn’t the canonical in-memory representation. The pipeline continues to build and store FactualExplanation / AlternativeExplanation instances populated with legacy dict payloads; domain objects are only synthesized later when exporting to JSON via legacy_to_domain(...) adapters.	Keeps legacy dict logic as the source of truth, so ADR-008’s promised clarity, invariants, and downstream transformability are not realized. Future features must still traverse legacy structures, defeating the decision’s rationale.	Critical	Plan the remaining refactor so explainer workflows populate models.Explanation directly, and make adapters the boundary layer instead of the core representation.
domain_to_legacy cannot round-trip conjunctive rules. FeatureRule.feature deliberately accepts sequences for conjunctions, but domain_to_legacy blindly casts every feature to int, which will raise on lists/arrays produced by from_legacy_dict when conjunctions are present.	Any adapter round-trip on explanations containing conjunctions will crash, preventing exports and breaking ADR guarantees about adapter parity.	High	Update domain_to_legacy to emit list-valued features when fr.feature is non-scalar, and add regression tests covering conjunctive scenarios.
Required top-level metadata is still missing. ADR-008 calls for task, model info, calibration parameters, and provenance in the Explanation object, yet the dataclass only exposes task, index, prediction, rules, provenance, and a generic metadata map—no structured model/calibration fields.	Downstream consumers cannot reliably access calibration settings or model descriptors through the domain model, blocking schema-v1 alignment and hurting interoperability.	High	Introduce explicit fields (or well-defined sub-objects) for model descriptors and calibration parameters, and populate them during construction.
Golden-fixture parity tests are absent. ADR-008’s migration plan requires “adapter output matches golden fixtures byte-for-byte,” yet current tests only assert structural properties (key presence, lengths) without comparing against canonical fixtures.	Without fixture-based checks, regressions can slip through (e.g., changed ordering, missing keys), undermining the compatibility promise for legacy consumers.	Medium	Capture representative golden payloads and add snapshot/fixture assertions to guarantee byte-for-byte parity across adapter round-trips.
Overall assessment
Progress made: Domain dataclasses, adapters, JSON schema helpers, and basic round-trip tests exist, showing partial execution of Phase B/C deliverables.

Key blockers: The system still relies on legacy structures internally and suffers from adapter shortcomings; without resolving these, ADR-008 remains only partially implemented.

## 2

Severity scale
5 – Critical: Breaks ADR-required invariants or causes hard failures in common scenarios.

4 – High: Major semantic drift that risks consumers receiving incorrect paper-aligned data.

3 – Moderate: Important completeness gap; functionality works but ADR promises are unmet.

2 – Low: Minor deviation or future-hardening concern.

1 – Informational: Not currently actionable.

Gap analysis findings
Conjunctive rules cannot round-trip through the adapters (Severity 5).
The ADR explicitly calls for FeatureRule objects that can represent conjunctive conditions. legacy_to_domain already emits FeatureRule.feature as a list whenever it encounters a conjunctive legacy rule, but domain_to_legacy blindly casts every feature back to int, which raises a TypeError for lists and prevents JSON export for any explanation containing a conjunction. This is a hard failure that blocks ADR-008 compliance in realistic datasets that produce conjunctive rules.

Per-rule metadata required for paper semantics is dropped on domain → legacy conversion (Severity 4).
ADR‑008 mandates that each rule bind the observed feature value and expose calibrated weights with their uncertainty intervals. The domain model stores these details (feature_value, value_str, is_conjunctive, instance_prediction), and legacy_to_domain can read them from legacy payloads. However, domain_to_legacy only emits rule texts and integer features, omitting the observed values, display text, conjunctive flags, and instance-level predictions. Once an explanation flows through the domain model (e.g., during CalibratedExplanations.to_json), these semantics are irretrievably lost, so downstream consumers no longer see the paper-aligned bindings the ADR promised.

Top-level metadata fields promised by the ADR are missing from the domain model (Severity 3).
ADR‑008 specifies that Explanation should hold task, model information, calibration parameters, and provenance. The current dataclass only exposes task, index, prediction, rules, and optional generic provenance/metadata, with no structured slots for model info or calibration parameters. Likewise, _legacy_payload only feeds task, rules, feature weights/predictions, and predictions into the adapter, so the required calibration metadata never reaches the domain objects. Consumers of the new model therefore cannot reliably access the calibration context that the ADR committed to expose.

Adapter heuristics can silently duplicate mismatched metric arrays (Severity 3).
When feature_weights/feature_predict arrays are shorter than the rules list, _safe_pick reuses the last available value instead of flagging the mismatch. ADR‑008 stresses preserving calibrated prediction semantics, but this fallback masks data quality issues by attributing the wrong uncertainty interval or weight to later rules. The unit test already covers cases with missing support values, showing the function returns None, but any partially populated vector will lead to duplicated values rather than explicit gaps. That undermines confidence in the interval semantics the ADR highlighted.

Adapter parity tests do not meet the ADR’s “golden fixture” requirement (Severity 3).
The ADR requires byte-for-byte parity tests between adapters and golden fixtures. Existing tests only perform structural assertions (key presence, array lengths) and a basic round-trip check; they never compare against the golden JSON snapshot nor verify value equality. Without stronger parity coverage, regressions like the metadata loss described above can land unnoticed, leaving ADR-008’s migration guarantees unenforced.

Areas needing attention
P0 / Critical: Fix domain_to_legacy so conjunctive features (list-valued) export safely, preserving ADR invariants.

P1 / High: Extend domain-to-legacy conversions to emit full per-rule metadata (feature values, value strings, conjunctive flags, instance predictions) to maintain paper semantics.

P1 / Moderate: Enrich the domain model (and its adapters) with explicit model/calibration metadata fields so consumers receive the promised context.

P1 / Moderate: Replace _safe_pick’s silent duplication with explicit handling (e.g., None or validation errors) to avoid misreporting uncertainty intervals.

P1 / Moderate: Strengthen adapter tests to assert parity against the golden fixture values, not just shape, per ADR adoption guidance.

Addressing these gaps will align the implementation with ADR‑008’s guarantees and prevent downstream regressions in serialized explanation semantics.

# ADR-009

## 1

Severity Metric
Level	Description
Critical (P0)	Blocks ADR acceptance or leaves a headline requirement unimplemented.
High (P1)	Major functional or configurational gap; workarounds are partial or manual.
Medium (P2)	Behavioral drift or missing guardrails that can surprise users but have mitigation.
Low (P3)	Documentation/telemetry polish or incremental refinement.
Implemented ADR Coverage
User-supplied preprocessors can be injected via ExplainerConfig, are fit once during WrapCalibratedExplainer.fit, and are reused across calibration/inference.

Preprocessor telemetry snapshots (auto-encode flag, transformer identifier, mapping snapshot) are attached to the explainer and exposed through both batch telemetry and runtime_telemetry, with integration coverage ensuring parity.

Gap Analysis
Area	Gap Description	Severity	Evidence	Recommended Action
Automatic encoding path	ADR calls for a built-in automatic encoding option (auto_encode='auto') that persists mappings. Implementation only handles user-supplied preprocessors; there is no internal trigger for auto encoding or use of transform_to_numeric, and adoption notes still flag this as pending.	Critical (P0)	ADR requirement & adoption backlog; wrapper only fits provided preprocessors and otherwise no-ops	Implement the 'auto' path that instantiates the library’s encoder, persists mappings on the wrapper, and routes telemetry through _build_preprocessor_metadata.
Unseen-category policy	Configuration stores unseen_category_policy, but the wrapper never consults it during preprocessing or inference, leaving ADR-mandated behavior undefined.	High (P1)	ADR decision for unseen policy; wrapper only records the flag without applying it anywhere else.	Define enforcement points (e.g., in auto encoder transform) that raise/ignore unseen categories per config, and add tests exercising both 'error' and 'ignore'.
DataFrame/dtype validation	ADR expects validation to detect DataFrames with non-numeric columns and enforce NaN/dtype policies. Current validate_inputs_matrix simply converts to np.asarray and checks dimensionality/finite values, so categorical object columns slip through silently.	Medium (P2)	ADR validation mandate; validator implementation lacks dtype inspection beyond shape checks.	Extend validation to flag non-numeric dtypes when no preprocessor/auto-encoder is active, and surface actionable messages for DataFrame inputs.
Telemetry docs alignment	Documentation instructs readers to access preprocessor["identifier"], but the runtime payload actually exposes transformer_id, risking confusion for adopters.	Low (P3)	Doc snippet vs. metadata key emitted by wrapper.	Update documentation (and doctests if any) to reference transformer_id, or alias the field in the code for backward compatibility.
Recommended Next Steps
Build the auto-encoding pipeline with mapping persistence and policy-aware inference safeguards, then add integration tests mirroring the user-supplied transformer suite.

Wire unseen_category_policy into the preprocessing flow (both auto and custom paths) and document the runtime behavior.

Enhance validate_inputs_matrix (or a wrapper-specific guard) to detect categorical DataFrames early and prompt users to enable auto encoding or supply a transformer.

Align telemetry documentation/examples with the emitted field names to avoid instrumentation regressions.

These items cover the outstanding blockers for ADR-009 adoption and clarify remaining polish work before the ADR can be considered fully implemented.

## 2

Overview
ADR-009’s core objective—native preprocessing support with reproducible mappings—has only been partially realized. The current implementation successfully wires user-supplied preprocessors through the wrapper, captures telemetry snapshots, and exposes that provenance in explanation outputs. However, the ADR’s flagship “auto” encoding path and unseen-category safeguards remain unimplemented, and validation has not yet been upgraded to provide the DataFrame-aware diagnostics the ADR demands.

Severity scale
High – Prevents ADR-009’s promised behavior or leaves users without the documented guarantees.

Medium – Partially delivers the ADR but with notable usability or robustness gaps.

Low – Cosmetic or documentation cleanups that do not block the intended workflow.

Delivered coverage
ExplainerConfig already exposes the preprocessing knobs (preprocessor, auto_encode, unseen_category_policy) the ADR specified, and the builder forwards them for future wiring.

WrapCalibratedExplainer fits and reuses user-provided preprocessors across fit, calibrate, and inference phases, while emitting telemetry snapshots that flow through to runtime metadata and explanation containers.

Gaps & recommended follow-up
Area	ADR expectation	Current state	Gap	Severity
Auto-encoding pipeline	Wrapper should invoke either built-in transform_to_numeric or a supplied transformer; 'auto' mode must learn and persist mappings. 	Wrapper only runs a user-supplied preprocessor; comment explicitly states “No built-in auto encoding is activated here,” and there are no references to transform_to_numeric. 	Automatic encoding path—and its deterministic mapping persistence—has not been built, so the ADR’s marquee feature is unavailable.	High
Unseen-category policy	Configurable 'error'/'ignore' policy must be enforced when categorical encoders see new values. 	Policy is stored on the wrapper but never consulted; codebase only contains the assignment sites reported by ripgrep. 	Users cannot rely on the promised behavior for unseen categories, leaving ADR commitments unmet.	High
Auto-encode flag behavior	`auto_encode=True	False	'auto'` should control preprocessing behavior, not just telemetry. 	_auto_encode is normalised solely for metadata; no code path branches on it to alter preprocessing.
DataFrame-aware validation	Validation layer should detect DataFrames, flag non-numeric columns, and provide actionable messages. 	validate_inputs_matrix coerces inputs to NumPy and checks shapes/NaNs but does not inspect column dtypes or provide categorical guidance. 	Users still receive generic errors when passing DataFrames, undercutting ADR ergonomics goals.	Medium
Suggested remediation
Implement the 'auto' encoding workflow in WrapCalibratedExplainer: detect non-numeric columns, call transform_to_numeric, persist mappings on the wrapper, and include them in _build_preprocessor_metadata. Guard the path with unit and integration tests similar to the existing user-supplied preprocessor coverage.

Enforce unseen_category_policy during inference for both the auto-encode path and any helper wrappers; surface descriptive errors or silent ignores per configuration, and document the behavior.

Extend validate_inputs_matrix (or add a new validator) to recognize pandas objects, report which columns are non-numeric, and recommend enabling auto encoding—delivering the actionable guidance promised in ADR-009.

These fixes would close the highest-severity gaps and align the implementation with ADR-009’s accepted decision.

# ADR-010

## 1

Severity scale
Critical (5): Blocks ADR-010 goals or contradicts the decision outright.

High (4): Major usability/maintainability issue that undermines the split without immediate breakage.

Medium (3): Important documentation or developer-experience gap that can confuse adopters but has workarounds.

Low (2): Minor polish or backlog-quality follow-up.

Gap analysis against ADR-010
Area	Gap description	Severity
Core dependency trim	The core package still installs ipython, lime, and matplotlib by default instead of confining them to extras, directly contradicting ADR-010’s mandate to keep the runtime minimal and move those packages behind optional groups.	Critical (5)
Evaluation extra contents	Evaluation scripts import xgboost, lime, and shap, yet the [eval] extra only installs shap and seaborn, leaving out core requirements and forcing manual dependency management despite ADR-010’s call for a finalized evaluation bundle.	High (4)
Automated viz test skips	Tests marked @pytest.mark.viz call plotting routines that require matplotlib, but the suite lacks a skip-on-missing-extras hook—users must remember to run pytest -k "not viz" manually, which falls short of ADR-010’s directive to “conditionally skip when extras are absent.”	High (4)
Contributor guidance	CONTRIBUTING never explains the core-vs-evaluation split or how extras should be used, leaving maintainers without the documentation ADR-010 requires alongside README updates.	Medium (3)
Evaluation environment file	ADR-010 asks for both a README and an environment lock; only the README exists—no environment.yml or requirements file is present in evaluation/, so reproducing runs still requires guesswork.	Medium (3)
Extras documentation accuracy	README and researcher docs claim [eval] installs xgboost and plotting libraries, which is currently untrue; this inconsistency will mislead adopters once the extras are expected to be authoritative.	Medium (3)
Progress and healthy areas
Optional dependency groups (viz, notebooks, dev, eval) are already defined in pyproject.toml, satisfying the structural part of Adoption Step 1.

Plotting modules perform guarded imports and emit actionable errors when matplotlib is missing, fulfilling the runtime behavior ADR-010 calls for.

An evaluation/README.md documents the scope and entry points, covering half of Adoption Step 5.

Recommended focus
Immediately reprioritize the dependency list to remove optional tooling from project.dependencies and rely on the extras instead (Critical).

Expand the [eval] extra to match the actual imports in evaluation code (xgboost, lime, plotting stack, etc.), and fix documentation claims in lockstep (High/Medium).

Add automated viz skips—for example, register a pytest_collection_modifyitems hook that adds pytest.skip when matplotlib cannot be imported—so minimal environments do not need custom CLI flags (High).

Author a contributor-facing section explaining the split, extras, and CI expectations, and ship an evaluation environment file (environment.yml or requirements-eval.txt) to finish Adoption Step 5 (Medium).

## 2

ADR-010 Gap Analysis Summary
Severity scale used below: Critical (blocks ADR objective), High (major risk to objective), Medium (noticeable drift but workarounds exist), Low (minor misalignment).

Area	ADR expectation	Current state	Severity	Recommended action
Core dependency footprint	Move ipython, lime, matplotlib, etc. out of the default runtime and into extras so the core install stays lean.	pyproject.toml still lists ipython, lime, and matplotlib as mandatory dependencies; requirements.txt mirrors the heavier stack, so a default install still pulls visualization tooling.	Critical	Promote plotting/notebook tools to the documented extras and keep the base dependency list to the minimal runtime set before cutting a release.
Optional extras completeness	Define viz, notebooks, dev, and eval groups with representative package lists.	Extras exist, but the [eval] set only includes shap/seaborn while evaluation scripts import xgboost, lime, and other tools; README claims the extra installs xgboost and venn-abers, creating a mismatch.	High	Audit evaluation scripts, align the [eval] extras with actual imports, and update the README text so users installing extras get a working environment.
Visualization test isolation	Tests that require plotting must be marked and skipped when the viz extra is absent.	Many integration tests marked @pytest.mark.viz still call .plot(...), but there is no collection-time skip; a core-only install will fail unless users remember -k "not viz" manually.	High	Add a pytest_collection_modifyitems hook (or fixture) that checks for matplotlib and skips viz/viz_render cases automatically when the extra is missing.
Evaluation environment docs	Provide an evaluation/README.md plus an environment specification for reproducibility.	README exists and explains the split, but there is no pinned environment.yml/requirements file in the folder.	Medium	Add a lockfile (e.g., evaluation/environment.yml) that references the [eval] extra and any remaining packages so researchers have a reproducible starting point.
CI coverage for extras	Optional workflow to exercise evaluation stack separately from core CI.	CI runs a “core-only” job that filters out viz tests, but there is no dedicated evaluation workflow that installs [eval] and validates the research harness.	Medium	Introduce an evaluation workflow (even on a reduced schedule) that installs the [eval] extra and smoke-tests representative scripts/notebooks.
Contributor documentation	Document the core vs. evaluation split and ensure quickstart paths work without viz.	README includes install commands for viz/dev, but CONTRIBUTING lacks explicit guidance; quickstart instructions still rely on the heavy default dependency set noted above.	Low	After trimming the default deps, add a CONTRIBUTING note summarizing when to install extras and how CI enforces the split.
Additional Observations
The evaluation README already reiterates that the folder is optional, which aligns with ADR intent; tightening the dependency story will make it actionable.

The existing “core-only” CI job is a good safety net, but it compensates for packaging drift by using pip install . --no-deps and manual dependency selection; once the default dependency list is slimmed down, this workaround can be simplified.

# ADR-011

## 1

ADR-011 Gap Analysis
Severity scale
Critical – Direct ADR requirement missing, creating an immediate policy violation.

High – Requirement partially implemented; risk of policy drift without remediation.

Medium – Supporting control missing; monitor but not yet blocking.

Compliant – Requirement fully addressed; keep under routine maintenance.

Findings
Requirement (ADR-011)	Current state & gap	Severity	Evidence
Central deprecate(msg, *, once_key) helper ensures one warning per session and applies across public surfaces.	All public boundaries call warnings.warn directly (e.g., alias handling in api/params.py, module shim in core/__init__.py, runtime entry points in core/calibrated_explainer.py), so warnings repeat per call/site and there is no shared keying mechanism.	High	see evidence
Maintain a migration guide with side-by-side examples and align CHANGELOG messaging accordingly.	The current docs tree lacks any migration-focused page, yet the v0.6.0 changelog points users to a non-existent guide, leaving adopters without official instructions.	Critical	see evidence
Track deprecation/migration status in RELEASE_PLAN_v1.md via a status table.	RELEASE_PLAN_v1.md only carries narrative milestones; no structured status table exists to record deprecation windows, so coverage across releases is opaque.	High	see evidence
Enforce “two minor releases before removal” and publish migration notes per release via CI policy gates.	CI workflows run linting, tests, and perf checks but do not include automated deprecation window enforcement or migration-note validation, leaving removal timing unchecked.	High	see evidence
Provide an API snapshot diff tool and guard against accidental surface changes.	Scripted export plus an integration test keep the public API snapshot under source control; this portion of ADR-011 is satisfied.	Compliant	see evidence
Offer optional rewrite helpers to ease alias migrations.	scripts/migrate_aliases.py scans projects and suggests replacements, meeting the “optional rewrite helper” goal.	Compliant	see evidence
Recommended follow-ups
Introduce a shared deprecation helper: add a deprecations.py utility (for example under src/calibrated_explanations/utils/) that memoizes once_key emissions and replace direct warnings.warn calls in core/__init__.py, api/params.py, core/calibrated_explainer.py, and legacy shims with the new helper.

Author and link a migration guide: place a dedicated page (e.g., docs/appendices/migration_0_6_to_0_8.md) capturing alias mappings, module move guidance, and before/after snippets; update CHANGELOG.md and landing pages to point to it.

Add a deprecation status table: embed a structured markdown table near the top of improvement_docs/RELEASE_PLAN_v1.md enumerating feature/symbol, first warning release, earliest removal, and owner so timelines are auditable.

Bake policy checks into CI: extend .github/workflows/lint.yml or test.yml with a lightweight script that parses the status table to ensure no symbol scheduled for removal violates the “two minor release” promise and that each flagged removal has accompanying migration notes.

Addressing the high- and critical-severity gaps will bring the project into alignment with ADR-011’s deprecation governance expectations.

## 2

Severity scale
High – Blocks ADR-011 compliance or risks breaking public contracts in the next release.

Medium – Material erosion of the policy (user-facing noise, governance gaps) but short-term mitigations exist.

Low – Follow-up hygiene or documentation needed for completeness.

Gap analysis for ADR-011
Area	Gap vs. ADR-011	Evidence	Impact	Severity	Recommended action
Centralized warning helper & once-per-session semantics	ADR-011 requires a deprecate(msg, *, once_key) helper that emits once per session per symbol. Current deprecation paths call warnings.warn directly (e.g., parameter aliases, plugin metadata, deprecated methods, legacy module import) with no shared helper or keying, so repeated use spams warnings instead of collapsing to one per session.	Users receive duplicate warnings, contradicting the “predictable policy” promise and the v0.6.0 release note that already advertises once-per-session behavior.	High	Implement the shared deprecate() helper with an in-memory once-per-key cache, replace direct warnings.warn calls, and add tests that assert repeated invocations do not issue duplicate warnings.
Policy timeline enforcement	ADR-011 mandates a two-minor-release window before removals and CI gates to enforce it. The repo has no metadata or automation that tracks introduction/removal windows; the release plan focuses on other milestones without any deprecation status table or gating logic.	Risk of premature removals or missed cleanups because nothing encodes “introduced in X, earliest removal Y,” and CI cannot block violations.	High	Add a structured deprecation registry (YAML/JSON) that records intro/removal versions, wire a CI check that fails if a removal PR violates the window, and surface the table in RELEASE_PLAN_v1.md.
Migration guide & status table	ADR-011 calls for a migration guide with side-by-side examples and a status table in the release plan. There is no migration guide in the docs tree, and the release plan lacks any tabular deprecation status tracking.	Users cannot find authoritative upgrade instructions, and maintainers lack a single view of active vs. retired deprecations.	High	Author docs/migration/index.md (or equivalent) with before/after snippets, link it from README/CHANGELOG, and embed a maintained status table in the release plan.
Tooling for API diffs & rewrite helpers	ADR-011 expects an API snapshot diff tool and optional rewrite helpers. The repo only contains a snapshot script that dumps symbols; it does not compute diffs or integrate with CI, and there are no rewrite helpers for notebooks/scripts.	Without automated diffs, regressions can slip in unnoticed; lack of rewrite helpers makes migrations manual for users.	Medium	Extend the snapshot tooling to compare against the latest baseline and fail CI on unexpected changes; add simple rewrite utilities (e.g., CLI script) for alias renames.
Warning policy vs. strict-mode toggle	ADR-011’s open question proposes a CE_DEPRECATIONS=error switch. No such environment variable exists; instead, pytest unconditionally upgrades DeprecationWarning to errors, with ad hoc ignores.	Contributors cannot exercise the “strict opt-in” mode outside pytest, and current filters risk hiding legitimate deprecations while running the test suite.	Medium	Implement the env-variable override inside the new deprecate() helper and relax pytest’s blanket error::DeprecationWarning in favor of the helper-based control.
Communication accuracy	The changelog advertises “warnings emitted once per session… See migration guide,” but neither behavior nor documentation exists yet.	Mismatch between promises and implementation erodes trust in release notes.	Low	Once the helper and guide exist, update the changelog entry (or append errata) to reflect the actual shipment status.
Additional observations
Tests currently assert only that a warning occurs, not that duplicate calls stay silent, so new once-per-session behavior will need fresh coverage.

The proposed CE_DEPRECATIONS strict mode is entirely unimplemented beyond the ADR note.

No automated tests or commands were run for this analysis.

# ADR-012

## 1

Overview
ADR-012 sets the documentation pipeline expectations around Sphinx quality gates, gallery/example execution, dependency management through packaging extras, and artifact publishing. The current repository satisfies parts of the decision (HTML builds with -W, link checking, artifact upload, navigation smoke tests), but several high-impact gaps remain.

Severity scale
Score	Meaning
4	Critical break with ADR intent; blockers that undermine the policy’s guarantees.
3	High risk or drift; requirement is partially met but leaves room for silent regressions.
2	Moderate concern; requirement is met but lacks hardening or is fragile.
1	Low concern; cosmetic or future-cleanup item.
Gap analysis
#	Requirement (ADR-012)	Current state & evidence	Severity	Impact	Recommended action
1	“Render notebooks via sphinx-gallery or nbconvert (MVP acceptable) into HTML; failures fail CI.”	The docs workflow builds HTML and runs linkcheck/tests, but it never runs sphinx-gallery/nbconvert or even stages notebooks under docs/. CI therefore cannot fail on broken notebooks. find docs -name '*.ipynb' returns nothing, and the quickstart hub still hyperlinks to raw GitHub .ipynb files instead of shipping rendered pages.	4	Broken or outdated notebooks will ship unnoticed, violating the core guarantee of ADR-012 and forcing users back to GitHub to run notebooks manually.	Move or symlink the vetted notebooks into the documentation tree, add a sphinx-gallery or nbconvert --execute step to .github/workflows/docs.yml, and make the build fail on execution errors before uploading artifacts.
2	“Gallery jobs install [viz] and [notebooks] extras.”	CI installs docs/requirements-doc.txt and pip install ., but never exercises the packaging extras declared in pyproject.toml. The doc-specific requirements duplicate the dependencies instead of pulling them through the extras ADR-012 named.	3	Extras can silently drift from the doc build environment, so users installing calibrated-explanations[viz,notebooks] may hit missing pins or incompatibilities that CI never sees.	Replace the manual dependency list with pip install ".[viz,notebooks]" (keeping only doc-only Sphinx pins separate) so the workflow verifies the extras stay healthy.
3	“Examples must be runnable headlessly with seeded randomness and light datasets (<30 s per example on CI hardware).”	The doc tests cover only two Python quickstart flows; none of the notebooks run in CI, so there’s no automated proof that every checked-in example remains headless-friendly or within the runtime budget.	3	Heavy or unseeded notebooks could regress without warning, breaking ADR-012’s contribution rule and eroding contributor confidence.	Once notebook execution is wired into CI (Gap #1), enforce per-notebook execution and optionally capture timing to flag >30 s runs, ensuring future contributions respect the headless/runtime contract.
Compliant elements
Sphinx HTML builds already run with -W, satisfying the “warnings are errors” requirement.

Link checking and documentation smoke tests (pytest tests/docs) are wired into the same workflow, and artifacts are uploaded for review as ADR-012 prescribes.

Addressing the highlighted gaps will bring ADR-012 fully to life: notebooks will render automatically, packaging extras stay honest, and CI will enforce the headless/runtime promises that the ADR made to contributors.

## 2

Overview
ADR-012 (“Documentation & Gallery Build Policy”) establishes CI gates, example-rendering expectations, dependency handling, contribution rules, and publishing requirements for the documentation system. The current repository satisfies several pillars (e.g., blocking docs CI, artifact uploads, doc smoke tests), but key deliverables remain open. The severity scale used below is:

Severity 0 – On track: No action needed.

Severity 1 – Low: Improvement is nice-to-have.

Severity 2 – Medium: Action needed but not immediately blocking.

Severity 3 – High: Action required soon; risk to ADR goals.

Severity 4 – Critical: Immediate attention required.

Confirmed compliance (Severity 0)
Blocking docs CI with -W and linkcheck. The docs workflow builds HTML with warnings treated as errors, runs linkcheck, executes pytest tests/docs, and uploads the HTML artifact, aligning with ADR gate and publishing requirements.

Release governance mirrors ADR-012. The release checklist keeps ADR-012 front and center and explicitly blocks releases on the same CI gates, ensuring the policy remains enforceable during cutovers.

Quickstart smoke coverage. Documentation tests exercise classification and regression quickstarts with seeded randomness and small datasets, demonstrating headless execution of the primary examples.

Gaps & action items
Requirement	Gap & Evidence	Severity	Recommended Next Steps
Example rendering via gallery/nbconvert	ADR-012 mandates rendering notebooks to HTML with CI failures on breakage. The docs workflow only builds standard Sphinx HTML and linkcheck; there is no gallery/nbconvert stage. Docs still link to raw GitHub notebooks instead of published HTML renders.	Severity 3	Introduce an example-rendering job (e.g., nbconvert or sphinx-gallery) that executes notebooks, publishes HTML into docs/_build, and fails on errors. Wire it into CI alongside the existing gates.
Optional extras for visualization/notebooks	ADR-012 requires keeping visualization backends as optional extras and installing [viz]/[notebooks] in gallery jobs. Currently, matplotlib, ipython, and lime are part of the core dependency list rather than optional extras, defeating the opt-in contract. The docs requirements file also hard-pins these libs irrespective of extras.	Severity 3	Rescope pyproject.toml so visualization and notebook tooling live only under extras, and adjust docs CI to pip install .[viz,notebooks] (or equivalent) when the gallery job runs. Update requirements-doc accordingly.
Enforcing example runtime ceiling (<30 s)	ADR-012 sets an expectation that examples run under 30 s headlessly. Current smoke tests assert telemetry correctness but do not measure duration, so regressions in runtime would go unnoticed.	Severity 2	Add timing assertions or monitoring (e.g., pytest markers with duration thresholds or CI telemetry) to fail when an example exceeds the 30 s budget.
Gallery technology decision	ADR-012’s open question about choosing sphinx-gallery vs. nbconvert for the MVP remains unresolved in the documentation set.	Severity 1	Record the chosen renderer in docs (e.g., within docs/foundations/governance/test_policy or a maintainer note) and reflect it in the workflow once implemented.
Addressing the high-severity items—example rendering and dependency optionality—will unblock full ADR-012 compliance, while the medium/low items tighten enforcement and institutionalize remaining decisions.

# ADR-013

## 1

Severity scale
High – Blocks ADR-013 guarantees or materially weakens the guard-rails it mandates.

Medium – Important misalignment that risks user confusion or partial loss of guarantees, but with existing mitigations.

Low – Non-blocking divergence (documentation, tooling, or naming) that still merits follow-up.

Gaps against ADR-013
High – Runtime protocol enforcement and output validation are missing.
The ADR requires the runtime to assert that a resolved calibrator actually satisfies the classification/regression protocols and to validate probability/interval outputs for shape, dtype, and monotonicity before calibration proceeds.
In the current implementation, _obtain_interval_calibrator simply calls plugin.create(...) and stores the result without checking it against ClassificationIntervalCalibrator / RegressionIntervalCalibrator or validating its predictions.
That leaves the system dependent on later call sites to discover protocol violations, undermining the guard-rail intent of the ADR.

Medium – LegacyIntervalContext adaptor and read-only guarantees are absent.
ADR-013 calls for a dedicated adaptor that presents frozen, read-only calibration artefacts so plugins cannot mutate explainer state, and it expects the default plugin to “simply store the objects it receives and return them during create().”
The current context builder returns a regular dict for metadata, which is immediately mutated both in the explainer (_capture_interval_calibrators) and by plugins (LegacyIntervalCalibratorPlugin.create), so the “read-only” contract is not enforced.
This keeps legacy behaviour but fails the ADR goal of isolating plugin-side mutations.

Low – Default plugin still re-instantiates calibrators instead of returning frozen instances.
The ADR’s “Legacy default” clause expects the plugin to hand back the already-prepared IntervalRegressor/VennAbers instances from the explainer, maintaining continuity without extra instantiation.
LegacyIntervalCalibratorPlugin.create rebuilds fresh calibrators every time using the raw calibration splits, which works but diverges from the intended “return the frozen instance” design and prevents sharing of cached state that future work might rely on.

Low – CLI coverage for interval validation is incomplete.
ADR-013 specifies a ce.plugins explain-interval --plugin <id> command (alongside list --intervals) to surface availability and validation results from the runtime checks.
The current CLI only offers list, show, trust, and untrust subcommands; there is no interval explanation command to surface validation outcomes once they are implemented.

These items outline the main deltas between ADR-013 and the current codebase, ordered by their expected impact on the guarantees ADR-013 is meant to deliver.

## 2

Overview
An assessment of ADR-013 (“Interval Calibrator Plugin Strategy”) shows the core scaffolding is in place—protocols, registry wiring, and configuration hooks exist—but several contract-breaking behaviours remain. The findings below group gaps by severity, using a three-tier scale:

Critical – Violates an explicit ADR contract and risks incorrect runtime behaviour.

High – Materially diverges from ADR intent, likely to cause maintenance or safety issues.

Medium – Partial implementation that needs follow-up but is less urgent.

Critical gaps
Requirement	Observation	Impact
FAST plugins must return calibrators that satisfy ClassificationIntervalCalibrator	FastIntervalCalibratorPlugin.create returns a mutable list of heterogeneous calibrators, not a single object implementing the protocol. Downstream logic compensates by special-casing sequences, signalling the protocol was never met.	Breaks the ADR-013 contract that create returns a protocol-compliant calibrator. Callers expecting .predict_proba etc. on the returned object will fail, so this is Critical.
High-priority gaps
Requirement	Observation	Impact
Interval context must be a read-only view	_build_interval_context injects plain dictionaries into the frozen dataclass, and both the explainer and default plugin mutate those mappings (e.g., context.metadata.setdefault(...)).	Violates the ADR guarantee that plugins receive an immutable view. This allows accidental cross-plugin state leaks and contradicts the “read-only” contract, so it is High severity.
Medium-priority gaps
Requirement	Observation	Impact
Default plugin identity and behaviour	ADR-013 calls for a DefaultIntervalCalibratorPlugin that simply exposes pre-existing calibrators. The implementation ships LegacyIntervalCalibratorPlugin, which reinstantiates VennAbers/IntervalRegressor each time and writes back to metadata. While behaviour matches legacy flows, the ADR’s emphasis on returning already-prepared instances isn’t met.	Functionally equivalent but diverges from the documented pattern, increasing drift between spec and code. Medium severity because behaviour is mostly preserved but maintainers may misinterpret the contract.
Recommended follow-ups
FAST contract fix (Critical)
Ensure FastIntervalCalibratorPlugin.create returns an object implementing ClassificationIntervalCalibrator/RegressionIntervalCalibrator. Wrapping the current list behaviour behind a façade that exposes the required methods would satisfy ADR-013 without regressing FAST features.

Make interval context structurally immutable (High)
When building IntervalCalibratorContext, wrap dict-like payloads in MappingProxyType (or equivalent) and stop mutating them later. Adjust LegacyIntervalCalibratorPlugin and _capture_interval_calibrators to return enriched metadata instead of mutating the context in place.

Align default plugin semantics with ADR naming (Medium)
Rename or supplement LegacyIntervalCalibratorPlugin with an explicit DefaultIntervalCalibratorPlugin that simply returns the already-instantiated calibrator stored on the explainer, matching the ADR language and reducing confusion.

No automated tests were run for this analysis.

# ADR-014

## 1

Severity Scale
5 – Critical: Blocks ADR intent or causes incorrect behaviour.

4 – High: Major feature gap that undermines key ADR outcomes.

3 – Medium: Important but non-blocking gap that complicates adoption or safety.

2 – Low: Missing polish/supporting feature; functionality mostly intact.

1 – Informational: Documentation or nice-to-have follow-up.

Implemented Elements (for context)
The core plot protocols (PlotRenderContext, PlotBuilder, PlotRenderer, PlotRenderResult) are defined, giving plugins the expected type surface.

Registry support exists for registering builders, renderers, and styles, and the plotting entry point iterates the resolved style chain via the registry.

Gaps Requiring Attention
ADR Requirement & Gap	Evidence	Severity	Notes / Suggested Focus
Legacy builder/renderer must wrap the legacy plotting helpers so fallback preserves behaviour. Current LegacyPlotBuilder just returns the context and LegacyPlotRenderer returns an empty result without calling the legacy plotting helpers, so any fallback to the legacy plugin produces no figure.	ADR requirement vs. implementation and usage in .plot() fallback chain	5 – Critical	Wire the legacy plugin through `legacy.plotting` (or equivalent) before returning, ensuring parity when fallbacks occur.
Resolver must honour renderer overrides (renderer kwarg, CE_PLOT_RENDERER, project config). _resolve_plot_style_chain and the surrounding .plot() logic only inspect styles; renderer selection from kwargs/env/config is ignored.	ADR requirement; resolver implementation showing style-only logic; no references to CE_PLOT_RENDERER in code	4 – High	Extend resolution to track both style and renderer choices per ADR; ensure kwargs/env/project defaults can select a renderer independently.
Helper base classes in viz/plugins.py are missing. The ADR expects reusable scaffolding for plot plugins, but the viz package has no plugins.py, forcing plugin authors to re-implement hooks.	Requirement vs. package layout lacking plugins.py	3 – Medium	Introduce the promised helper module (lifecycle hooks, validation helpers) to reduce duplication and align with ADR guidance.
validate_plotspec should live in viz/plotspec.py and be invoked before rendering. Validation currently sits in viz/serializers.py, and the default renderer calls the matplotlib adapter without validating the artefact.	Requirement; actual location; renderer omission	3 – Medium	Move or re-export validation in viz/plotspec.py and call it from PlotSpecDefaultRenderer before dispatching to the adapter.
CLI tooling incomplete (validate-plot, set-default --plot-style). The ADR calls for additional CLI helpers, but the current CLI only supports listing/trust commands; searches for the new subcommands return nothing.	Requirement; CLI implementation excerpt without those commands; missing command searches	2 – Low	Add the missing CLI subcommands to cover validation and default-style management, aligning ops tooling with the ADR.
Addressing the critical/high items will restore ADR conformance for runtime behaviour, while medium/low items close the remaining developer-experience and tooling promises.

## 2

Severity scale
Level	Meaning
Critical	Breaks the architecture promised by ADR-014 or causes incorrect/no plot output for expected user flows. Requires immediate remediation.
High	Major functional or contract gap that blocks third-party plugin authors or prevents configuration paths described in the ADR.
Medium	Missing robustness, tooling, or validation that increases risk but has workarounds.
Low	Documentation or developer-experience gaps with limited runtime impact.
Current coverage snapshot
The project already exposes PlotRenderContext, PlotBuilder, PlotRenderer, and PlotRenderResult, matching the ADR’s protocol requirements.

Explanation contexts now capture plot fallbacks derived from plugin metadata, so explanation plugins can hint preferred styles.

Plot styles, builders, and renderers are registered through the new registry APIs with default legacy and plot_spec.default entries.

Gap analysis
ADR-014 requirement	Current state	Severity	Notes / Risk
Provide helper base classes in viz/plugins.py for common lifecycle/validation hooks.	The viz package contains no plugins.py helpers—only builders, adapters, etc. are present.	High	Third-party authors must re-implement boilerplate and validation, increasing inconsistency risk and violating the ADR mandate for shared tooling.
Legacy builder/renderer must wrap the legacy plotting helpers so fallback styles actually render.	LegacyPlotBuilder.build just returns the context and the renderer returns an empty result; no legacy drawing is triggered.	Critical	When non-legacy styles fall back to the legacy style (the default safety net) nothing is rendered, breaking .plot() parity that ADR-014 considers mandatory.
Builder metadata must include default_renderer along with other required fields.	Metadata validation enforces style, output_formats, dependencies, etc., but there is no default_renderer key or enforcement.	High	Without a declared default renderer, tooling cannot automatically chain builders to their preferred renderer as the ADR specifies.
.plot() resolution order should consider explicit renderer overrides and CE_PLOT_RENDERER, falling back through configured chains.	Resolution logic only tracks styles (CE_PLOT_STYLE, pyproject style, fallbacks) and never inspects renderer overrides.	High	Users cannot select alternate renderers even when multiple renderer options exist, limiting extensibility promised by ADR-014.
Registry should raise dedicated PlotPluginError when resolution fails.	No PlotPluginError type exists; resolution returns None, leading to aggregated strings and a generic ConfigurationError later.	Medium	Error handling diverges from ADR guidance, producing less actionable diagnostics and preventing consumers from catching plot-specific failures.
Default renderer must validate PlotSpec payloads before drawing.	PlotSpecDefaultRenderer.render passes artefacts directly to matplotlib_adapter.render without calling validate_plotspec.	Medium	Invalid PlotSpecs can reach the renderer unchecked, counter to ADR intent and increasing the chance of backend crashes.
CLI helpers (ce.plugins list --plots, validate-plot, set-default) should exist.	Repository search finds no implementation references outside the ADR itself.	Medium	Without CLI tooling, administrators lack the observability and configuration surface ADR-014 mandates for plugin operations.
Documentation updates (writing plot plugins guide, migration plan).	No new docs covering plot plugin authoring or migration were located in docs/ or improvement_docs beyond the ADR itself.	Low	Reduces discoverability but does not immediately block runtime behaviour; still required for ADR completion.
Recommended focus
Immediate fix: Rework the legacy plot builder/renderer so fallback execution truly mirrors `legacy.plotting`, restoring safe default behaviour.

High-priority compliance: Ship the missing viz/plugins.py helpers, extend metadata schema to expose default_renderer, and broaden .plot() resolution to honour renderer overrides/environment settings.

Stability improvements: Introduce a PlotPluginError hierarchy and integrate validate_plotspec into the default renderer path.

Tooling & docs: Implement the CLI commands and author the promised documentation to complete ADR-014’s deliverables.

# ADR-015

## 1

ADR-015 Gap Analysis
Severity scale:

High – Violates a core ADR guarantee or blocks third-party plugin authors.

Medium – Mismatch that can confuse operators or degrade adoption but has workarounds.

Low – Minor polish/clarity gaps.

Area	Severity	Evidence & Details	Recommended Attention
CalibratedExplanations.from_batch still returns a pre-built legacy container instead of instantiating a collection from plugin-supplied batch data, so request metadata and plugin collection metadata never land on the runtime façade. This keeps plugins coupled to the legacy _use_plugin=False path ADR-015 was meant to retire.	High	ADR calls for a lightweight façade that stores explainer, request metadata, and plugin-supplied collection metadata via from_batch(...), but the implementation only plucks a container object out of collection_metadata and returns it without copying or annotating metadata.	Rework CalibratedExplanations.from_batch (and related collection wiring) so it constructs a fresh container from the batch payload, wiring through request metadata and plugin-provided artefacts.
Environment override names diverge: ADR specifies CE_EXPLANATION_PLUGIN / CE_EXPLANATION_PLUGIN_FAST, yet the resolver looks for per-mode keys like CE_EXPLANATION_PLUGIN_FACTUAL / _ALTERNATIVE. Operators following the ADR will find their overrides ignored.	Medium	ADR precedence ladder lists the two environment variables explicitly, but _build_explanation_chain builds the key by appending the upper-cased mode (CE_EXPLANATION_PLUGIN_FACTUAL, etc.).	Align the resolver with the ADR (accept the documented keys and/or update the ADR + tooling documentation together).
Predict-bridge invariants aren’t enforced. ADR mandates that any component seeing [predict, low, high] must fail if predict falls outside [low, high], but LegacyPredictBridge just forwards values unchanged, so a plugin can silently emit inconsistent intervals.	High	ADR requirement for strict interval enforcement; LegacyPredictBridge.predict simply returns arrays from the legacy explainer with no checks on predict, low, or high relationships.	Add guardrails (e.g., inside the bridge or monitor) to assert the low ≤ predict ≤ high invariant before batches reach plugins.
These items should be prioritised to bring the codebase back in line with ADR-015’s guarantees.

## 2

Severity scale
Critical – Breaks the ADR’s core contract or makes the documented workflow unusable.

High – Major deviation that risks incorrect behaviour or blocks extensibility promised by the ADR.

Medium – Noticeable gap that weakens guarantees or increases tech debt but has a workaround.

Low – Minor polish/consistency issues.

ADR-015 gap analysis
Area	Gap summary	Severity	Evidence	Suggested follow-up
Built-in FAST plugin availability	ADR-015 requires three in-tree plugins (factual, alternative, FAST) so defaults work without extra installs. The repository only registers factual and alternative adapters and tells operators to install external_plugins to obtain FAST support; _resolve_explanation_plugin even raises if nothing is found, so explain_fast(_use_plugin=True) fails out of the box.	Critical	Ship an in-tree FastExplanationPlugin that mirrors legacy FAST behaviour (including interval/plot metadata) so the documented defaults remain functional.
Collection reconstruction contract	ADR-015 states CalibratedExplanations.from_batch(...) becomes the common constructor, storing explainer, request metadata, ordered explanations, etc. Current code expects the plugin to stuff a ready-made container into collection_metadata["container"] and simply returns it, leaving no path for new containers to be built from batch payloads.	High	Rework ExplanationBatch handling so plugins supply data (instances, metadata, classes) and from_batch materialises the collection, capturing telemetry and request metadata as the ADR describes.
Trust enforcement during resolution	The ADR promises only trusted plugins are chosen unless a user opts in to an untrusted identifier. _resolve_explanation_plugin falls back to find_explanation_plugin, which returns the descriptor regardless of trust, so untrusted registrations are silently activated.	High	Gate resolution on descriptor.trusted unless the override explicitly names the identifier or the user enables trust (e.g., CE_TRUST_PLUGIN).
Predict bridge interval guarantees	ADR-015 states the bridge “enforces” low <= predict <= high and any violation must raise. LegacyPredictBridge.predict forwards raw arrays and never checks interval consistency, so malformed calibrator output would slip through silently.	High	Add validation in the bridge (or monitor) to assert the invariant before returning to plugins, failing fast on inconsistent calibrator output.
Context immutability & helper handles	The ADR frames helper_handles as read-only perturbation buffers inside a frozen ExplanationContext. Runtime wiring injects the mutable CalibratedExplainer instance itself ({"explainer": self}), giving plugins full, writable access to internals.	Medium	Replace the direct explainer handle with purpose-built, immutable helper objects (or read-only wrappers) that expose only the approved perturbation/pre-computed data.
These items capture the largest deviations between ADR-015 and the current implementation and should be prioritised according to their severity.

# ADR-016

## 1

Gap Analysis Overview
I reviewed ADR-016 line-by-line and compared each normative requirement with the current plotting implementation in calibrated_explanations. To prioritize follow-up work, I’m using a three-level severity scale:

High – Blocks ADR compliance or breaks the documented contract.

Medium – Partially implemented or implemented in a fragile way.

Low – Cosmetic drift or missing documentation that does not immediately block compliance.

Findings by ADR Decision Area
1. Plot kinds and metadata
Missing kind, mode, and feature_order fields on PlotSpec – The ADR requires these to live directly on the spec, but the dataclass still matches the original ADR-007 MVP and exposes only title, figure_size, header, and body. Builders infer kind/mode later, so callers cannot supply or inspect them at the dataclass layer as mandated.
Severity: High.

Derived feature_order loses the original feature indices – When builders convert a dataclass to a dict they synthesize feature_order as range(len(entries)), which reflects the rendered order but not the source indices that ADR-016 says must be preserved. Any sorting step therefore erases the caller’s feature index contract.
Severity: High.

No integrated save-behavior metadata – ADR-016 standardizes save_behavior hints, yet bar-plot builders neither accept nor propagate this block. The triangular helper only injects defaults ad hoc inside plotting.py, so callers cannot request filesystem output through the spec itself unless they hand-edit the dict afterward.
Severity: Medium.

2. Legacy parity through plugins
The ADR’s delegation of pixel-perfect parity to the legacy plugin is honored: the default plot builder/renderer pair still routes through the legacy pathway, and the PlotSpec builders focus on semantic data only.
Severity: N/A (compliant).

3. Non-panel plots
Triangular and global helpers already emit plain dict payloads with kind and mode, matching the documented escape hatch for non-panel plots.
Severity: N/A (compliant).

4. Validation and testing hooks
Validator still enforces only the old MVP envelope – viz.serializers.validate_plotspec checks for plotspec_version, a body list, and bar labels/values, but it never verifies the presence or legitimacy of kind, mode, feature_order, or the dataclass instances that ADR-016 calls out.
Severity: High.

Builders do not call the validator before handing payloads to adapters – Even where we do produce dict payloads (e.g., build_factual_probabilistic_plotspec_dict), we return the raw dictionary immediately, so structural drift can slip through unvalidated.
Severity: Medium.

5. Hybrid execution
The codebase still exposes both execution paths (legacy default plus optional PlotSpec rendering) through the plugin registry and plotting.py, satisfying the ADR’s coexistence requirement.
Severity: N/A (compliant).

Additional Observations
The core dataclasses and adapter docstrings still reference ADR-007, which can confuse future maintainers but does not currently block functionality.
Severity: Low.

Priority Backlog
Gap	Severity	Impact
Add first-class kind, mode, feature_order, and save_behavior fields to PlotSpec (and update builders/tests)	High	Blocks adherence to ADR-016 contracts; downstream adapters can’t rely on spec metadata.
Preserve original feature indices in feature_order instead of re-numbering	High	Breaks caller expectations once bars are sorted or filtered.
Expand validate_plotspec to enforce ADR-016 invariants and invoke it from builders before returning	High	Without this guard, adapters receive incompatible payloads undetected.
Thread save_behavior hints through builders instead of post-processing dicts	Medium	Callers can’t request saves declaratively via PlotSpec.
Update builder pipeline to call the validator and document the behavior	Medium	Prevents future drift.
Refresh docstrings/references from ADR-007 to ADR-016	Low	Avoids confusion but not urgent.
No automated tests or commands were run for this analysis (informational review only).

## 2

Severity scale
Critical – Blocks ADR intent outright; functionality cannot meet contract without major work.

High – Substantial divergence that risks incorrect behaviour or breaks downstream integrations if left unresolved.

Medium – Noticeable misalignment that could cause drift or maintenance burden but has workarounds.

Low – Cosmetic or documentation drift with limited impact.

Gap analysis
Area	ADR-016 expectation	Observed implementation	Gap & impact	Severity	Recommendation
PlotSpec structure	PlotSpec must expose kind, mode, and feature_order, and support save_behavior metadata for adapters.	The PlotSpec dataclass only holds title, figure_size, header, and body; there are no fields for kind/mode/feature order/save hints, so builders cannot populate them directly.	Core schema from ADR-016 is absent, so any consumer of the dataclass lacks required metadata. Adapters must infer kind/mode heuristically, which breaks the contract and risks misclassification when additional plot types are introduced.	Critical	Extend PlotSpec (and related dataclasses) with the ADR-required fields and update builders to set them explicitly.
PlotSpec validation hook	Builders must call viz/plotspec.validate_plotspec to enforce presence of kind/mode, dataclass integrity, and numeric intervals before adapters render.	The only validator lives in viz/serializers.validate_plotspec, checks a legacy envelope (plotspec_version, body bars), and is neither located in plotspec.py nor invoked by builders that return dataclasses.	Structural checks mandated by the ADR never run, so malformed specs can reach adapters undetected. Missing kind/mode isn’t caught because the validator doesn’t inspect them.	High	Implement the new validator in viz/plotspec.py per ADR-016, ensure it enforces kind/mode/interval rules, and invoke it in each builder before returning.
Feature ordering metadata	Specs that display features must expose the original feature indices through feature_order for interoperability.	Builders accept features_to_plot, but the helper plotspec_to_dict rewrites feature_order to range(len(entries)), losing the original indices and ignoring the ADR contract.	Downstream consumers cannot reconcile rendered bars with source features, breaking semantic parity and making validation impossible.	High	Preserve the original features_to_plot order when populating feature_order, and carry it through the dataclass schema.
Legacy parity boundary	ADR-016 delegates colour/layout quirks to the legacy plot plugin so PlotSpec builders stay semantic.	While a legacy builder/renderer pair exists, the PlotSpec path still bakes in legacy-specific toggles (solid_on_interval_crosses_zero, probability palette reproduction) inside the shared dataclasses and builders.	PlotSpec semantics are coupled to legacy behaviour, limiting adapter flexibility and obscuring which parts belong to the canonical spec versus legacy compatibility.	Medium	Isolate parity-only switches into the legacy plugin (or adapter flags) and keep the core spec focused on semantic roles such as color_role or mode.
Documentation alignment	ADR-016 replaces ADR-007 as the governing schema, but the PlotSpec module docstring still references the older decision.	Readers are pointed to outdated guidance, increasing the chance of further drift.	Low	Update module documentation to reference ADR-016 and summarise the current contract.
Next steps
Prioritise the schema/validation work (critical & high severities) to unblock compliant adapters.

Plan a follow-up refactor to decouple legacy toggles from the canonical spec.

Refresh documentation once the structural changes land to keep contributors aligned.

# ADR-017

## 1

ADR-017 Gap Analysis
Methodology & Severity Metric
Severity score = Impact (1 = localized, 2 = module-wide, 3 = cross-cutting) × Urgency (1 = long-term, 2 = next milestone, 3 = immediate). A score ≥7 is considered High, 4–6 Medium, ≤3 Low.

Requirements are taken from ADR-017, notably the mandates on module naming, attribute access, helper structure, automation, and telemetry tracking.

Compliance Highlights
The contributor guide now carries the ADR-017 quick-reference table, covering naming rules for modules, classes, callables, and registry identifiers, satisfying the documentation guidance from the decision record.

Canonical module moves (e.g., core.venn_abers, core.interval_regressor, viz.plots) have been completed with shims parked under legacy/, aligning with the Phase 2 rename objectives.

Outstanding Gaps
Score	Impact × Urgency	Issue & Evidence	Recommended Action
9	3 × 3	Double-underscore state still exposed. Helpers such as core/calibration_helpers.py continue to mutate _CalibratedExplainer__initialized, directly violating ADR-017’s ban on introducing or propagating mangled names outside legacy shims.	Replace helper access with public/protected accessors per Phase 3 §1 of the roadmap; audit for additional SLF001 suppressions touching __* attributes and refactor them onto supported APIs.
8	4 × 2 (Impact rounded up because tooling spans repo)	Naming guardrails lack automated enforcement. Ruff is configured for general PEP8 naming (select = ["N"]) but no rule or hook exists to flag snake_case file violations or new double-underscore fields, despite Phase 0 requiring such tooling. Manual # noqa: SLF001 suppressions bypass enforcement today.	Add explicit Ruff rules (e.g., enable N814, custom flake8-builtins plugins) or a bespoke CI script to scan filenames and attribute additions; fail CI on violations to make ADR-017 blocking ahead of the v0.9.0 milestone.
6	3 × 2	Kitchen-sink helper module persists. utils/helper.py still aggregates unrelated filesystem, import, and validation utilities under ambiguous names (safe_isinstance, safe_import), contradicting ADR-017’s mandate for topic-focused utility modules and the roadmap’s Phase 1 split item.	Execute the planned decomposition into focused modules (e.g., fs_utils, import_utils, validation_utils) and update import paths; reserve safe_* prefixes only when they communicate real semantics.
6	2 × 3	No telemetry for naming-lint drift. ADR-017’s status tracking calls for telemetry to observe naming-lint debt, yet there is no code in src/ emitting such signals (rg "lint drift" produces no hits), leaving maintainers without the intended feedback loop.	Extend existing telemetry infrastructure (e.g., perf.cache, core.wrap_explainer) with a lightweight channel that records lint warning counts per run; surface the metric in release dashboards to inform ADR-017 gating.
4	2 × 2	Legacy shims still first-class in runtime helpers. While transitional modules live under legacy/, primary helpers (e.g., core/prediction_helpers.py) still rely on underscore-prefixed attributes and legacy hooks guarded only by noqa: SLF001, signaling continued coupling to pre-ADR naming.	As part of the double-underscore remediation, add accessor methods on CalibratedExplainer and update helpers/tests to consume them, allowing eventual removal of SLF001 suppressions.
Watchlist / Upcoming Milestones
Telemetry & enforcement gates (v0.9.0 → v1.0.0): ADR-017 expects lint promotion to blocking status and quarterly legacy audits post-v1.0.0; once automation is in place, schedule dashboards/checks so the release checklist can cite concrete metrics.

Documentation parity: Continue scanning contributor and architecture docs to ensure deprecated identifiers are explicitly called out as the helper splits and accessor refactors land.

This analysis should equip the assignee with specific loci (core/calibration_helpers.py, utils/helper.py, pyproject.toml) and severity-ranked priorities to close the remaining ADR-017 compliance gaps.

## 2

Severity scale
Severity	Criteria
High	Blocks ADR-017 compliance or risks regressions if left unresolved.
Medium	Important for consistency/tooling but unlikely to break consumers immediately.
Low	Cosmetic or already compliant areas to monitor.
High-severity gaps
Mangled private attributes still in active code paths
Evidence: Core helpers and explainer internals continue to set or read __-prefixed attributes, despite ADR-017 limiting new double-underscore usage to legacy-only code. Tests also exercise the mangled APIs directly, reinforcing the dependency.
Impact: Violates ADR-017 rule 4, making it impossible to remove or protect these attributes without widespread breakage; raises risk that new code will copy the pattern.
Recommendation: Introduce protected accessors or migration shims that expose single-underscore names, refactor helpers/tests to use them, and add lint (or a custom Ruff rule) that rejects new double-underscore references outside legacy/.

Transitional shim naming out of policy
Evidence: _legacy_explain.py lives inside calibrated_explanations.core instead of under legacy/ or using a deprecated_ prefix, conflicting with ADR-017 rule 5 and the remediation roadmap that calls for isolating shims.
Impact: Contributors cannot easily distinguish active modules from transitional ones, so new features may target the wrong surface.
Recommendation: Relocate the shim into calibrated_explanations.legacy (or rename to deprecated_legacy_explain.py), update import sites, and emit a targeted DeprecationWarning to guide downstream users.

Medium-severity gaps
Kitchen-sink helper module not yet decomposed
Evidence: utils/helper.py still aggregates unrelated filesystem, import, and numeric helpers, contrary to ADR-017 rule 6 and Phase 1 action item to split the file by responsibility.
Impact: Makes it hard to discover helpers, encourages ambiguous names, and complicates enforcement of naming conventions.
Recommendation: Break the module into focused units (fs_utils, import_utils, etc.), leaving thin re-export shims if needed for compatibility.

Tooling gaps for double-underscore detection
Evidence: ADR-017 implementation notes call for lint or CI guardrails that fail on new double-underscore attributes, but the current Ruff configuration only enables generic naming rules and lacks any custom detector.
Impact: Without automated checks, developers can inadvertently add more mangled names even after manual cleanup.
Recommendation: Extend pre-commit/CI (e.g., custom Ruff plugin or script) to scan for __\w+ identifiers outside sanctioned legacy paths and fail builds when they appear.

Documentation drift around transitional modules
Evidence: The nomenclature review still cites a core.py shim that no longer exists after refactors, breaching ADR-017 rule 8 to keep docs aligned with canonical names.
Impact: Readers may chase non-existent files, hampering onboarding and review.
Recommendation: Refresh the review and related docs to reference current module names (or reinstate a documented shim if compatibility requires it).

Low-severity observations (compliant/healthy areas)
Contributor guidance updated
The main contributing guide includes ADR-017 quick references, giving developers an accessible summary of expectations.

Plugin identifiers follow dot-delimited lowercase format
Built-in plugin registration adheres to the ADR-017 identifier scheme, so registry-facing code is aligned with the standard.

Recommended next steps
Prioritise the high-severity refactors (protected accessors + shim relocation) to unblock later enforcement work.

Schedule the helper-module split and lint enhancements as part of Phase 3 tooling hardening.

Audit documentation for stale module references and keep ADR-017 excerpts synchronized with the codebase.

# ADR-018

## 1

Overview
Reviewed ADR-018 (code documentation standardization) expectations alongside the current implementation plans to understand required behaviors such as numpydoc formatting, coverage targets, deprecation guidance, and automation hooks.

Assessed tooling and CI integration that enforce ADR-018, including the documented coverage script and lint workflow stages that run pydocstyle with the numpy convention and enforce an overall coverage floor.

Executed the repository’s coverage script, which reports 99.59 % overall docstring coverage (modules/classes/functions at 100 %, methods at 99.24 %), confirming the baseline enforcement level and highlighting the small set of remaining gaps.

Severity scale
Level	Meaning
4 – Critical	Clear ADR violation impacting public APIs or misleading consumers; immediate remediation required.
3 – High	Material non-compliance that risks user confusion or tooling failure; schedule remediation next cycle.
2 – Medium	Localized inconsistency or missing coverage that weakens ADR guarantees but has limited blast radius.
1 – Low	Stylistic or completeness improvements that would tighten alignment but do not currently block ADR goals.
Gap inventory
Gap	Evidence	Impact & Severity	Recommended action
IntervalRegressor.__init__ docstring is outdated (documents model, X_cal, y_cal parameters that are not present) and therefore fails ADR-018’s “type information parity” rule for public callables.	The numpydoc block still lists parameters removed from the signature.	Misleads library users and reviewers about required inputs, undermining trust in generated docs. Severity 4 (Critical)	Rewrite the docstring to describe the current single-argument contract, documenting expected CalibratedExplainer behavior and any side effects (e.g., cached bins). Include accurate type/shape guidance.
Property setter IntervalRegressor.bins lacks any docstring, leaving a public mutator undocumented despite ADR-018’s coverage mandate.	Setter body appears without summary text.	Breaks the “public callables must be documented” guarantee for a core calibrator API. Severity 3 (High)	Add a one-line summary and clarify accepted shapes (1D array-like) plus behavior when None is supplied.
Internal guard methods WrapCalibratedExplainer._assert_fitted/_assert_calibrated omit the required one-line summaries for underscore-prefixed helpers.	Helpers are bare functions with logic but no docstrings.	Violates ADR-018’s internal-helper rule; weakens linting consistency because these are frequently chained. Severity 2 (Medium)	Add concise summaries (e.g., “Ensure the wrapper has completed fit before continuing”) and note raised exceptions.
Nested CombinedPlotPlugin class definitions created in the registry functions expose undocumented __init__/build/render methods. They are not prefixed with _, so callers inspecting them receive zero doc guidance.	Each nested class lacks docstrings for all three methods.	Leaves dynamically returned plugin objects undocumented, which can surprise users when debugging or extending plotting. Severity 2 (Medium)	Either convert the nested class into a documented dataclass/NamedTuple or add inline docstrings describing builder/renderer delegation semantics.
Several deprecated shim modules warn at runtime but do not include .. deprecated:: or Warnings sections in their docstrings, contrary to ADR-018’s guidance on Sphinx-visible deprecation notices.	Example: legacy plotting shim docstrings contain plain text only.	Users browsing the API docs miss structured deprecation metadata; automation cannot surface timelines. Severity 2 (Medium)	Amend module docstrings with a .. deprecated:: directive that states the replacement module and removal timeline, mirroring ADR expectations.
Residual undocumented methods (combined plugins + guard helpers + bins setter) explain the gap between 396 total methods and 393 documented reported by the coverage tool.	Custom AST audit lists the specific methods still lacking docstrings.	Confirms outstanding cleanup tasks needed before tightening the CI --fail-under threshold. Severity 1 (Low)	Track these in the ADR-018 documentation debt checklist and re-run the coverage script to ensure method coverage reaches 100 %.
Additional observations
Most importable modules—including the plugin registry—now open with clear high-level docstrings, aligning well with ADR-018’s module-summary requirement.

WrapCalibratedExplainer’s public workflow methods already adopt structured numpydoc sections and cross-references, reflecting good adherence in critical APIs; once the internal helper docstrings are added, that class will be a strong exemplar.

Tooling integration is mature: CI runs pydocstyle plus the coverage script (failing below 94 %), and configuration pins the numpy convention, which satisfies the ADR’s automation clause. Consider raising the --fail-under threshold after addressing the remaining documented gaps to lock in the higher baseline.

Addressing the severity‑3+ items first (IntervalRegressor docstrings) will close the most pressing ADR-018 deviations; the remaining medium/low findings can be batched into the next documentation debt sweep while keeping the coverage dashboard above the mandated floor.

## 2

Overview
ADR-018 mandates consistent numpydoc documentation, module summaries, deprecation messaging, and automation guardrails for calibrated_explanations. I reviewed the current implementation, sampled representative APIs, and executed the in-repo coverage tooling to identify compliance gaps and their severity.

Severity scale
Critical (P0) – breaks a published guardrail or materially erodes public API guarantees.

High (P1) – violates an explicit ADR requirement on widely used APIs, risking regression before release.

Medium (P2) – violates ADR guidance but has limited blast radius or easy remediation.

Low (P3) – stylistic drift or missing polish that does not threaten rollout.

High-severity gaps (P1)
WrapCalibratedExplainer public inference APIs lack required numpydoc sections and type detail.
Methods such as explain_counterfactual, explore_alternatives, explain_fast, explain_lime, predict, predict_proba, and calibrated_confusion_matrix only include a short summary or See Also block. They omit the ADR-required Parameters, Returns, and Raises sections and fail to document accepted shapes/ranges for key arguments (threshold, uq_interval, etc.), leaving consumers without the stability cues ADR-018 promises for this contract surface. Given these methods define the stable wrapper API explicitly called out in the ADR, the gap is high-severity.

Medium-severity gaps (P2)
Private guardrail helpers miss the mandated one-line summaries.
_assert_fitted and _assert_calibrated enforce runtime guardrails on the wrapper but have no docstrings, contradicting the ADR rule that internal helpers carry at least a single-line summary.

IntervalRegressor.bins setter lacks documentation.
The property setter updates Mondrian bins without any docstring, so tooling reports it as undocumented; it should explain expected shapes/dtypes per ADR-018’s internal-helper rule.

Low-severity gaps (P3)
wrap_explainer module docstring is not descriptive.
The file-level docstring (“Mechanical extraction…”) does not summarise the module’s purpose, key abstractions, or constraints as ADR-018 requires for module headers.

Strengths and existing guardrails
Coverage automation is in place. The repository includes a docstring coverage script with a passing baseline (99.59% overall, 99.24% for methods) and CI enforces a 94% minimum, aligning with ADR-018’s automation hook requirement.

Tooling guidance is documented. CONTRIBUTING.md spells out how to run pydocstyle (numpydoc convention) and the coverage checker, satisfying the ADR adoption step for contributor enablement.

Configuration matches ADR direction. pyproject.toml configures pydocstyle with the numpy convention and excludes only non-target directories, so the lint gate is ready once the style gaps are resolved.

Recommended next steps
Prioritise rewriting the WrapCalibratedExplainer public method docstrings using full numpydoc sections, explicitly documenting argument types, accepted ranges, return values, and stability guarantees to uphold the ADR’s public API guardrail.

Add concise docstrings to _assert_fitted, _assert_calibrated, and the IntervalRegressor.bins setter to restore 100% compliance and keep internal tooling noise-free.

Replace the wrap_explainer module header with a paragraph describing its wrapper role, scikit-learn style workflow, and any notable constraints to satisfy the module-level guidance.

# ADR-019

## 1

Severity scale
I assessed each gap with a qualitative metric aligned to ADR-019’s release expectations: Critical (blocks the ADR’s promised safeguards), High (materially undermines enforcement or auditability), Medium (needs remediation but doesn’t immediately break the gate), and Low (follow-up hygiene).

Compliance summary
ADR-019 requirement	Current status	Severity	Evidence
≥90% package-wide fail-under in local + CI runs	Local/CI still use --cov-fail-under=88, .coveragerc remains at 85, and the release checklist gates on 88	Critical	pytest.ini and release checklist show 88; .coveragerc retains 85
≥95% coverage for critical-path modules enforced via per-path thresholds	No per-module thresholds in .coveragerc; baseline modules remain well below 95% (plotting 63.1%, builders 70.4%, etc.)	Critical	.coveragerc lacks per-path gates; gap plan table documents deficits
Codecov patch coverage ≥88% as a required status check	Default patch status is informational; only a scoped “runtime-calibration” target exists with zero tolerance but not marked required	High	ADR mandates gating; Codecov config keeps the default informational
Exemptions documented with rationale and expiry	.coveragerc lists exclusions with rationale but no expiry metadata	Medium	ADR requires expiry; excludes omit dates
Public API guardrails (WrapCalibratedExplainer, prediction helpers, registry, etc.) stay exercised	Gap plan notes major cold paths across explanation, registry, CLI, prediction helpers, legacy gateways	High	Modules underpinning the public API remain under-tested
Migration plan Step 3: workflow enforces 90%	Test workflow delegates to make test-cov, which inherits the 88% fail-under, so Phase 2 target (90%) remains unmet	High	Standardization plan requires CI update; workflow still below target
Waiver inventory governance	Inventory exists and is empty (good), but sustained audits rely on raising fail-under to 90 and catching regressions	Medium	Inventory maintained, but effectiveness tied to missing gates
Release checklist alignment	Release checklist still instructs running pytest --cov --cov-fail-under=88, lagging behind the ADR’s final gate	High	Checklist mismatch will allow sub-90% releases
Focus areas needing attention
Raise the enforced coverage floor to 90%.
Update pytest.ini, .coveragerc, make test-cov, and the release checklist so contributors, CI, and release managers all fail below the ADR-mandated threshold. Without this, ADR-019’s headline guarantee is not delivered.

Implement per-path enforcement for critical modules.
Add module-level fail_under checks (or coverage json parsing) for calibrated prediction helpers, interval regression, serialization, registry, and plugins. The current coverage snapshot shows these modules far below 95%, so remediation and gating must happen together.

Make Codecov’s ≥88% patch check required.
Flip the default patch status from informational to required, or rely on GitHub branch protection tied to the “runtime-calibration” context. ADR-019 explicitly calls for blocking merges on sub-88% patch coverage.

Finish remediation for the public API hotspots.
The coverage gap plan outlines specific cold paths (plotting router, explanation caching, registry trust toggles, CLI error handling, prediction helpers, package gateways). These align exactly with ADR-019’s guardrail clause and must reach ≥95% (critical path) or ≥90% (gateways).

Attach expiry metadata to every .coveragerc exclusion.
Add “Expires YYYY-MM-DD” (or equivalent) comments and review cadence to satisfy the documented-exemptions clause. This also keeps the waiver inventory meaningful when cross-referenced during releases.

Keep governance docs synchronized.
Once the technical gates are updated, ensure the release checklist and related governance material (waiver audit, release plan milestones) reflect the new numbers so future audits do not backslide to 88%.

Addressing the critical and high-severity items restores ADR-019’s intended protections; medium-severity tasks polish the sustainability posture so the guardrails stay effective over time.

## 2

Overview
Latest make test-cov run completes successfully with 91.1 % statement coverage (9,475 statements / 843 misses).

ADR-019 mandates a 90 % package-wide floor, 95 % thresholds on critical paths, ≥88 % Codecov patch gating, documented/expiring exemptions, and sustained public-API coverage.

Severity scale
Critical – Directly violates an accepted ADR requirement and can block release readiness.

High – ADR requirement is only partially met, leaving a material risk gap.

Medium – Process/documentation gaps that weaken long-term compliance.

Low – Minor follow-ups that improve polish but do not threaten compliance.

Gap analysis
Gap	Evidence	Severity	Recommended action
CI/local gates still enforce --cov-fail-under=88 (pytest) and fail_under=85 (.coveragerc) instead of the required 90 % package floor for the v1.0 gate.	ADR requires CI --cov-fail-under=90 and matching local tooling; pytest.ini holds the threshold at 88 and .coveragerc still sits at 85.	High	Raise both pytest and .coveragerc fail-under values to 90, stage a dry-run in CI, and communicate the uplift before branch protection switches on.
Codecov patch gate is non-blocking (informational: true) and only targets a subset of runtime paths, undermining the ≥88 % patch guard requirement.	ADR mandates blocking patch coverage ≥88 %; codecov.yml keeps the default gate informational and narrows scope to “runtime-calibration.”	High	Flip informational to false, broaden the monitored paths to all src/calibrated_explanations/**, and verify PRs fail when patch coverage drops below 88 %.
Critical-path modules fall short of the 95 % target (e.g., core/calibrated_explainer.py 90.1 %, core/interval_regressor.py 94.7 %, core/prediction_helpers.py 90.0 %, serialization.py 87.0 %).	ADR calls for ≥95 % on calibrated prediction helpers, interval regression, serialization, registry; coverage report shows persistent misses in those files.	Critical	Prioritize targeted tests per the coverage gap remediation plan (categorical init flows, Mondrian validation, serialization error paths) until each module reaches ≥95 %.
Public API modules remain under-covered (src/__init__.py 62.7 %, core/__init__.py 87.5 %), violating ADR guardrails that forbid excluding public entry points.	ADR states public API coverage must be preserved; coverage output shows large holes in lazy-import/deprecation logic.	Critical	Add tests that exercise lazy attribute access, deprecation toggles, and caching paths so API regressions surface immediately.
Plotting router remains at 63.0 % coverage, far below risk tolerance for shared rendering helpers and the published coverage gap plan.	The plan highlights plotting as a major deficit; current run confirms minimal progress (246 misses).	High	Execute Iteration 1 of the remediation plan—parameterize router tests, cover uncertainty segments, and normalize cross-platform save paths.
Legacy shims are blanket-excluded without documenting expiry dates, contrary to ADR guidance on exemptions.	ADR expects .coveragerc comments to include rationale and expiry dates; current omissions list reasons only.	Medium	Attach explicit expiry/review dates (or remediation tickets) to each exclusion and log them in the waiver inventory so they are audited regularly.
Serialization and registry dashboards required for release checklist are absent from governance docs, leaving no visibility into ADR enforcement during release prep.	ADR implementation status calls for coverage dashboards in the release checklist by v1.0.0-rc; current governance docs do not mention them (no release checklist updates referencing ADR-019 dashboards).	Medium	Update release checklist/governance docs to require reviewing the Codecov dashboard and module thresholds before freeze.
Areas on track
Package coverage currently exceeds the 90 % floor (91.1 %), so the raw metric is healthy once gates catch up.

make test-cov exists and mirrors CI options aside from the threshold gap; dev extra includes pytest-cov, fulfilling step 4 of the adoption plan.

Addressing the high/critical gaps above will bring ADR-019 fully into force and reduce the risk of future regressions slipping through coverage gates.

# ADR-020

## 1

Overview
ADR-020 aims to freeze the legacy user API surface by pairing a canonical contract document with signature regression tests and a release-process checkpoint. The repository has made tangible progress—most notably the contract analysis and a first wave of tests—but several guardrails remain incomplete or fragile.

Severity scale
High – A gap that leaves the legacy API unprotected against regressions the ADR explicitly targets.

Medium – A gap that weakens process visibility or makes it easy to bypass guardrails.

Low – A minor documentation or tooling improvement that would bolster confidence.

Gap assessment
Area	ADR expectation	Observed status	Severity
Release governance	Add a legacy-API checkbox to the release checklist so reviewers confirm contract/test updates travel with any intentional change.	The current release checklist covers navigation, quickstarts, telemetry, and gating builds but never asks about the legacy API contract or its tests, so a release could ship a breaking change unnoticed.	High
Regression tests for wrapper methods	Signature tests should ensure wrapper methods keep required knobs like threshold, low_high_percentiles, and lifecycle helpers.	The existing regression file checks WrapCalibratedExplainer.predict/predict_proba but never asserts that explain_factual or explore_alternatives continue recognizing the contract parameters—they still accept **kwargs, so a refactor could silently drop normalization for threshold/low_high_percentiles.	High
Contributor workflow	Contributors should consult the canonical contract before proposing API changes.	The contributor hub currently points people at the plugin contract and testing policy but never mentions the legacy user API contract, so reviewers have no process cue to update it alongside changes.	Medium
Contract documentation availability	Maintain a canonical description of the legacy surface.	The legacy contract document is present and detailed, mapping every supported method/parameter pair across the wrapper, core explainer, and collections, giving reviewers a clear reference.	Low (positive control)
Existing regression coverage	Ensure CI fails when critical parameters disappear.	The current regression suite already guards constructor signatures, prediction helpers, and collection ergonomics, and companion tests confirm wrapper defaults forward threshold and low_high_percentiles to the underlying explainer.	Low (positive control)
Recommended next steps
Augment the release checklist with an explicit “Legacy API contract updated?” checkbox (linking to the document and regression tests) so every release review audits the guardrail trio.

Extend regression coverage to assert that WrapCalibratedExplainer.explain_factual/explore_alternatives still normalize threshold, low_high_percentiles, and bins—for example by introspecting _normalize_public_kwargs or asserting behaviour via lightweight spies.

Update contributor guidance (and optionally PR templates) to reference improvement_docs/legacy_user_api_contract.md, ensuring authors attach contract/test updates to any API-impacting change.

Implementing these items will close the remaining high- and medium-severity gaps and fully realize ADR-020’s guardrail strategy.

## 2

Summary
Canonical contract is in place. The legacy API surface is fully documented, covering constructors, prediction helpers, explanation factories, and collection ergonomics, giving maintainers a concrete reference for ADR-020’s guardrails.

Signature regression tests are live. Unit tests assert the required methods exist and enforce parameter signatures for WrapCalibratedExplainer, CalibratedExplainer, and explanation collections, preventing accidental drift.

Gap analysis (Severity scale: High ⟶ immediate risk to ADR outcome; Medium ⟶ important but non-blocking; Low ⟶ minor follow-up)
Area	ADR-020 expectation	Current status	Severity	Evidence
Release governance	Add a legacy-API checkbox to the release checklist so reviewers confirm contract/test updates with any API change.	The published checklist omits any legacy API verification item, so releases lack an explicit gate for contract/test synchronization.	High	See cited checklist
Contributor workflow	Contributor guide should direct API authors to consult/update the contract when proposing changes.	CONTRIBUTING.md references ADRs broadly but never mentions the legacy contract, leaving reviewers without a documented reminder to update it.	Medium	See contributor guide
Notebook surveillance	Periodically audit notebooks against the contract to fold new usage into guardrails.	No scripts, docs, or tests automate or even remind maintainers to run these audits; the only mention of “audit notebooks” is the ADR itself.	Medium	Search results
Recommended follow-up

Add a legacy API confirmation step (checkbox + guidance) to docs/foundations/governance/release_checklist.md, linking to the contract and regression tests (addresses High-severity gap).

Update CONTRIBUTING.md to require checking improvement_docs/legacy_user_api_contract.md and updating the signature tests whenever contributors touch public API entry points (addresses Medium gap).

Define and document a lightweight notebook audit process—ideally scripted under tests/docs/ or scripts/—and reference it from the release checklist or contributor workflow (addresses Medium gap).

# ADR-021

## 1

Overview
CalibratedExplainer resolves interval learners through the plugin registry for both default and FAST modes, keeping the selected calibrator cached on the explainer instance.

Classification predictions delegate directly to VennAbers.predict_proba with output_interval=True, while regression predictions route to IntervalRegressor for percentile intervals or thresholded probabilities, matching the ADR’s three-mode breakdown.

The explanation pipeline records prediction, interval bounds, and—when available—the full probability cube, and CalibratedExplanations stores threshold/percentile metadata and caches probability matrices for downstream consumers.

Gap analysis (Severity scale: High = immediate semantic risk, Medium = meaningful functional gap, Low = documentation/interop polish)
High – No enforcement of the low ≤ predict ≤ high invariant promised in ADR-021.
The ADR states every runtime pathway (PredictBridge, plugins, explanation containers, downstream helpers) “must assert low <= predict <= high … treated as a failure rather than a recoverable warning.” In practice, _predict and explain_predict_step simply return arrays from the calibrators without validating the relationship, and the regression path even falls back to zero-filled placeholders when suppress_crepes_errors is enabled. This leaves the system unable to detect or halt on broken intervals, exposing consumers to silently incorrect confidence information.

Medium – FAST explanations drop the calibrated probability cube.
The ADR expects FAST perturbations to reuse the same Venn-Abers calibrator so “CE-formatted outputs contain both per-feature impacts and the original probability interval metadata.” However, explain_fast builds its prediction dictionary without adding the __full_probabilities__ entry that CalibratedExplanation relies on to cache per-instance probability matrices. As a result, FAST classification runs cannot expose the probability cube for JSON export or analysis, diverging from the documented semantics.

Low – JSON export metadata still embeds a live function, undermining the ADR’s promise about downstream tooling.
The consequences section calls out consistent behaviour for telemetry and JSON export. Yet _collection_metadata injects the assign_threshold callable directly into the payload, and _jsonify leaves it untouched, so any consumer attempting to json.dumps the result will hit a serialization error. While the core interval semantics are intact, this interop hole means the ADR’s downstream guarantee is not yet met.

## 2

Overview
ADR-021 codifies how CalibratedExplainer should deliver calibrated intervals across classification, percentile regression, and thresholded regression. The current implementation largely mirrors the documented flows: classification continues to rely on Venn-Abers, percentile regression on conformal predictive systems (CPS), and thresholded regression on a CPS + Venn-Abers hybrid, with metadata propagated through the explanation pipeline.

Alignment highlights
Classification – Interval learners resolved via the plugin registry still instantiate Venn-Abers calibrators, and _predict_impl always asks for interval outputs, ensuring probability/interval triples are produced and cached for explanations.

Percentile regression – IntervalRegressor is attached during initialization, _predict_impl validates percentiles before delegating to predict_uncertainty, and the explanation container records low_high_percentiles so downstream consumers preserve CPS semantics.

Thresholded regression – The probabilistic branch normalizes thresholds, recomputes CPS/Venn-Abers splits as required, caches the active threshold, and keeps CalibratedExplanations.y_threshold so payloads remain probability-oriented rather than percentile-based.

Shared behaviours – Interval calibrators are cached per explainer (including fast paths), predictions/intervals are always stored in the explanation payload, and the cached probability cube is exposed through collection helpers, satisfying ADR-021’s shared guarantees except for the explicit invariant checks noted below.

Gaps and severity
Area	Observation	Severity*
Interval validity invariant	ADR-021 mandates that every runtime pathway assert low <= predict <= high, treating any violation as a hard failure. The current implementation returns the (predict, low, high) tuple from _predict_impl and threads it through prediction_helpers.explain_predict_step into explanation objects without validating the relationship, leaving potential violations undetected. This gap affects every mode promised by the ADR and could let downstream consumers observe inconsistent “calibrated” intervals.	High (direct contradiction of ADR guarantee across all interval outputs)
*Severity scale: High – violates a mandatory ADR semantic with broad impact; Medium – partial coverage or limited-scope risk; Low – documentation/telemetry tidy-up or minor edge-case.

Addressing the high-severity invariant check should be the top priority to bring the codebase fully in line with ADR-021.

# ADR-022

## 1

Severity scale
Critical – The ADR commitment is unmet in a way that blocks the intended information architecture and needs immediate remediation.

Major – Partial implementation exists, but material gaps reduce the value of the ADR and should be prioritised next.

Compliant – The commitment is fully implemented and operating as intended; continue routine monitoring.

Gap summary
ADR-022 commitment	Current status	Severity	Notes
1. Restructure the Sphinx toctree into seven top-level sections	Not met	Critical	Landing page still groups content into four buckets (“Get started”, “Audience hubs”, “Shared foundations”, “Appendices”) rather than the mandated seven sections (Overview, Get Started, How-to Guides, Concepts & Architecture, Reference, Extending the Library, Governance & Support).
2. Split quickstarts into runnable classification/regression guides with troubleshooting	Implemented	Compliant	Dedicated guides exist for classification and regression quickstarts, each linked to CI smoke tests, with troubleshooting coverage alongside them.
3. Move maintainer/contributor workflows under Extending the Library & Governance	Implemented	Compliant	Contributor hub defers governance content to docs/foundations/governance/*, concentrating maintainer policies under the prescribed sections.
4. Create telemetry concept page and align plugin docs with registry/CLI workflows	Partially met	Major	Plugin contract now teaches registry/CLI usage, but the telemetry concept page was replaced with a relocation notice instead of substantive conceptual guidance.
5. Establish content ownership and pre-release review checkpoints	Implemented	Compliant	Section owners and a detailed release checklist codify the review cadence, satisfying the governance requirement.
Details and recommended follow-up
Seven-section navigation (Critical).

Impact: Readers still encounter a hybrid navigation model, so ADR-022’s role-based architecture is not in place, undermining discoverability and the ownership structure defined for each section.

Action: Rebuild the top-level toctree to expose the seven mandated sections explicitly (Overview, Get Started, How-to Guides, Concepts & Architecture, Reference, Extending the Library, Governance & Support) and ensure each maps to a dedicated index page.

Telemetry concept coverage (Major).

Impact: The concept page now only redirects readers to governance material, so ADR-022’s intent—presenting telemetry as a concept that ties into calibration narratives—is lost.

Action: Restore substantive concept-level content (purpose, schema overview, relationship to calibrated outputs) while keeping opt-in governance guidance linked for operators.

Quickstart split and validation (Compliant).

Classification and regression guides are separated, runnable, and linked to CI tests, meeting ADR expectations for validated snippets and troubleshooting support.

Continue running the smoke tests during release checklists to maintain coverage.

Maintainer workflow placement (Compliant).

Contributor landing page points to plugin contract, governance policies, and extending guides located in the sanctioned sections, aligning with ADR-022’s relocation requirement.

Maintain this structure as new maintainer content lands.

Governance cadence (Compliant).

Section ownership table and release checklist codify the pre-release review process envisioned by ADR-022.

Ensure each quarterly audit records outcomes against these artifacts to preserve accountability.

By addressing the critical navigation gap and the major telemetry concept shortfall, the project will bring ADR-022 back into full alignment while preserving the strengths already delivered for quickstarts, contributor workflows, and governance cadence.

## 2

Overview
ADR-022 commits the docs team to restructuring the Sphinx navigation into seven top-level sections, shipping validated quickstarts, relocating maintainer workflows, delivering telemetry guidance that matches the registry/CLI workflow, and formalising ownership plus pre-release checks.

Severity scale
Severity	Interpretation
High	Direct ADR requirement is unmet in a way that blocks the intended information architecture or confuses readers/owners.
Medium	ADR requirement is only partially met; content exists but placement or signalling undermines the intent.
Low	ADR requirement is satisfied; monitor during routine reviews.
Gap analysis
Commitment	Status	Severity	Evidence & notes
Restructure the Sphinx toctree into seven top-level sections (Overview, Get Started, How-to Guides, Concepts & Architecture, Reference, Extending the Library, Governance & Support).	Gap	High	The current landing page exposes “Get started,” “Audience hubs” (practitioner, researcher, contributor), “Shared foundations,” and “Appendices,” rather than the seven explicit sections mandated by ADR-022. Shared foundations then nests concepts, how-to, reference, and governance, so the promised top-level segmentation (including “Extending the library” and “Governance & support”) is not surfaced in the nav, making it harder to audit or delegate ownership at a glance.
Split quickstart material into validated classification and regression guides with troubleshooting.	Complete	Low	The “Get started” hub links to separate classification/regression quickstarts, each with runnable code and CI-backed environment callouts, and includes a troubleshooting checklist for common failures.
Move maintainer and contributor workflows under Extending the Library and Governance & Support.	Partial	Medium	Governance material (release checklist, PR guide, nav crosswalk) now sits under the governance subtree, but there is still no top-level “Extending the library” section—maintainer content is exposed as an “Audience hub” (contributor/index), so Extending/Governance aren’t first-class nav buckets as the ADR requires.
Create a telemetry concept page and align plugin docs with the registry/CLI workflow introduced in v0.8.0.	Partial	Medium	The plugin contract fully documents the registry, CLI discovery, and trust controls, but the telemetry “concept page” is an orphaned pointer that immediately redirects readers elsewhere instead of explaining the concept itself, so the architectural intent isn’t met.
Establish content ownership and pre-release review checkpoints.	Complete	Low	Ownership is captured per section with explicit responsibilities, and the release checklist audits navigation, quickstarts, optional extras, telemetry alignment, and owner sign-off before every release.
Recommended follow-ups
Restore the mandated top-level navigation. Promote “How-to guides,” “Concepts & architecture,” “Reference,” “Extending the library,” and “Governance & support” to first-class toctree entries so the ADR-022 structure is visible and auditable (currently the structure collapses under “Shared foundations” or “Audience hubs”).

Introduce a real telemetry concept page. Replace the orphan stub with a concept article that explains telemetry semantics at a conceptual level before deferring to governance details, ensuring readers understand why and when to enable it.

Clarify the “Extending the library” lane. Either retitle and reposition the contributor hub or add a dedicated Extending section that surfaces contributor workflows, keeping governance material separate as ADR-022 intended.

With those High/Medium gaps resolved, the documentation will once again reflect ADR-022’s original information architecture goals while retaining the strengths already delivered in quickstarts and governance procedures.

# ADR-023

## 1

Scope of ADR-023
ADR-023 documents the decision to exclude matplotlib_adapter.py from coverage while still running all visualization tests without coverage and maintaining an 85% project-wide bar inherited from ADR-019.

Severity scale used
Score	Description
1 (Negligible)	Cosmetic/documentation drift with no observable risk.
2 (Low)	Minor misalignment; unlikely to hide defects but should be corrected.
3 (Moderate)	Meaningful inconsistency that can block contributors or create noise.
4 (High)	Breaks a promised safeguard or meaningful part of the decision.
5 (Critical)	Completely defeats the ADR’s mitigation, exposing the team to the original risk.
For each gap, Severity = Impact × Likelihood, where both dimensions use the 1–5 scale above.

Gaps Identified
1. Viz test execution path promised by ADR is absent (Severity 25 = 5 × 5)
ADR expectation: CI should run the visualization suite with pytest --no-cov -m viz, keeping functionality validated while coverage skips the adapter.
Reality:

The default pytest configuration marks every visualization test for omission via --ignore=tests/unit/viz/... in addopts, so the suite never runs—either with or without coverage.

The main CI workflow only invokes make test-cov (which inherits that addopts) and does not schedule an alternate job using --no-cov to exercise the viz marker.

Impact (5 – Critical): The risk ADR-023 tried to mitigate—silently breaking matplotlib rendering while relying on documentation that tests still run—reappears. Any regression in matplotlib_adapter.py would now go undetected in automation.
Likelihood (5 – Critical): This is the active, current behavior in every CI run.

Attention needed: Stand up the promised “viz without coverage” test job (or equivalent script) and remove the global ignores so those tests execute under that job.

2. Coverage threshold documented at 85% but enforced at 88% (Severity 15 = 3 × 5)
ADR expectation: By exempting the adapter, the team can continue meeting ADR-019’s 85% coverage target.
Reality: pytest.ini forces --cov-fail-under=88, so CI will fail below 88%, not 85%.

Impact (3 – Moderate): Contributors following ADR guidance (e.g., expecting 85% to be acceptable) will still see CI failures. This misalignment generates avoidable churn and erodes trust in the ADR.
Likelihood (5 – Critical): Every CI test run applies the stricter threshold today.

Attention needed: Either align the pytest flag with 85% (matching the ADR) or update ADR-023 to explain the higher bar so expectations stay synchronized.

Summary of areas functioning as expected
The coverage configuration does omit src/calibrated_explanations/viz/matplotlib_adapter.py, matching the documented implementation detail.

Addressing the two high-severity gaps above will bring the implementation back in line with ADR-023’s intent and restore confidence that matplotlib regressions will be caught while honoring the documented coverage policy.

## 2

Severity scale
Critical – Directly undermines the ADR’s stated safeguards, creating an immediate risk of regressions.

High – Meaningfully erodes confidence in the decision by preventing intended mitigation or validation steps.

Medium – Causes divergence between documentation and practice, risking contributor confusion or drift over time.

Gap analysis for ADR-023
Critical severity
Viz regression suite is never exercised in CI

ADR-023 assumes CI runs viz tests without coverage (pytest --no-cov -m viz) alongside the standard coverage job, but the GitHub Actions workflow only calls make test-cov (which inherits coverage flags) and a “core-only” job that explicitly excludes viz (pytest -q -k "not viz"). No CI step invokes the coverage-free viz run the ADR depends on.

Impact: viz regressions can merge undetected because neither automated job runs the adapter tests the ADR treats as the safety net.

Default pytest configuration prevents running the viz adapter tests at all

The ADR documents that individual viz tests pass (pytest tests/unit/viz/test_*.py) and that the suite should run without coverage, yet pytest.ini hard-ignores the adapter-focused files that previously failed under coverage. Those ignores apply even when developers pass --no-cov or target the files directly, so the “manual safety valve” described in the ADR is non-functional.

Impact: maintainers cannot easily validate changes to matplotlib_adapter.py, eliminating the mitigation the ADR relies on and increasing the likelihood of silent breakage.

High severity
Coverage threshold in practice conflicts with ADR expectations

ADR-023 states the exemption still maintains the ADR-019 floor at “85%+”, but pytest.ini enforces --cov-fail-under=88, and the contributor guide reiterates the 88% gate. This makes the ADR inaccurate about compliance requirements and can mislead reviewers when assessing waivers or regressions tied to the exemption.

Impact: documentation drift increases the risk of incorrect waivers or misplaced confidence in the stated rationale.

Medium severity
Follow-up process documentation never materialized

The ADR calls for documenting the split viz/coverage workflow in contributor docs and CI configuration comments, but the contributing guide only mentions the coverage-enforced pytest invocation, with no guidance on running the viz suite without coverage. This leaves new contributors unaware of the required manual step once the ignores are lifted, and further entrenches the drift noted above.

Recommended focus
Immediate: Restore a coverage-free viz test job (or equivalent command) in CI and adjust pytest.ini so developers can execute the viz suite when needed.

Short term: Reconcile ADR-023’s narrative with the enforced 88% gate (either by updating the ADR or aligning the configuration) and add the missing contributor documentation so the long-term maintenance plan is clear.

# ADR-024

## 1

Severity scale
High – Breaks the ADR contract in a way that risks runtime failures or user-visible regressions.

Medium – Contract divergence that can mislead integrators or force workarounds but does not immediately break execution.

Low – Documentation drift or minor behavioural mismatch with negligible direct impact.

Identified gaps
#	Area	Evidence	Impact	Severity	Recommendation
1	_plot_global ignores show=False	ADR-024 requires every helper to honour the shared headless contract so callers can run without a visual backend when show=False and no save metadata is provided. The implementation always executes plt.show() at the end of _plot_global, regardless of the show flag’s value.	Headless or CI workloads that dutifully pass show=False still pay the cost of materialising figures and trigger GUI calls, violating the contract and risking hangs or errors in environments without a display server.	High	Guard the entire rendering path (including the final plt.show()) behind the show flag, mirroring the other helpers. Early-return when show=False and no persistence metadata is supplied so ADR-024’s headless guarantee holds.
2	path + title + ext contract drift	ADR-024 states filenames are assembled via raw string concatenation (path + title + ext) with no separator injection, meaning callers must pre-bake separators. The current _compose_save_target helper now detects directory-like paths and inserts OS separators automatically.	Documentation consumers still believe separators are never added; they may over-compensate (e.g., double slashes) or mis-document downstream contracts. Behaviour is actually more user-friendly, so the risk is mostly confusion.	Low	Update ADR-024 (or cross-linking docs) to describe the new _compose_save_target semantics so plotspec authors know separators are normalised for directory inputs.
No additional contract gaps were observed in the other legacy helpers; their guards and payload expectations align with the ADR.

## 2

Overview
You asked for an in-depth gap analysis of ADR-024 (“Legacy Plot Input Contracts”) against the current implementation in src/calibrated_explanations/legacy/plotting.py. The ADR positions itself as the canonical reference for consumers building plot specifications, so any drift can directly affect downstream builders.

Severity rubric
Severity	Definition
Critical	Mismatch blocks implementers or invalidates a documented contract with high risk of runtime failure.
High	Mismatch forces consumers to add undocumented workarounds or diverge from the intended contract.
Medium	Mismatch can cause confusion or duplicated effort but is unlikely to break flows outright.
Low	Minor documentation drift or cosmetic inaccuracy with negligible practical impact.
Gap findings
#	Area	Observed gap	Evidence	Severity	Recommended attention
1	_plot_global display control	ADR promises that every helper honours show=False to keep headless runs silent, but _plot_global always calls plt.show() even when show=False, so consumers cannot suppress rendering when matplotlib is present.	ADR contract: show short-circuits before the matplotlib guard.
Implementation: show read from kwargs, but plt.show() executed unconditionally.	Critical	Gate the final plt.show() on show, add parity tests, and update ADR once behaviour matches.
2	_plot_global save contract	ADR states that all helpers expose identical title/path/save_ext semantics, yet _plot_global lacks parameters for saving altogether, so plotspec builders cannot meet the documented requirement.	ADR shared mechanics and instance contract claim identical display controls and cite _plot_global lines.
Implementation signature omits title, path, and save_ext; there is no saving loop.	High	Either extend _plot_global with save support or narrow the ADR language to exempt it, then propagate guidance to docs.
3	Save-path construction	ADR warns that filenames are produced via raw string concatenation (path + title + ext) with no separator insertion, but the current helper normalises paths through _compose_save_target, automatically inserting OS separators when the base looks like a directory.	ADR wording about raw concatenation.
Implementation uses Path(path) to join directory-style bases before concatenation.	Medium	Refresh ADR text to describe _compose_save_target semantics so builders don’t overcompensate (e.g., double separators).
Additional observations
No discrepancies were found in the per-instance helpers’ interval and payload handling; the ADR descriptions still match the code paths today.

Once the critical/high issues are resolved, consider re-validating ADR-024 before promoting it from “Draft” to an accepted state, so downstream contracts stay trustworthy.

Let me know if you’d like suggested code changes or test ideas to close the critical/high gaps.

# ADR-025

## 1

Overview
I reviewed ADR-025 against the current legacy plotting implementation and its unit tests to surface any mismatches or missing coverage that could jeopardize the documented rendering contract.

Gap Analysis
Area	Evidence	Severity*	Notes
_plot_regression axis symmetry	ADR-025 states that interval mode “tracks running min/max to set symmetric x-limits around all contributions.” The implementation merely forwards the raw extrema to Matplotlib (ax_main.set_xlim(x_min, x_max)), so negative and positive ranges can differ.	High	The rendered scale can drift from the documented symmetric framing, creating divergent plots relative to the ADR. Restoring symmetry (e.g., clamp to ±max(
Interval guard for one-sided explanations	ADR-025 requires interval rendering to raise a Warning when an explanation is one-sided. The code performs this check (raise Warning(...)). However, the shared test double hard-codes is_one_sided() to return False, so the guard is never exercised.	Medium	Without a regression test, the guard could regress silently (e.g., refactoring might drop the exception). Add a targeted test that toggles is_one_sided() to ensure the documented failure path remains enforced.
*Severity legend: High = documented behaviour currently diverges in production code; Medium = behaviour matches ADR but lacks automated protection; Low (none listed) = minor documentation/test completeness issues.

## 2

Overview
Architectural Decision Record ADR-025 defines strict behavioural and visual requirements for the legacy plotting helpers. I reviewed the current implementation in src/calibrated_explanations/legacy/plotting.py against those commitments to surface compliance gaps that could jeopardize parity as the renderer evolves.

Gap Analysis
Area	Gap Description	Evidence	Severity*	Recommended Action
Matplotlib guard contract	Helpers are allowed to return early whenever plt is missing—even if the caller requested file output (i.e., provided path/title). ADR-025 requires the guard to raise a runtime error in this situation so automation gets a clear failure instead of silently skipping rendering.	High	Move the early-return condition so it only triggers when both show=False and no path/title pair is supplied; otherwise always call __require_matplotlib() so missing dependencies surface as runtime errors.
Regression interval backdrop	ADR-025 specifies a translucent “generalised weight” band spanning the global interval when _plot_regression runs in interval mode. The implementation computes the bounds but leaves the fill_betweenx call commented out, so the backdrop never renders.	Medium	Restore the fill_betweenx band (matching _plot_probabilistic) and ensure it respects the computed gwl/gwh values so interval plots visually align across helpers.
Regression axis scaling	ADR-025 requires the regression helper to keep the feature-weight axis symmetric around zero by tracking extrema. The current logic simply takes the raw min/max values, so asymmetric ranges lead to off-centre plots.	Medium	After accumulating extrema, set limit = max(abs(x_min), abs(x_max)) and call ax_main.set_xlim(-limit, limit) so positive/negative contributions share equal scale.
Triangular regression margins	For non-probabilistic runs, the triangular plot should pad both axes by 10 %. When all values coincide, the code still applies a zero-width margin, leaving identical limits even though a warning is emitted; this can trigger Matplotlib “singular transformation” issues and prevents the visual separation ADR-025 mandates.	Medium	After computing the spans, detect near-zero ranges and fall back to a small fixed epsilon before applying the ±10 % padding so the axis limits always expand.
*Severity scale: High = correctness or automation blocker, Medium = visual/behavioural drift that can mislead users, Low = minor polish issue.

These items represent the most material divergences from ADR-025 and should be prioritised to keep legacy semantics stable for upcoming renderer work.

# ADR-026

## 1

Overview
ADR-026 documents how explanation plugins should be resolved, initialised, and validated at runtime, including request shaping, telemetry, and prediction-safety guarantees.

The current implementation covers most of the lifecycle expectations (resolution, metadata validation, context building, and batch validation), but a few behavioural gaps remain that warrant follow-up.

Severity scale
High – breaks a published contract or allows silent correctness failures.

Medium – feature works but omits ADR-mandated observability or resilience.

Low – cosmetic or ease-of-use differences that do not block correct behaviour.

Gap analysis
Area	ADR expectation	Observed implementation	Severity	Notes
Inclusive bounds enforcement on calibrated predictions	_PredictBridgeMonitor must ensure every (low, predict, high) triple respects low <= predict <= high, and violations must fail fast.	The monitor only records whether a bridge method was invoked, and LegacyPredictBridge forwards predictions without checking interval consistency, so malformed intervals would pass through silently.	High	Add invariant checks (and hard failures) to the monitor or bridge before returning payloads.
Telemetry coverage for interval dependencies	After accepting a batch, both metadata and telemetry should expose the resolved interval source and dependency hints.	Batch metadata receives interval_dependencies, but the telemetry payload omits that field, preventing downstream audit of the interval chain.	Medium	Include the dependency tuple in the telemetry snapshot alongside the interval source.
Normalising Mondrian bins in ExplanationRequest	Each invocation should normalise thresholds, percentiles, Mondrian bins, ignored-feature mask, and extras before handing the request to plugins.	Thresholds and percentiles are copied/tuplised, but bins is passed through untouched and stored verbatim on the request dataclass, so mutable inputs can leak through the “immutable” contract.	Low	Convert bins to an immutable copy (e.g., tuple or array) when building the request to match the ADR promise.
Compliant coverage snapshot
Plugin resolution honours overrides, environment variables, pyproject settings, descriptor fallbacks, and default identifiers before failing, matching ADR lifecycle requirements.

Runtime metadata checks enforce schema version, supported tasks/modes, and required capability tags before a plugin is accepted.

Explanation batches are validated for container/explanation inheritance, instance structure, and mutable metadata, ensuring plugins return ADR-compliant payloads.

Legacy plugins explicitly invoke the calibrated predict bridge before delegating, satisfying the ADR’s “bridge must be exercised” requirement, and interval hints are aggregated per mode for resolver reuse.

## 2

Gap Analysis for ADR-026 — Explanation Plugin Semantics
#	Area	Gap Description	Severity	Evidence	Recommended Action
1	Predict bridge safeguards	ADR-026 requires the calibrated predict bridge to reject any interval triple where low <= predict <= high does not hold, and to treat violations as hard failures. The current _PredictBridgeMonitor only records which methods were invoked before delegating to the wrapped bridge, and LegacyPredictBridge.predict simply forwards whatever the explainer returns without enforcing the invariant. This leaves the system unable to stop drift-inducing intervals from propagating.	High	Requirement: ADR-026 §2, lines 71-77.
Implementation: _PredictBridgeMonitor delegation without validation.
LegacyPredictBridge.predict returns payloads without bounds checks.	Extend the monitor (or underlying bridge) to verify every triple and raise ConfigurationError when predict lies outside [low, high], covering all bridge entry points (predict, predict_interval).
2	Explanation context immutability	ADR-026 states that the ExplanationContext handed to plugins must expose frozen, read-only handles (task metadata, helper handles, interval/plot settings). Although the dataclass is declared frozen, the runtime injects plain mutable dictionaries for helper_handles, interval_settings, and plot_settings, allowing plugins to mutate shared state across runs. This weakens the contract and can cause subtle bugs if a plugin alters the dictionaries in-place.	Medium	Requirement: ADR-026 §1, lines 48-53.
Implementation: _build_explanation_context populates standard dicts for helper_handles, interval_settings, and plot_settings before embedding them into the frozen dataclass.	Wrap each mapping in an immutable view (e.g., MappingProxyType) before constructing ExplanationContext, ensuring plugin code cannot mutate runtime-owned state.


# ADR-027

## 1

ADR-027 compliance highlights
The docs landing page leads with a combined classification and regression quickstart before introducing the audience hubs, satisfying the “Getting started first” and navigation structure mandates.

The Get Started hub calls out both quickstarts, an interval-regression signpost, and direct links to interpretation resources, aligning with the explanations-first and interpretation-emphasis rules.

Practitioner, researcher, and contributor hubs foreground calibrated workflows before optional tooling, with dedicated advanced pages that consolidate telemetry, performance, and extensibility topics as required.

Superseded architecture documents have been archived, and the unified documentation blueprint now governs ongoing work, demonstrating adoption-plan progress.

Severity scale
High: Gap undermines ADR intent or future compliance unless resolved immediately.

Medium: Gap weakens the standard for a key audience or workflow but has a contained blast radius.

Low: Minor drift; monitor but does not currently block ADR expectations.

Gaps requiring attention
Requirement (ADR-027)	Gap summary	Impact	Severity	Evidence
Rule 1 – Getting started page must link from the classification quickstart to the task API comparison and troubleshooting.	The Get Started toctree lists installation, classification/regression quickstarts, and troubleshooting, but no task API comparison link exists anywhere in the docs, so readers miss the promised side-by-side guidance.	Practitioners lose a sanctioned comparison between tasks when onboarding.	Medium	See cited lines.
Adoption plan Step 5 – Require PR reviewers to confirm classification/regression parity and interpretation-first framing.	The PR template checklist omits any parity or interpretation-first review item, leaving the governance control unenforced.	Without this guardrail, future changes can regress parity and narrative focus unnoticed.	High	See cited lines.
Rule 5 (Researcher hub) – Advanced track must include a future work ledger tied to the literature.	The researcher advanced page links to theory, telemetry, and concept guides but lacks a future-work ledger; no such ledger exists elsewhere in the docs tree.	Researchers lack a centralized roadmap for ongoing studies, reducing the hub’s completeness.	Medium	See cited lines.
Recommended follow-ups
Add a “Task API comparison” page (or section) under docs/get-started/ and link it from the classification quickstart table to fulfill Rule 1.

Update .github/pull_request_template.md (and, if needed, companion governance docs) with an explicit reviewer checkbox covering classification/regression parity plus interpretation-first framing to close Adoption Step 5.

Extend docs/researcher/advanced/index.md (or a child page) with a maintained future work ledger cross-referenced to the cited literature, keeping it updated alongside research artefacts.

## 2

Overview
The top-level documentation already emphasises parity between classification and regression quickstarts, directs readers to interpretation guidance, and organises navigation so “Get started” appears before the audience hubs in accordance with ADR‑027’s structural rules.

Practitioner content continues to point users at both quickstarts, highlights the interpretation guide, and clearly defers optional tooling to an advanced lane, aligning with the audience-first mandate.

The dedicated quickstarts demonstrate factual and alternative workflows for both classification and regression, reinforcing ADR‑027’s parity and interpretation focus.

Severity scale
High – Direct ADR‑027 rule violation that risks confusing primary audiences.

Medium – Partial coverage; core workflows exist but key ADR framing (parity, alternatives, interpretation checkpoints) is missing or incomplete.

Low – Editorial or governance follow-up; unlikely to mislead audiences but should be aligned with the new standard.

Gap analysis
Severity	Area needing attention	Details & impact
High	Foundations how-to: pipeline integration	The pipeline guide only walks through a classifier example and stops at explain_factual, leaving out probabilistic/interval regression guidance and alternative workflows. This breaks ADR‑027 rules 2 and 3 for shared foundations content that practitioners and contributors rely on.
Medium	Researcher hub landing copy	The researcher hub intro steers readers to replication and advanced threads but never reiterates the factual/alternative workflows or interpretation checkpoints that ADR‑027 mandates for every audience hub opening section.
Medium	PlotSpec how-to	Only a regression plotting recipe is documented; there is no classification example or explanation of alternative plots, so readers miss the parity and explanations-first framing when they opt into this tooling.
Medium	Export explanations how-to	All snippets focus on factual batches, omitting guidance for exporting alternative explanations or regression-specific payload nuances, undermining ADR‑027’s parity and alternative storytelling expectations.
Low	Navigation crosswalk governance note	The governance checklist still refers to “ADR‑022 information architecture,” which has been superseded; updating the reference will prevent confusion during audits of ADR‑027 adoption.
Recommended next actions
Expand the pipeline integration how-to with mirrored regression (probabilistic + interval) and alternative scenarios, and link back to the interpretation guide before optional telemetry notes.

Update the researcher hub hero section to call out factual/alternative workflows, calibrated probabilities/intervals, and interpretation touchpoints before directing readers to replication or advanced material.

Add classification and alternative plotting walkthroughs (or explicit links) to the PlotSpec guide so readers see parity and explanations-first messaging even in optional extras.

Amend the export how-to with examples for explore_alternatives, regression payloads, and reminders to couple exports with interpretation guidance.

Refresh the navigation crosswalk language to reference ADR‑027 and mark the verification checklist items that now act as ADR‑027 compliance controls.
