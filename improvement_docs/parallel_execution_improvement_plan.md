<!-- markdownlint-disable-next-line MD041 -->
> **Status note (2025-11-04):** Last edited 2025-11-04 · Archive after: Replace once ADR-004 remediation completes (target v0.10.0) · Implementation window: v0.9.0 runtime polish → v0.10.0 runtime realignment.

Last updated: 2025-11-04

# Parallel Execution Improvement Plan

## Goal

Deliver the runtime polish and ADR-004 remediation items called out in Release Plan §v0.9.0 (item 11) and §v0.10.0 (item 4) by reducing parallel overhead, expanding configuration surfaces, and enabling workload-aware strategy selection.

## Release Plan alignment

| Phase | Release target | Release Plan reference | Alignment summary |
| --- | --- | --- | --- |
| Phase 0 | v0.9.0 runtime polish | [§v0.9.0 · item 11](RELEASE_PLAN_V1.md#v090-documentation-realignment--targeted-runtime-polish) | Establishes documentation owners and telemetry schema groundwork required before toggles ship. |
| Phase 1 | v0.9.0 runtime polish | [§v0.9.0 · item 11](RELEASE_PLAN_V1.md#v090-documentation-realignment--targeted-runtime-polish) | Defines chunk-size and configuration knobs promised as opt-in runtime controls. |
| Phase 2 | v0.10.0 runtime realignment | [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment) | Addresses ADR-004 backlog for payload sharing, batching, and lifecycle management. |
| Phase 3 | v0.10.0 runtime realignment | [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment) | Delivers workload-aware strategy selection and adaptive gating called out in the ADR remediation line item. |
| Phase 4 | v0.10.0 runtime realignment | [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment) | Fulfils the testing and benchmarking commitments for ADR-004 prior to release gate review. |
| Phase 5 | v0.10.0 release prep | [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment) & [§v1.0.0-rc · item 5](RELEASE_PLAN_V1.md#v100-rc-release-candidate-readiness) | Updates user guidance, changelog, and telemetry artefacts required during the release candidate checklist. |

## Phase 0 – Foundations (Week 0–1)

> Release Plan alignment: [§v0.9.0 · item 11](RELEASE_PLAN_V1.md#v090-documentation-realignment--targeted-runtime-polish)

- **Task P0.1 – Align documentation and owners**
  - Deliverables: add practitioner playbook coverage (`docs/practitioner/advanced/parallel_execution_playbook.md`).
  - Dependencies: none.
  - Success metric: guidelines referenced in documentation.
- **Task P0.2 – Telemetry design spike**
  - Deliverables: design doc for executor telemetry schema (timings, utilisation, fallback counters) consistent with ADR-004.
  - Dependencies: telemetry governance guidelines from Release Plan §v0.8.0 (telemetry concept page).
  - Success metric: approved design with schema fields and collection points identified.

## Phase 1 – Configuration Surface (Week 2–4)

> Release Plan alignment: [§v0.9.0 · item 11](RELEASE_PLAN_V1.md#v090-documentation-realignment--targeted-runtime-polish)

- **Task P1.1 – Decouple chunk size knobs**
  - Deliverables: proposal + implementation plan to introduce `instance_chunk_size` and optional `feature_chunk_size` while keeping `min_batch_size` as the gating threshold.
  - Dependencies: unit tests covering `InstanceParallelExplainPlugin` chunk assembly.
  - Success metric: design approved; backlog ticket created with acceptance criteria (chunk-size override, backward compatibility, feature batching optional).
- **Task P1.2 – Extend `ParallelConfig` options**
  - Deliverables: specification for `task_size_hint_bytes`, `force_serial_on_failure`, backend preference flags (per ADR-004).
  - Dependencies: ADR-004 decision, Release Plan §v0.10.0 #4 configuration commitments.
  - Success metric: configuration options documented with migration notes and telemetry hooks defined.

## Phase 2 – Executor & Plugin Refactor (Week 5–8)

> Release Plan alignment: [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment)

- **Task P2.1 – Share heavy payloads**
  - Deliverables: design for moving immutable arrays (`perturbed_feature`, rule boundaries, baseline predictions) to shared/global state to minimise pickling.
  - Dependencies: feature task builder invariants, cache reset hooks (ADR-003).
  - Success metric: prototype showing ≥40% reduction in task payload size measured via pickle dumps.
- **Task P2.2 – Batch feature tasks**
  - Deliverables: plan for optional feature batching (configurable chunk size) with deterministic result ordering.
  - Dependencies: Task P1.1 (feature chunk size knob), tests ensuring merge order stability.
  - Success metric: benchmark showing overhead reduction on D ≥ 256 scenarios without accuracy drift.
- **Task P2.3 – Executor context manager & cancellation**
  - Deliverables: implement context management, cooperative cancellation, and resource cleanup per ADR-004.
  - Dependencies: telemetry design (P0.2) for instrumentation integration.
  - Success metric: integration tests covering `with ParallelExecutor(...)` usage and forced cancellation path.

## Phase 3 – Workload-aware Strategy (Week 9–11)

> Release Plan alignment: [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment)

- **Task P3.1 – Auto-strategy heuristics**
  - Deliverables: workload estimator using `task_size_hint_bytes`, dataset shape, platform to choose strategy (threads/processes/joblib) automatically.
  - Dependencies: Phase 1 hooks, telemetry instrumentation to validate choices.
  - Success metric: heuristic achieves ≥1.3× speedup vs current auto on benchmark suite in `evaluation/scripts/parallel_ablation.py`.
- **Task P3.2 – Dynamic gating & chunk sizing**
  - Deliverables: runtime adjustments for chunk sizes based on telemetry (e.g., adaptively expand when work items per worker low).
  - Dependencies: Task P3.1 outputs, telemetry pipeline.
  - Success metric: ablation suite demonstrates no regressions for small workloads and improved scaling on large workloads.

## Phase 4 – Testing & Benchmarking (Week 12–14)

> Release Plan alignment: [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment)

- **Task P4.1 – Spawn lifecycle tests (Windows + Linux)**
  - Deliverables: pytest suite covering thread/process/joblib strategies under spawn, fork, and nested parallelism constraints.
  - Dependencies: CI updates, fixture support for Windows spawn.
  - Success metric: passing CI on Windows runners; failures block merge per Release Plan §v0.10.0 gates.
- **Task P4.2 – Benchmark automation**
  - Deliverables: integrate `evaluation/scripts/parallel_ablation.py` into CI/perf dashboards with baseline comparisons and regression thresholds.
  - Dependencies: telemetry pipeline + instrumentation.
  - Success metric: automated report attached to PR checks with alert thresholds (<1.1× regression vs baseline flagged).

## Phase 5 – Rollout & Documentation (Week 15–16)

> Release Plan alignment: [§v0.10.0 · item 4](RELEASE_PLAN_V1.md#v0100-runtime-boundary-realignment) · [§v1.0.0-rc · item 5](RELEASE_PLAN_V1.md#v100-rc-release-candidate-readiness)

- **Task P5.1 – Update user guidance**
  - Deliverables: refresh docs (README performance appendix, practitioner guide callouts) reflecting new heuristics and defaults.
  - Dependencies: practitioner playbook (`docs/practitioner/advanced/parallel_execution_playbook.md`) and telemetry results.
  - Success metric: docs mention strategy auto-selection, chunk knobs, and provide upgrade notes.
- **Task P5.2 – Release notes & changelog**
  - Deliverables: draft release notes for v0.10.0 summarising parallel improvements, migration steps, and telemetry fields.
  - Dependencies: completed implementation tasks.
  - Success metric: changelog entry referencing ADR-004 closure and linking to updated docs.

## Risks & Mitigations

- **Risk**: Increased shared state complexity when moving payloads to globals.
  - Mitigation: integrate with cache reset hooks (`CalibratorCache.forksafe_reset`) and add regression tests for sequential correctness.
- **Risk**: Auto-strategy heuristics misclassifying workloads.
  - Mitigation: gate rollout behind environment variable (default to manual) until telemetry confidence achieved.
- **Risk**: CI resource constraints for spawn tests.
  - Mitigation: add nightly job for heavy spawn coverage; keep smoke tests lightweight on PRs.

## Tracking

- Link tasks P0–P5 into the Release Plan status table (v0.9.0 item 11, v0.10.0 item 4).
- Surface progress in sprint reviews; use telemetry dashboards once available to validate improvements.
- Retire this document once ADR-004 moves to “Implemented” and guidelines are updated with final defaults.
