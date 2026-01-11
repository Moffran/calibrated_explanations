> **Status note (2026-01-06):** Last edited 2026-01-06 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-029 — Reject Integration Strategy

Status: Accepted

Date: 2026-01-06

Authors: Core maintainers

Supersedes: N/A

Related: ADR-006-plugin-registry-trust-model, ADR-013-interval-calibrator-plugin-strategy, ADR-014-plot-plugin-strategy, ADR-015-explanation-plugin, ADR-020-legacy-user-api-stability

## Context

The reject learner exists as a dedicated orchestration layer and can be invoked manually via `CalibratedExplainer.initialize_reject_learner(...)` and `CalibratedExplainer.predict_reject(...)`. The current default behavior is **no reject by default**, with optional initialization when `reject=True`. This default must remain intact.

We need to integrate reject in a way that:

- Preserves open-source parity and existing public API behavior.
- Enables optional, policy-driven integration into predict/explain workflows.
- Supports future alternative reject strategies.
- Provides a pathway for reject visualization and governance-friendly outputs.

This ADR captures decisions and open questions across four areas:

1. Invocation model (manual default).
2. Strategy extensibility (how to support multiple reject strategies).
3. Output integration (if any).
4. Visualization (plot plugin integration).

## Decision

### Decision 1 — Invocation model

**Decision:** Adopt A3. Use a **Policy Enum** named `RejectPolicy` for reject integration.

**Rationale:** A policy enum cleanly separates *how/when* reject is applied from *which* reject strategy is used. It provides explicit behavior, is more discoverable than booleans, and makes future invocation plugins easier to add without breaking defaults.

### Decision 2 — Strategy extensibility

**Decision:** Adopt B2 only. Use a lightweight registry inside
`RejectOrchestrator` for reject strategies.

### Decision 3 — Output integration

**Decision:** Adopt C3. Use a **Structured Wrapper/Envelope** for reject-aware outputs.

**Rationale:** When a reject policy is active, the output should be explicitly structured to contain both the explanation and the rejection status. This is governance-friendly, versionable, and avoids schema ambiguity allowing users to strictly type-check returns.

### Decision 4 — Visualization

**Decision:** No visualization integration (D1).

## Open Questions

### Invocation model
- What are the specific enum members and their semantics?
- How does the policy interact with legacy `reject=True` initialization?

Defined `RejectPolicy` members (implemented in `core.reject.policy`):

- `NONE` — Default. No reject integration; legacy behaviour unchanged.
- `PREDICT_AND_FLAG` — Always run prediction; include a `rejected` flag in the `RejectResult` envelope but do not change whether an explanation is produced.
- `EXPLAIN_ALL` — Always produce explanations for all instances; include reject status in the envelope when applicable.
- `EXPLAIN_REJECTS` — Always predict; produce explanations only for instances that are rejected (useful for audit-focused workflows).
- `EXPLAIN_NON_REJECTS` — Always predict; produce explanations only for non-rejected instances (useful to avoid explaining decisions you won't act on).
- `SKIP_ON_REJECT` — If an instance is rejected, skip prediction and explanation (short-circuit); otherwise proceed. Useful for strict gating where further computation should be avoided.

These members cover common integration patterns; additional policies may be added later (e.g., sampling-based explanation policies, delayed-explain modes, or hybrid policies).

### Strategy extensibility
- What lifecycle hooks are required for registry-managed strategies (init, fit, update, invalidate)?
- How should strategies be selected and configured (per explainer, per call, or config-driven)?
- What audit/telemetry metadata should be emitted for strategy selection?

### Output integration
- What is the precise schema of the envelope (e.g., `RejectedExplanation` class)?
- How does this affect existing downstream consumers like plotters?

### Visualization
- If visualization is revisited, should it be a dedicated plugin or an overlay?

## Alternatives Considered

### 1) Invocation model alternatives

**A1 — Manual only (status quo)**
- Pros: Zero contract change; safest behavior.
- Cons: Requires manual wiring; easy to misuse or forget.

**A2 — Simple boolean opt-in**
- Pros: Minimal configuration; simple UX.
- Cons: Limited control over behavior; unclear semantics.

**A3 — Policy enum**
- Pros: Explicit behavior; flexible integration paths.
- Cons: Slightly larger API surface.

**A4 — Dedicated wrapper method (e.g., `explain_with_reject`)**
- Pros: No changes to existing methods; explicit invocation.
- Cons: API proliferation; two ways to do the same thing.

**A5 — Global config opt-in**
- Pros: Centralized control for deployments.
- Cons: Hidden behavior; can surprise users.

**Recommendation:** A3.

### 2) Strategy extensibility alternatives

**B1 — Single built-in strategy only**
- Pros: Lowest complexity.
- Cons: Blocks alternative strategies.

**B2 — Reject strategy registry inside `RejectOrchestrator`**
- Pros: Lightweight; local to reject subsystem; easy to extend.
- Cons: Less uniform than full plugin chains.

**B3 — Full plugin chain for reject strategies**
- Pros: Consistent with plugin architecture; supports fallback chains.
- Cons: More setup and infrastructure.

**B4 — Factory injection via kwargs**
- Pros: Flexible for advanced users.
- Cons: No discoverability; inconsistent usage.

**B5 — External callback hooks**
- Pros: Maximum flexibility.
- Cons: Hard to standardize and support.

**Recommendation:** Adopt B2.

### 3) Output integration alternatives

**C1 — No integration (manual only)**
- Pros: No contract changes.
- Cons: Fragmented UX.

**C2 — Optional metadata fields on explanation objects**
- Pros: Minimal wrapper; low friction.
- Cons: Schema changes can be subtle and hard to version.

**C3 — Structured wrapper/envelope**
- Pros: Explicit schema; governance-friendly; versionable.
- Cons: Requires opt-in handling; changes return type when enabled.

**C4 — Separate reject report API**
- Pros: Keeps explanation output unchanged.
- Cons: Multi-step UX.

**Recommendation:** C3.

### 4) Visualization alternatives

**D1 — No visualization**
- Pros: No extra work.
- Cons: Users must build custom plots.

**D2 — Dedicated reject plotting plugin**
- Pros: Modular; consistent with plot plugin strategy.
- Cons: Requires plugin registration and documentation.

**D3 — Extend existing plot workflows**
- Pros: No new plugin types.
- Cons: Risk of cluttered plots or implicit behavior.

**D4 — Helper function (non-plugin)**
- Pros: Simple; low overhead.
- Cons: Inconsistent with plugin architecture.

**Recommendation:** D1.

## Consequences

- **Invocation:** Users will use an explanation policy enum to opt-in to reject integration. Default remains "no reject".
- **Output:** When reject is enabled via policy, the return type will be a structured envelope (e.g., `RejectResult` or similar) containing explanation and status.
- **Strategy:** Extensibility is handled via a lightweight registry within the `RejectOrchestrator`.
- **Visualization:** No changes to visualization; users handle reject visualization manually if needed.

## Implementation plan (recommended)

1. Publish the clarified ADR (this document) and cross-link it from `RELEASE_PLAN_v1.md` and v0.10.2 release documentation.
2. Define `RejectPolicy` enum in `src/calibrated_explanations/core/reject.py` (or a closely related module) with the initial members listed above and docstrings explaining semantics.
3. Define a `RejectResult` envelope dataclass in `src/calibrated_explanations/explanations/reject.py` that contains at minimum: `prediction` (optional), `explanation` (optional), `rejected: bool`, `policy: RejectPolicy`, and `metadata`.
4. Add a `RejectOrchestrator` component (new `core/prediction/reject_orchestrator.py`) that:
	- Hosts a lightweight registry of reject strategies (B2).
	- Exposes a `resolve_strategy(identifier_or_none)` method and lifecycle hooks (`init`, `fit`, `invalidate`).
	- Provides a `apply_policy(policy, inputs, explainer_handle, **kwargs)` entry point that returns a `RejectResult`.
5. Wire the `RejectOrchestrator` into the existing prediction/explanation orchestrator: add an optional `reject_policy: RejectPolicy = RejectPolicy.NONE` parameter to the explanation entry point(s) and have the orchestrator delegate to `RejectOrchestrator.apply_policy(...)` when `policy != NONE`.

	- Allow `reject_policy` to be specified per call (e.g., `explainer.explain(..., reject_policy=...)`) so invocation semantics can vary for the same explainer instance.
	- Also support an explainer-level default: allow setting a persistent `reject_policy` via the explainer constructor or during `calibrate(...)` (e.g., `explainer.calibrate(..., default_reject_policy=...)`). Per-call parameters should override the explainer-level default.

		- Important constraint for wrappers: `WrapCalibratedExplainer.__init__` MUST NOT accept `default_reject_policy` as a constructor argument. Wrapper-level defaults should be provided only through `calibrate(...)` (the wrapper may pass the value through to the underlying `CalibratedExplainer`). This enforces a single public entrypoint for explainer-level defaults and avoids surprising state during wrapper construction.

		- Operational rule: Selecting any non-`NONE` `RejectPolicy` MUST implicitly enable the reject orchestration (equivalent to `reject=True`) for the affected call(s) or explainer-level default. In practice this means that supplying `reject_policy != RejectPolicy.NONE` (per-call or via explainer default) triggers `RejectOrchestrator` initialization and evaluation even if the legacy `reject` flag was `False`.
6. Implement a default built-in reject strategy (the current behaviour) that preserves existing semantics and is registered under the orchestrator's registry as `builtin.default`.
7. Add CI tests that exercise each `RejectPolicy` member, ensuring return schemas and interactions with existing explanation consumers remain backward compatible when `NONE` is used.
8. Update documentation and examples: `docs/improvement/` release plan, add `docs/reject.md`, and user-facing examples demonstrating the common policies.
9. Defer visualization integration until a separate ADR or plugin is proposed.

## Verification checklist (recommended additions)

- `RejectPolicy` is implemented and documented in `src/`.
- `RejectResult` envelope type exists and is used when non-`NONE` policies are selected.
- `RejectOrchestrator` registry contains `builtin.default` strategy and can be extended via registry APIs.
- Orchestrator entry points accept a `reject_policy` parameter and default to `NONE`.
- Unit tests cover the schema and behavior for each policy and assert legacy behaviour remains unchanged when `NONE` is used.
- ADR-029 is published and linked from the release plan.

## Future Considerations

- Define registry interfaces, lifecycle hooks, and strategy selection conventions.
- Design the `RejectPolicy` enum and the structured output envelope class.
