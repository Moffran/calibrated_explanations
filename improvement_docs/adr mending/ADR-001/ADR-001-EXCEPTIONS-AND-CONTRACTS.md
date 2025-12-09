# ADR-001: Exceptions and Contracts

## Status
**Active** - Defines the allowed architectural exceptions for v0.10.0.

## Context
ADR-001 defines a strict layered architecture. However, practical implementation requires specific exceptions to allow for:
1. Shared exception hierarchies (ADR-002).
2. Orchestration patterns where higher-level components coordinate lower-level ones.
3. Interface definitions where plugins depend on core contracts.

## Allowed Exceptions (The "Allow-list")

### 1. Exception Hierarchy
**Rule:** Any package may import from `calibrated_explanations.core.exceptions`.
**Rationale:** We use a centralized exception hierarchy rooted in `CalibratedExplanationsError`. All components must be able to raise these standard exceptions.
**Constraint:** Imports must be strictly from the `exceptions` module, not general `core` logic.

### 2. Orchestrator Pattern
**Rule:** `explanations` package may import from `calibration` and `core`.
**Rationale:** The `explanations` package acts as the primary orchestrator (Facade) for the library. It needs to instantiate calibration models and core data structures to generate explanations.
**Constraint:** This is a one-way dependency. `core` and `calibration` must NEVER import from `explanations`.

### 3. Interface/Protocol Definition
**Rule:** `plugins` may import from `calibrated_explanations.core.interfaces`.
**Rationale:** The `core` package defines the protocols (Abstract Base Classes) that plugins must implement.
**Constraint:** Plugins should only depend on the interfaces, not concrete implementations in core.

### 4. Visualization Layer
**Rule:** `viz` may import from `explanations` and `core`.
**Rationale:** Visualization components need to inspect the objects they are rendering. They are "sinks" in the dependency graph—nothing depends on `viz`.

### 5. Legacy Compatibility
**Rule:** `legacy` package is exempt from strict boundary checks.
**Rationale:** To facilitate gradual migration, legacy code is isolated but allowed to depend on other parts of the system until it is refactored or removed in v2.0.

### 6. CalibratedExplainer Orchestration
**Rule:** `core` (specifically the CalibratedExplainer orchestrator) may import from `calibration`, `plugins`, `explanations`, `cache`, `parallel`, `integrations`, and `api`.
**Rationale:** The orchestrator wires together calibration, cache, parallel execution, plugin discovery, and API validation at runtime. These imports remain constrained to the orchestrator so that other `core` modules stay boundary-pure.

### 7. Shared Services
**Rule:** `calibration` and `parallel` may import from `cache`.
**Rationale:** Cache is treated as a shared service (ADR-003) consumed by calibration workflows and parallel execution to avoid redundant computation.

### 8. Plugin Adapter Bridge
**Rule:** `plugins` may import from `explanations`, `viz`, and `calibration` while the ADR-015 plugin bridge stabilises.
**Rationale:** In-tree plugin adapters delegate to the legacy explanations/viz implementations to preserve behaviour while the dedicated plugin contracts harden. These imports are scoped to the adapters and will be revisited when the ADR-015 runtime path is declared stable.

### 9. Visualization Hooks from Explanations
**Rule:** `explanations` may import from `viz`.
**Rationale:** Explanation payloads include optional narrative hooks that render via the visualization layer. Imports are limited to the adapter shims used for rendering.

### 10. Plugin Discovery from Explanations
**Rule:** `explanations` may import from `plugins`.
**Rationale:** Explanation orchestrators perform plugin discovery when resolving custom explanation strategies. These imports remain within orchestrator modules to avoid broad coupling.

## Enforcement
These exceptions are encoded in `scripts/check_import_graph.py` in the `BoundaryConfig.allowed_cross_sibling` dictionary.
Any import violating ADR-001 that is NOT in this list will cause the CI build to fail.
