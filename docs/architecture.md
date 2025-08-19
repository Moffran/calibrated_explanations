# Architecture Overview

This page summarizes the high-level component and data flow for the library. See `improvement_docs/component_diagram.md` for the development version; this page is the user-facing snapshot.

## Explain Flow

```mermaid
flowchart LR
    subgraph EXT[External]
        D[Training Data]
        M[Trained Model]
    end
    subgraph CE[calibrated_explanations]
        CFG[Config & Validation]
        CAL[Calibration Orchestrator]
        STRAT_CAL[(Calibration Strategies)]
        CACHE[(Cache)]
        PAR[Parallel Executor]
        ART[Calibrated Artifacts]
        EXP[Explanation Manager]
        STRAT_EXP[(Explanation Strategies)]
        SCHEMA[Schema & Envelope]
        PLOT[PlotSpec Generator]
        RENDER[Renderers]
        PLUG[Plugin Registry]
        MET[Metrics]
        LOG[Logging]
    end
    D --> M
    M --> CFG
    D --> CFG
    CFG --> CAL --> STRAT_CAL --> ART --> EXP --> STRAT_EXP --> SCHEMA
    STRAT_CAL --> PAR
    STRAT_EXP --> PAR
    STRAT_EXP --> CACHE
    STRAT_CAL --> CACHE
    SCHEMA --> PLOT --> RENDER
    PLUG --> STRAT_CAL
    PLUG --> STRAT_EXP
    CACHE -. stats .-> MET
    PAR -. timings .-> MET
    CAL -. logs .-> LOG
    EXP -. logs .-> LOG
```

## Layers

```mermaid
flowchart TB
    UI[Public API]
    ORCH[Orchestrators]
    STRATS[Strategies]
    SERVICES[Support Services\n(Cache, Parallel, Schema, Plugins, Validation)]
    EXT[External Model & Data]
    UI --> ORCH --> STRATS --> SERVICES --> EXT
```

## Component Roles

| Component | Role |
|-----------|------|
| Config & Validation | Normalize inputs, enforce constraints, structured errors |
| Calibration Orchestrator | Run selected calibration strategy, produce calibrated sets/intervals |
| Explanation Manager | Produce explanation objects using calibrated artifacts |
| Cache | Reuse deterministic intermediates; bounded memory footprint |
| Parallel Executor | Strategy-agnostic concurrency selection |
| Schema & Envelope | Versioned JSON contract for portable explanations |
| PlotSpec / Renderers | Backend-agnostic visualization generation |
| Plugin Registry | Controlled discovery & trust gating for extensions |
| Metrics & Logging | Observability (latency, hit ratio, warnings) |

## Evolution

Initial phases keep cache disabled & serial execution by default to preserve baseline behavior. Later phases progressively enable heuristics and pluggable backends.
