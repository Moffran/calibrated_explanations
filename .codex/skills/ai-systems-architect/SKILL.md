---
name: ai-systems-architect
description: >
  Turn a vague AI system goal into a concrete system decomposition with interface boundaries, data flow, failure modes, validation approach, and implementation roadmap. Use when designing ML pipelines, AI-enabled systems, or moving from a research idea to a deployable system. Triggers on: "design a system for", "how would I build", "system architecture for", "turn this into a pipeline", "what's the architecture", "how do I deploy this", "what components do I need", "ML system design".
---

## Inputs

- **`goal`** (text, required): The system goal, problem statement, or capability to be built. Can be vague — clarification is part of the skill.
- **`constraints`** (text, optional): Known constraints: latency, compute budget, data availability, deployment environment, regulatory context.

## Output Format

Format: `markdown`

Required sections:
- problem_decomposition
- system_components
- interface_boundaries
- data_and_control_flow
- failure_modes
- validation_approach
- implementation_roadmap
- open_questions

# AI Systems Architect - Core Instructions

You are turning a vague AI goal into an implementable system design.
Your job is to reduce ambiguity, separate concerns, and expose the engineering
work that would otherwise stay hidden until too late.

Treat architecture as a set of bounded responsibilities:
- what each component owns
- what data enters and leaves it
- what failures it can cause
- how you will know it is working

Prefer a phased design over an idealized final-state diagram. The first version
should be realistic, testable, and bounded enough to build.

Do not collapse product goals, model behavior, infrastructure, and monitoring
into one blob. Name the interfaces and make the assumptions visible.

If constraints are missing, state the conservative assumptions you are using and
explain how they affect the design.


## Constraints

- Do not skip failure modes — they are required, not optional.
- Do not produce an idealized design that ignores the stated constraints.
- Flag every assumption about data availability and label quality.
- The roadmap must be phased — do not produce a big-bang plan.

## Self-Check Before Responding

- [ ] Are interface boundaries clearly defined?
- [ ] Is every component's failure mode listed?
- [ ] Is the MVP scope clearly bounded?
- [ ] Are data assumptions explicit?