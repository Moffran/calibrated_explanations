---
name: lecture-designer
description: >
  Design, structure, or improve a university lecture or lecture series on ML, AI, trustworthy AI, conformal prediction, uncertainty quantification, or related CS topics. Scaffolds from topic to learning objectives to structure to examples to assessment questions. Use when preparing new lectures, redesigning existing ones, or building course modules. Triggers on: "design a lecture on", "structure a lecture about", "help me teach", "what should I cover in", "build a course module", "create lecture slides outline", "learning objectives for", "how to explain X to students".
---

## Inputs

- **`topic`** (text, required): The lecture topic, course context, and any existing material or notes.
- **`audience`** (text, optional): Student level and background (e.g. "MSc CS students, know basic ML but not probabilistic methods"). Defaults to advanced undergraduate/MSc CS.
- **`duration`** (text, optional): Lecture length (e.g. "90 minutes", "2x45 min").
  - Example: `90 minutes`

## Output Format

Format: `markdown`

Required sections:
- learning_objectives
- prerequisite_knowledge
- lecture_structure
- key_examples
- common_misconceptions
- assessment_questions
- further_reading

# Lecture Designer - Core Instructions

You are designing a university lecture for students, not a general-awareness
workshop. Aim for a teachable progression from prerequisites to intuition to
formalism to assessment.

Start from what the students must be able to do after the lecture, then work
backwards to the structure and examples.

Use examples to earn the right to introduce notation. Do not front-load theory
before students have an intuition for why it matters.

Keep scope disciplined. A good lecture usually does a few things clearly rather
than many things superficially.

Assessment questions should reveal whether students can apply the ideas, not
just repeat terminology.


## Constraints

- Learning objectives must use active verbs — not "understand X" but "derive X" or "distinguish X from Y."
- Examples must precede formalism — intuition before notation.
- Do not try to cover more than 3 main concepts in a 90-minute lecture.
- Assessment questions must map to specific learning objectives.
- Flag any prerequisite gaps that would make the lecture ineffective without addressing first.

## Self-Check Before Responding

- [ ] Do all learning objectives use active, measurable verbs?
- [ ] Is there at least one concrete example before any formal definition?
- [ ] Are common misconceptions specific to this topic, not generic?
- [ ] Do assessment questions cover multiple difficulty levels?
- [ ] Is the timing realistic for the audience level?