# Research Context and Locked Claims

## Defensible novelty

Do **not** claim:
- violence detection is new;
- crowd anomaly detection is new;
- combining a fight detector, crowd detector, and notifications is inherently novel;
- no commercial system performs similar fusion.

Claim:

> The project explicitly defines, implements, and reproducibly evaluates incident-level temporal fusion of person-level violence and surrounding crowd-response signals, then converts the fused evidence into a persistent incident with severity, deduplication, evidence, and human review.

## Research contribution categories

1. Unified incident representation.
2. Temporal fusion and incident-state mechanism.
3. Multi-signal severity reasoning.
4. Evidence-backed alert lifecycle.
5. Operational evaluation.

## Terminology

Preferred:
- crowd-safety risk;
- crowd-crush/stampede risk indicators;
- observable precursors;
- early warning;
- escalating incident.

Avoid:
- guaranteed stampede prediction;
- predicting future stampedes with certainty.

## Source materials used to define this pack

The project was previously consolidated in:
- `Capstone_Crowd_Safety_Project_Master_Document`
- `Capstone_Preliminary_Report_Chapters_1_to_3`
- three prior deep-research reports on violence/crowd anomaly CCTV systems
- project review slide deck

These remain academic/reference sources. This handoff pack converts the locked decisions into implementation requirements; it is not a replacement literature review.
