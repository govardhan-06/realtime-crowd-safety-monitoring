# Milestone 4 — Temporal Fusion and Incident Engine

## Objective
Implement the project's core research contribution.

## Required work
- common timestamp-aligned signal schema;
- rolling temporal windows;
- smoothing;
- configurable feature normalisation;
- spatial association/ROI logic;
- persistence;
- fused risk;
- reason codes;
- incident lifecycle;
- hysteresis/decay;
- severity;
- deduplication;
- state-transition log;
- baseline modes:
  - violence-only;
  - crowd-only;
  - naive OR;
  - simple rule fusion.

## Critical behavior tests
- one-frame spike does not create repeated incidents;
- persistent violence can create an incident;
- crowd-only severe conditions can create an incident;
- violence + nearby crowd response raises severity;
- benign brief convergence can decay without alert;
- one continuous event remains one incident;
- resolving event closes only after configured quiet period;
- state transitions are deterministic.

## Acceptance criteria
- All thresholds/weights are config-driven.
- Every escalation has reason codes.
- Raw signal values remain available for evaluation.
- Same signal stream yields same incident stream.
- Baseline and proposed strategies can be run on the same stored feature stream.
