# Milestone 4 — Temporal Fusion and Incident Engine

## Objective

Implement the project's core research contribution.

## Required work

- common timestamp-aligned signal schema;
- explicit signal/stage availability state;
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

## Signal-health rule

Missing/unavailable evidence is not normal evidence.

Examples:

```text
violence_status = unavailable
```

must not be silently transformed into:

```text
violence_score = 0
```

Fusion behavior under degraded signals must be explicit, deterministic, configurable, and testable.

## Critical behavior tests

- one-frame spike does not create repeated incidents;
- persistent violence can create an incident;
- crowd-only severe conditions can create an incident;
- violence + nearby crowd response raises severity;
- benign brief convergence can decay without alert;
- one continuous event remains one incident;
- resolving event closes only after configured quiet period;
- state transitions are deterministic;
- unavailable violence branch does not masquerade as negative violence evidence;
- crowd-only risk can still operate when the violence branch is unavailable if policy permits;
- same stored signal stream always yields the same incident stream.

## Acceptance criteria

- All thresholds/weights are config-driven.
- Every escalation has deterministic reason codes.
- Raw signal values and signal health remain available for evaluation.
- Same signal stream yields same incident stream.
- Baseline and proposed strategies can be run on the same stored feature stream.
- No VLM/generated explanation is consumed by fusion or state transitions.