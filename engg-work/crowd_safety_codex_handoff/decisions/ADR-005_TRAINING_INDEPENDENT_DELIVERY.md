# ADR-005 — Training-Independent End-to-End Delivery

## Status
Accepted.

## Context

The violence-recognition branch is the highest-risk component because dataset preparation, GPU availability, transfer learning, threshold calibration, and cross-dataset generalisation can all delay the project.

Making M4-M6 depend on a successful fine-tuning run would create an avoidable project-delivery risk.

## Decision

Split Milestone 3 into:

### M3A — delivery baseline

Integrate a ready-made binary temporal-video violence checkpoint behind a generic adapter.

M3A exists to:
- validate clip sampling;
- validate timestamp alignment;
- create `ViolenceEvidence`;
- integrate the full incident pipeline;
- unblock M4-M6.

### M3B — academic training experiment

Perform bounded transfer learning using X3D-S:

1. pretrained backbone;
2. replace binary classification head;
3. freeze backbone;
4. train head;
5. evaluate;
6. partially unfreeze later block(s) only if validation justifies it.

## Consequences

- M4 may use M3A evidence while M3B is still being improved.
- M3B results are reported separately.
- M3A is not automatically considered scientifically superior because it is a transformer or externally fine-tuned checkpoint.
- Any external checkpoint must have recorded provenance, revision, license, label mapping, and locally measured performance.
- The generic violence adapter must allow swapping M3A and M3B without changing the fusion engine.