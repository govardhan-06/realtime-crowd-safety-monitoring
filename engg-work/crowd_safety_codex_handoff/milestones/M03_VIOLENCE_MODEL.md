# Milestone 3 — Temporal Violence Model

## Objective
Produce a reproducible fine-tuned temporal model and integrate rolling violence evidence.

## Default model
X3D-S pretrained backbone with binary violent/non-violent classification head.

## Required work
- dataset manifest schema;
- dataset adapter(s);
- preprocessing;
- clip sampler;
- train/val/test split tooling;
- training config;
- checkpoint metadata;
- validation metrics;
- threshold calibration;
- inference adapter;
- rolling clip buffer integration;
- per-window violence evidence output.

## Experiments
Start simple:
1. frozen backbone + trained head;
2. partial unfreeze if needed;
3. only then broader fine-tuning.

Do not start with large transformer comparison.

## Acceptance criteria
- One reproducible training command.
- Train/validation/test leakage protections documented.
- Model checkpoint records config and label mapping.
- Held-out metrics generated.
- Offline pipeline produces timestamp-aligned scores.
- CPU fallback is supported for tests even if training/inference optimisation uses GPU.
