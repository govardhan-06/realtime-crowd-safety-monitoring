# Milestone 3 — Temporal Violence Evidence

## Objective

Produce timestamp-aligned violence evidence without making project delivery depend on successful fine-tuning, then perform one bounded transfer-learning experiment for the academic component.

Milestone 3 has two sub-milestones:

- **M3A — Pretrained Violence Baseline**
- **M3B — X3D-S Transfer-Learning Experiment**

M3A is required for the normal integration path.

M3B is an important experiment but must not block M4-M6.

---

# M3A — Pretrained Violence Baseline

## Goal

Integrate a ready-made binary temporal-video classifier behind a generic adapter so the complete incident pipeline can be developed before project-specific training is complete.

## Active checkpoint

Use the verified pretrained X3D-M checkpoint:

```text
repository: visionlab-ai/school-violence-detection-models
checkpoint: final/final_x3d_realtime.pt
revision: a744b6af7496f0cbfa4f0ba32acd46b65e52d4e1
sha256: e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349
architecture: x3d_m
input: 16 RGB frames, 224x224, mean=0.45, std=0.225
labels: non-violent / violent
threshold: 0.4 (starting value)
license: MIT
```

The checkpoint's model-card metrics are not project results; the project threshold is evaluated on reviewed clips in Colab.

This is a community checkpoint. Treat it as a development baseline, not as ground truth.

If it is incompatible or obviously unsuitable, replace it behind the same adapter rather than redesigning downstream code.

## Required work

- generic temporal-video classification interface;
- X3D-M adapter for the initial checkpoint;
- checkpoint/revision/config metadata;
- label mapping validation;
- clip sampler;
- rolling clip buffer integration;
- configurable inference cadence;
- timestamp alignment;
- per-window violence evidence output;
- threshold configuration/calibration on project dev clips;
- inference latency measurement;
- explicit status:
  - `available`
  - `degraded`
  - `unavailable`
- CPU fallback or mock/stub path for automated tests.

## Critical semantic rule

```text
model unavailable != violence_score 0
```

When inference fails or is unavailable, emit explicit unavailable evidence/status.

## M3A evaluation

Use a small local development suite containing:
- obvious violence;
- obvious non-violence;
- benign running/high motion;
- dense crowds;
- at least one longer surveillance-like clip.

Measure locally:
- basic precision/recall/F1 if labels are sufficient;
- score distributions;
- false-positive examples;
- inference latency.

Do not copy external model-card metrics into project results.

## M3A acceptance criteria

- one command/config path runs the pretrained model on an MP4;
- pipeline produces timestamp-aligned `ViolenceEvidence`;
- checkpoint + revision + label mapping are traceable;
- model output is converted into a stable generic contract;
- unavailable inference is represented explicitly;
- at least positive and negative development clips are inspected;
- downstream M4 can consume the evidence without knowing the model architecture.

---

# M3B — X3D-S Transfer-Learning Experiment

## Goal

Create one reproducible project-owned transfer-learning experiment for violent vs non-violent temporal classification.

## Default model

X3D-S pretrained on a large action-recognition corpus with a binary classification head.

Do not train from scratch.

## Required work

- dataset manifest schema;
- dataset adapter(s);
- lawful/allowed dataset acquisition documentation;
- preprocessing;
- clip sampler;
- leakage-safe train/validation/test split tooling;
- training config;
- checkpoint metadata;
- validation metrics;
- threshold calibration;
- X3D inference adapter implementing the same generic violence interface as M3A;
- comparison against M3A on a common held-out evaluation subset.

## Training sequence

### Experiment 1 — required

1. load pretrained X3D-S;
2. replace final classification head;
3. freeze the backbone;
4. train the binary head;
5. evaluate on validation and held-out data.

### Experiment 2 — optional/gated

Only if Experiment 1 is clearly insufficient and compute/time permits:

1. unfreeze the final X3D block(s);
2. use lower learning rate for pretrained layers;
3. fine-tune;
4. compare against Experiment 1.

Do not default to full-backbone fine-tuning.

## Smoke-test gate before real training

Before launching a full Colab run:
- train on a tiny subset;
- complete forward/backward pass;
- save and reload checkpoint;
- run inference from the saved checkpoint;
- confirm label mapping;
- confirm metrics pipeline.

## M3B acceptance criteria

Preferred success:
- one reproducible training command;
- train/validation/test leakage protections documented;
- checkpoint records config and label mapping;
- held-out metrics generated;
- M3B model can be swapped into the same inference adapter contract;
- comparison against M3A is generated.

Fallback success if training is constrained:
- the attempted training pipeline and blocker are documented;
- smoke-test path is reproducible;
- M3A remains fully usable;
- M4-M6 are not blocked.

## Milestone 3 exit condition

M3A must be complete.

M3B should have at least one reproducible bounded experiment or a documented resource/data limitation.

The project proceeds to M4 using the best currently validated violence adapter.
