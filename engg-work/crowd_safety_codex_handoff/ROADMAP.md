# Product Roadmap

## Delivery strategy

The project should be built as a vertical slice that becomes progressively more intelligent.

The completed M1 runner is the stable execution foundation.

The key delivery principle is:

> **Fine-tuning must not be on the critical path for obtaining a complete working POC.**

The system should first be completed end-to-end using pretrained components. Project-specific violence-model fine-tuning is then performed as a bounded academic experiment and evaluated against the pretrained baseline.

```text
M1 — Offline video foundation ✅
        |
        v
M2 — Pretrained perception + crowd signals
        |
        v
M3A — Pretrained temporal violence baseline
        |
        v
M4 — Temporal fusion + incident engine
        |
        v
M5 — Backend + evidence + dashboard + VLM
        |
        v
M6A — POC validation + baseline evaluation
        |
        v
========================================
          COMPLETE WORKING POC
========================================
        |
        v
M3B — X3D-S transfer-learning experiment
        |
        v
M6B — Final comparative evaluation
```

M3A is on the critical delivery path.

M3B is **not** on the critical delivery path.

Once M3A is complete, development proceeds directly through M4, M5, and M6A to obtain a complete working POC.

M3B is then performed as a bounded transfer-learning experiment. M6B compares the pretrained and project-fine-tuned violence models and produces the final experimental results.

---

## Milestone sequence

### M1 — Offline video foundation — COMPLETE

Outcome:

* deterministic local-video runner;
* config loading;
* timestamps;
* annotated output;
* structured run metadata;
* basic tests/benchmark harness.

Status:

* complete;
* do not redo unless a later integration exposes a concrete defect in the generic ingestion, scheduling, timing, or stage interfaces.

---

### M2 — Person tracking and crowd signals

Outcome:

* pretrained YOLO26n person detector;
* ByteTrack;
* persistent temporary trajectories;
* configurable ROIs;
* occupancy/density proxy;
* density change;
* speed and acceleration proxies;
* direction disorder;
* convergence;
* dispersal;
* counter-flow;
* congestion/stagnation;
* feature visualisation/export;
* detector/tracker stage-health reporting.

No model training is required in this milestone.

#### Exit condition

Dense normal and abnormal-motion videos produce stable, inspectable crowd-feature timelines with reproducible person tracks.

---

### M3A — Pretrained temporal violence baseline

Outcome:

* generic temporal-video classifier adapter;
* ready-made binary violence checkpoint integrated;
* rolling video clip buffer;
* configurable clip sampling;
* timestamp-aligned violence scores;
* model/checkpoint metadata;
* local development evaluation;
* explicit model-health semantics:

  * available;
  * degraded;
  * unavailable.

No project-specific training is required in M3A.

The purpose of M3A is to ensure that downstream development does not depend on successful fine-tuning.

#### Exit condition

The pipeline emits timestamp-aligned violence evidence on positive and negative development videos using a pretrained violence model.

M4 can begin immediately after this milestone.

---

### M4 — Temporal fusion and incident engine

Outcome:

* common timestamp-aligned signal schema;
* rolling signal windows;
* smoothing;
* persistence;
* spatial association;
* configurable fusion;
* fused risk score;
* deterministic reason codes;
* incident lifecycle;
* severity;
* hysteresis/decay;
* deduplication;
* degraded/missing-signal semantics;
* state-transition history;
* baseline modes:

  * violence-only;
  * crowd-only;
  * naive OR;
  * simple rule fusion.

This milestone contains the primary project-specific research contribution.

#### Exit condition

A long video produces one coherent incident timeline with deterministic state transitions, severity, reason codes, and duplicate suppression.

Missing model evidence must never be interpreted as normal evidence.

---

### M5 — Backend, evidence, dashboard and explainable alerts

Outcome:

* persistent incident storage;
* incident state-transition history;
* snapshot capture;
* pre-event evidence clip;
* post-event evidence clip;
* FastAPI endpoints;
* Next.js operator dashboard;
* incident list/detail views;
* acknowledge/dismiss/escalate actions;
* run/source status;
* deterministic reason-code display;
* signal timeline;
* optional VLM-generated incident explanation.

The VLM explanation layer operates only after an incident has already been created.

It must not:

* create incidents;
* close incidents;
* modify severity;
* modify incident lifecycle state;
* trigger emergency escalation;
* contribute signals to temporal fusion.

#### Exit condition

An operator can process a video-driven incident end-to-end, inspect evidence and deterministic reasons, optionally view an AI-generated evidence explanation, and disposition the incident through the dashboard.

The system must remain fully functional when the VLM is disabled or unavailable.

---

### M6A — POC validation and baseline evaluation

## Purpose

Validate the complete pretrained end-to-end system before starting project-specific violence-model fine-tuning.

Outcome:

* project-specific staged/curated evaluation set;
* development/test video manifests;
* hard-negative set;
* event annotations;
* baseline comparison:

  * violence-only;
  * crowd-only;
  * naive OR;
  * deterministic rule fusion;
  * proposed temporal incident fusion;
* event precision/recall/F1;
* false alerts per camera-hour;
* duplicate alerts per incident;
* detection delay;
* latency profiling;
* failure analysis;
* VLM explanation quality/failure checks;
* complete end-to-end demonstration.

M6A does **not** require M3B.

#### Exit condition

A complete working POC exists:

```text
Video
  ↓
YOLO26 + ByteTrack
  ↓
Crowd dynamics
  ↓
Pretrained violence evidence
  ↓
Temporal fusion
  ↓
Incident lifecycle + severity
  ↓
Evidence capture
  ↓
Dashboard
  ↓
Optional VLM explanation
  ↓
Human review
```

At this point, the project has a usable fallback even if later fine-tuning performs poorly.

---

# M3B — Bounded X3D-S transfer-learning experiment

## Purpose

Perform the project's controlled violence-model training experiment **after the complete POC is stable**.

Outcome:

* dataset manifests;
* lawful dataset acquisition/provenance;
* leakage-safe train/validation/test splits;
* reproducible preprocessing;
* reproducible training configuration;
* X3D-S pretrained backbone;
* binary violence/non-violence classifier head;
* frozen-backbone/head-only training experiment;
* held-out validation/test metrics;
* threshold calibration;
* optional partial backbone unfreezing only if justified;
* project-trained checkpoint;
* X3D inference adapter implementing the same contract as M3A.

Training sequence:

```text
Pretrained X3D-S
      ↓
Replace classification head
      ↓
Freeze backbone
      ↓
Train binary head
      ↓
Evaluate
      ↓
Optional:
unfreeze final block(s)
      ↓
low-learning-rate fine-tuning
```

Do not:

* train X3D from scratch;
* default to full-backbone training;
* redesign M4/M5 around X3D;
* change the `ViolenceEvidence` contract.

#### Exit condition

At least one reproducible X3D-S transfer-learning experiment has been completed and evaluated.

The resulting checkpoint can replace the M3A violence model without requiring changes to the fusion engine or product architecture.

---

### M6B — Final comparative evaluation and report hardening

## Purpose

Determine whether project-specific fine-tuning actually improves the violence component and/or complete incident-detection system.

Outcome:

* M3A pretrained violence model vs M3B X3D-S comparison;
* component-level violence metrics;
* rerun end-to-end incident evaluation using the M3B model;
* compare:

```text
M3A violence model
      +
same temporal fusion
      ↓
System A

vs

M3B fine-tuned X3D-S
      +
same temporal fusion
      ↓
System B
```

Measure:

* violence precision;
* violence recall;
* violence F1;
* event precision;
* event recall;
* event F1;
* false alerts per camera-hour;
* detection delay;
* duplicate alert rate;
* inference latency.

Also complete:

* final threshold calibration;
* final failure analysis;
* ablation experiments;
* final tables/graphs;
* final demo configuration;
* report/presentation results.

The M3B model should replace M3A in the final demo only if measured results justify doing so.

Fine-tuning is **not automatically considered an improvement**.

#### Exit condition

The final report can clearly state whether project-specific violence-model fine-tuning improved:

1. violence recognition;
2. operational incident detection;
3. neither.

---

## Execution order

The default implementation order is:

```text
M1 ✅
 ↓
M2
 ↓
M3A
 ↓
M4
 ↓
M5
 ↓
M6A
 ↓
Complete working POC
 ↓
M3B
 ↓
M6B
 ↓
Final evaluated system
```

Do not delay M4-M6A waiting for M3B.

---

## Parallelisable work

Keep parallel work conservative to avoid unnecessary complexity.

### After M1

Primary work:

* implement M2.

### After M2

Primary work:

* implement M3A.

### After M3A

Proceed directly through:

```text
M4 → M5 → M6A
```

### After the complete POC is stable

Execute:

```text
M3B → M6B
```

Dataset downloads or documentation preparation for M3B may happen earlier when convenient, but M3B implementation/training should not distract from reaching the complete POC.

Do not parallelise changes to shared signal/domain contracts without agreeing the schema first.

---

## Scope gate for learned fusion

Only implement a learned fusion model if all are true:

1. M4 deterministic fusion is complete;
2. M6A baseline evaluation is complete;
3. enough labelled incident windows exist;
4. train/validation/test splits can avoid event leakage;
5. the learned model can be compared directly against deterministic fusion;
6. implementing it does not threaten final project delivery.

Potential architectures:

* small MLP;
* temporal convolutional network;
* small LSTM.

Otherwise retain learned fusion as future work.

The deterministic temporal fusion system remains the primary viable implementation.

---

## Scope gate for detector fine-tuning

Do not fine-tune YOLO merely because training is possible.

Fine-tune the person detector only if:

1. evaluation shows meaningful missed detections in dense/occluded scenes;
2. those misses materially degrade crowd features;
3. a suitable lawful training dataset is available;
4. improvement can be measured.

Otherwise keep the pretrained detector.

---

## Scope gate for additional “new technology”

Do not add technologies merely to make the architecture appear more advanced.

Examples that are currently outside the core plan:

* blockchain;
* SAM/SAM2 segmentation;
* additional foundation models;
* distributed Kafka-style streaming;
* complex microservices;
* autonomous emergency dispatch.

Any additional technology must satisfy at least one of:

* materially improves measured detection/tracking quality;
* materially improves operator explainability;
* materially improves real-time performance/deployability;
* solves a documented project limitation;
* adds a defensible research experiment without threatening delivery.

The currently approved showcase technology is:

> **Vision-Language-Model-based explainable incident alerts**

The VLM provides a human-readable explanation of incident evidence while remaining outside the authoritative detection and incident-state pipeline.

---

## Final project target

The final system should demonstrate:

```text
CCTV / MP4
   │
   ├── YOLO26 person detection
   │       ↓
   │    ByteTrack
   │       ↓
   │    Crowd dynamics
   │
   └── Temporal violence model
             │
             ▼
      Incident-level
      temporal fusion
             │
             ▼
     Lifecycle + severity
             │
             ▼
       Evidence capture
             │
             ├── deterministic reason codes
             │
             └── optional VLM explanation
                         │
                         ▼
                 Operator dashboard
                         │
                         ▼
                   Human review
```

The final research evaluation should establish whether:

> Incident-level temporal fusion of violence evidence and surrounding crowd-response signals provides a better operational alerting trade-off than independent violence-only or crowd-only detection.
