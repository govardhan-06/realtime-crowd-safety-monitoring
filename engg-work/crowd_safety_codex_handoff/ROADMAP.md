# Product Roadmap

## Delivery strategy

The project should be built as a vertical slice that becomes progressively more intelligent.

Do not develop all ML modules in parallel before a working runner exists.

## Milestone sequence

### M1 — Offline video foundation
Outcome:
- deterministic local-video runner;
- config loading;
- timestamps;
- annotated output;
- structured run metadata;
- basic tests/benchmark harness.

Exit condition:
A single command processes a video reproducibly and emits an output video plus machine-readable frame/run metadata.

### M2 — Person tracking and crowd signals
Outcome:
- pretrained person detector;
- ByteTrack;
- trajectories;
- ROIs;
- density/motion/convergence/dispersal/counter-flow/congestion features;
- feature visualisation/export.

Exit condition:
Dense normal and abnormal-motion videos produce stable, inspectable crowd-feature timelines.

### M3 — Temporal violence model
Outcome:
- dataset adapters/manifests;
- X3D-S fine-tuning pipeline;
- checkpoint/config management;
- clip inference module;
- rolling violence evidence integrated into offline runner;
- component evaluation.

Exit condition:
The runner emits timestamp-aligned violence scores on held-out videos with a documented threshold and model version.

### M4 — Temporal fusion and incident engine
Outcome:
- common signal schema;
- rolling signal window;
- smoothing/persistence;
- rule/calibrated fusion;
- lifecycle;
- severity;
- deduplication;
- event/evidence records;
- baseline modes B1/B2/B3/B4.

Exit condition:
A long video produces one coherent incident timeline with reason codes and deterministic state transitions.

### M5 — Backend, evidence, dashboard
Outcome:
- persistent incident storage;
- snapshot/pre-post clip capture;
- FastAPI endpoints;
- Next.js incident list/detail;
- acknowledge/dismiss/escalate;
- run/source status.

Exit condition:
Operator can process/view a video-driven incident end-to-end and disposition it through the UI.

### M6 — Evaluation, calibration, demo hardening
Outcome:
- project-specific staged/curated evaluation set;
- hard negatives;
- baseline comparison;
- operational metrics;
- latency profiling;
- optional learned fusion experiment if justified;
- final demo scripts and reproducible report.

Exit condition:
Results directly answer the research question and the demo shows the complete incident lifecycle.

## Parallelisable work

After M1 stabilises:
- one stream can prepare dataset manifests/training environment for M3;
- one stream can implement M2 crowd features.

Do not parallelise shared signal contracts without agreeing their schema first.

## Scope gate for learned fusion

Only implement learned fusion if all are true:
1. M4 deterministic fusion is complete;
2. enough labelled incident windows exist;
3. train/validation/test split can avoid event leakage;
4. it can be evaluated against the deterministic baseline;
5. it does not threaten M5/M6 delivery.

Otherwise keep learned fusion as future work.
