# Milestone 6 — Evaluation and Demo Hardening

## Objective

Produce evidence that answers the research question and demonstrate the modern explainable-alert workflow without confusing generative explanation with detection performance.

## Required work

- final evaluation manifest;
- project-specific staged/curated footage;
- hard-negative set;
- baseline runner;
- event matcher;
- operational metrics;
- latency instrumentation;
- per-scenario failure analysis;
- reproducible reports;
- demo configuration;
- M3A vs M3B violence-model comparison;
- small reviewed VLM-explanation quality/failure analysis.

## Required incident-method comparison

- violence-only;
- crowd-only;
- naive OR;
- rule fusion;
- proposed temporal incident fusion.

## Required violence-model comparison

Where the same evaluation subset is compatible:
- M3A ready-made pretrained/fine-tuned violence baseline;
- M3B project X3D-S transfer-learning result.

Report project-measured metrics only.

## VLM evaluation

Do not include VLM output as a detection signal.

Review a small sample of generated explanations for:
- grounding in visible evidence;
- contradiction with deterministic reason codes;
- unsupported/hallucinated details;
- latency;
- timeout/failure rate.

If the VLM is unavailable, the core evaluation remains valid.

## Optional

Train a small learned fusion model only if the roadmap gate is satisfied.

Do not add extra “new technologies” during M6 unless they address a measured failure and do not threaten final delivery.

## Final demo sequence

1. play/process dense crowd video;
2. show YOLO26 person detections and ByteTrack tracks;
3. show crowd features;
4. show temporal violence evidence;
5. show temporal fusion forming one persistent incident;
6. show state/severity change and deterministic reason codes;
7. show snapshot/evidence clip;
8. show optional AI-generated evidence explanation;
9. operator acknowledges/escalates/dismisses;
10. show stored timeline;
11. show comparative evaluation.

## Acceptance criteria

- Metrics are reproducible from saved predictions.
- False alerts per camera-hour and detection delay are reported.
- Duplicate alerts are explicitly measured.
- M3A and M3B results are clearly distinguished.
- VLM explanation quality is discussed separately from incident-detection metrics.
- Failure cases are documented, not hidden.
- Report wording uses “early detection of risk indicators,” not guaranteed stampede prediction.
- Demo remains functional if the VLM/API is disabled.