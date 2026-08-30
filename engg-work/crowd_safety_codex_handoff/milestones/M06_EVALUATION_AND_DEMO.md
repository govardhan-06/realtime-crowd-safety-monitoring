# Milestone 6 — Evaluation and Demo Hardening

## Objective
Produce evidence that answers the research question.

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
- demo configuration.

## Required comparison
- violence-only;
- crowd-only;
- naive OR;
- rule fusion;
- proposed temporal incident fusion.

## Optional
Train a small learned fusion model only if the roadmap gate is satisfied.

## Final demo sequence
1. play/process dense crowd video;
2. show detections/tracks;
3. show crowd features;
4. show violence evidence;
5. show one incident form and escalate;
6. show evidence clip/reasons;
7. operator acknowledges/escalates;
8. show stored timeline;
9. show comparative evaluation.

## Acceptance criteria
- Metrics are reproducible from saved predictions.
- False alerts per camera-hour and detection delay are reported.
- Duplicate alerts are explicitly measured.
- Failure cases are documented, not hidden.
- Report wording uses “early detection of risk indicators,” not guaranteed stampede prediction.
