# CODEX HANDOFF

## Mission

Implement the locked capstone product **AI-Based Real-Time Crowd Safety Monitoring** in milestones without changing the research question or expanding scope.

The most important design principle is:

> Reuse mature perception components; put project-specific engineering/research effort into interpretable crowd signals, temporal incident fusion, lifecycle/severity, evidence, and operational evaluation.

## First action in a repository

Before editing code:

1. Read repository-level `AGENTS.md` and all applicable nested instructions.
2. Inspect existing language/framework/package conventions.
3. Map this handoff to the existing repo instead of imposing the suggested folder tree.
4. Identify existing tests, lint/typecheck/build commands.
5. Read `PRD.md`, `ARCHITECTURE.md`, `ROADMAP.md`.
6. Select **one milestone only**.
7. Read that milestone file and relevant ADRs.
8. Create repository-specific implementation planning artifacts according to the local Codex workflow.
9. Challenge the plan for unnecessary infrastructure/scope.
10. Implement, verify, review, and commit the milestone before starting the next.

## Current recommended starting milestone

**M1 — Offline Video Foundation**

Do not start by training X3D or building the dashboard.

A stable deterministic video runner is the dependency for all later work.

## Hard constraints

Codex MUST NOT:
- turn the project into facial recognition;
- add demographic inference;
- add persistent cross-camera identity;
- add autonomous emergency dispatch;
- claim guaranteed stampede prediction;
- train a detector or large video model from scratch;
- make RWF-2000 a hard dependency;
- introduce distributed infrastructure before a measured need;
- skip the violence-only/crowd-only/naive-OR baselines;
- replace the incident engine with repeated independent alerts;
- silently hard-code experimental thresholds throughout application logic.

## Preferred initial technical direction

Unless the repository already locks alternatives:

- Python 3.11+ for CV/ML/backend pipeline.
- OpenCV/FFmpeg-compatible video decode.
- PyTorch for video model.
- Pretrained YOLO-family detector.
- ByteTrack.
- X3D-S as first fine-tuned violence recognizer.
- FastAPI for final API.
- PostgreSQL for final persistence.
- Next.js for operator dashboard.
- YAML/TOML/structured config for experimental thresholds.
- JSONL/Parquet/CSV artifacts for early offline runs.

Treat versions as repository decisions. Pin them once compatibility is tested.

## Engineering rules

### Interfaces
Keep model adapters behind typed interfaces.

### Configuration
All experimental parameters should be explicit:
- detector cadence;
- clip length;
- violence cadence;
- ROI;
- feature windows;
- smoothing;
- thresholds;
- weights;
- persistence duration;
- hysteresis;
- evidence pre/post duration.

### Testing
Unit-test pure movement/fusion/state logic with synthetic data.
Do not require GPU for the core test suite.

### Verification loop
For every milestone:
1. run focused tests;
2. run lint/typecheck where configured;
3. execute at least one realistic integration path;
4. inspect generated artifact/visual output;
5. record performance/failures;
6. do a fresh review against the milestone acceptance criteria.

### Data
Do not commit datasets/checkpoints/raw CCTV footage.
Use manifests and environment-configured storage paths.

### Safety wording
Code/docs/UI should say “risk,” “warning,” or “detected indicators” rather than claiming a guaranteed future stampede.

## Milestone ordering

1. `milestones/M01_OFFLINE_VIDEO_FOUNDATION.md`
2. `milestones/M02_TRACKING_AND_CROWD_SIGNALS.md`
3. `milestones/M03_VIOLENCE_MODEL.md`
4. `milestones/M04_FUSION_AND_INCIDENT_ENGINE.md`
5. `milestones/M05_BACKEND_DASHBOARD_ALERTS.md`
6. `milestones/M06_EVALUATION_AND_DEMO.md`

M2 and M3 may proceed partially in parallel after M1 only if interfaces are stable.

## Definition of project done

The project is not done when YOLO boxes or a violence probability appear.

It is done when a test video can flow through:

```text
video
 -> person detection/tracking
 -> crowd features
 -> temporal violence evidence
 -> aligned temporal fusion
 -> one evolving incident
 -> severity/lifecycle
 -> evidence capture
 -> human-review dashboard
 -> stored outcome
 -> reproducible evaluation
```

and the final experiment compares the proposed incident-level method against the required independent baselines using event and operational metrics.

## Final implementation review checklist

- [ ] Offline and live/stream boundaries are clear.
- [ ] Detector/tracker are replaceable adapters.
- [ ] Crowd features are documented and unit tested.
- [ ] Violence model version/checkpoint/config is traceable.
- [ ] Fusion can replay from stored feature streams.
- [ ] Incident state is deterministic and deduplicated.
- [ ] Every alert contains reason codes.
- [ ] Human action is persisted.
- [ ] Raw model score is not treated as final incident severity.
- [ ] False alerts per camera-hour can be computed.
- [ ] Detection delay can be computed.
- [ ] Baseline modes run through the same evaluation harness.
- [ ] No restricted media/checkpoints are committed.
- [ ] No facial recognition/demographic inference exists.
- [ ] Documentation avoids guaranteed-prediction claims.
