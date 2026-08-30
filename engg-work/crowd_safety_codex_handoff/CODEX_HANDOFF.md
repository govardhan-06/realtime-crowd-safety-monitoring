# CODEX HANDOFF

## Mission

Implement the locked capstone product **AI-Based Real-Time Crowd Safety Monitoring** in milestones without changing the research question or expanding scope.

The most important design principle is:

> Reuse mature perception components; put project-specific engineering/research effort into interpretable crowd signals, temporal incident fusion, lifecycle/severity, evidence, explainability, and operational evaluation.

## Current project state

**Milestone 1 — Offline Video Foundation is complete.**

Do not redo M1 unless a later milestone exposes a concrete defect in the generic ingestion/timing/stage interfaces.

The next milestone is:

**M2 — Person Tracking and Crowd Signals**

## First action before each remaining milestone

Before editing code:

1. Read repository-level `AGENTS.md` and all applicable nested instructions.
2. Inspect the completed M1 implementation and its existing interfaces/tests.
3. Read `PRD.md`, `ARCHITECTURE.md`, `DATA_AND_MODELS.md`, `ROADMAP.md`.
4. Select **one milestone/sub-milestone only**.
5. Read that milestone file and relevant ADRs.
6. Map the handoff to the existing repository instead of imposing the suggested folder tree.
7. Identify existing tests, lint/typecheck/build commands.
8. Create repository-specific implementation planning artifacts according to the local Codex workflow.
9. Challenge the plan for unnecessary infrastructure/scope.
10. Implement, verify, review, and commit before starting the next milestone.

## Hard constraints

Codex MUST NOT:
- turn the project into facial recognition;
- add demographic inference;
- add persistent cross-camera identity;
- add autonomous emergency dispatch;
- claim guaranteed stampede prediction;
- train a detector or large video model from scratch;
- make RWF-2000 a hard dependency;
- make successful violence-model fine-tuning a prerequisite for M4-M6;
- let a VLM create, close, escalate, or change incident severity;
- introduce distributed infrastructure before a measured need;
- skip the violence-only/crowd-only/naive-OR baselines;
- replace the incident engine with repeated independent alerts;
- silently hard-code experimental thresholds throughout application logic.

## Preferred technical direction

Unless the repository already locks compatible alternatives:

- Python 3.11+ for CV/ML/backend pipeline.
- OpenCV/FFmpeg-compatible video decode.
- PyTorch for local video models.
- **YOLO26n** as the first pretrained person detector; try YOLO26s only if measured dense-scene recall requires it.
- **ByteTrack** as the first tracker.
- **M3A:** integrate a pretrained binary violence-classification checkpoint through a generic video-classification adapter.
- **M3B:** run a bounded X3D-S transfer-learning experiment: frozen backbone/head training first, partial unfreezing only if validation justifies it.
- FastAPI for final API.
- PostgreSQL for final persistence.
- Next.js for operator dashboard.
- A configurable VLM adapter for evidence explanation in M5; default implementation may use Gemini video/image understanding if credentials are available.
- YAML/TOML/structured config for experimental thresholds.
- JSONL/Parquet/CSV artifacts for offline runs.

Treat exact package/model versions as repository decisions. Pin them once compatibility is tested.

## Model availability rule

Every model-backed stage must expose explicit health/availability.

Examples:
- `available`
- `degraded`
- `unavailable`

Missing model evidence must **not** be converted to a normal/zero-risk score.

For example:

```text
violence model unavailable
!=
violence_score = 0
```

Downstream fusion should be able to distinguish “no violence evidence observed” from “violence branch did not produce evidence.”

## Engineering rules

### Interfaces

Keep detector, tracker, violence model, and VLM behind typed adapters.

### Configuration

All experimental parameters should be explicit:
- detector model/checkpoint;
- detector cadence;
- tracker config;
- clip length;
- violence cadence;
- ROI;
- feature windows;
- smoothing;
- thresholds;
- weights;
- persistence duration;
- hysteresis;
- evidence pre/post duration;
- VLM enable/disable flag;
- VLM provider/model identifier where used.

### Testing

Unit-test pure movement/fusion/state logic with synthetic data.

Do not require GPU, external APIs, or VLM credentials for the core test suite.

### Verification loop

For every milestone:

1. run focused tests;
2. run lint/typecheck where configured;
3. execute at least one realistic integration path;
4. inspect generated artifact/visual output;
5. record performance/failures;
6. do a fresh review against milestone acceptance criteria.

### Data

Do not commit datasets/checkpoints/raw CCTV footage.

Use manifests and environment-configured storage paths.

### Safety wording

Code/docs/UI should say “risk,” “warning,” “detected indicators,” or “possible incident” rather than claiming a guaranteed future stampede.

## Remaining milestone ordering

1. `milestones/M02_TRACKING_AND_CROWD_SIGNALS.md`
2. `milestones/M03_VIOLENCE_MODEL.md`
   - M3A pretrained violence baseline
   - M3B bounded transfer-learning experiment
3. `milestones/M04_FUSION_AND_INCIDENT_ENGINE.md`
4. `milestones/M05_BACKEND_DASHBOARD_ALERTS.md`
5. `milestones/M06_EVALUATION_AND_DEMO.md`

M3A is required before the normal M4 integration path.

M3B should be attempted for the academic experiment but **must not block M4, M5, or M6** if training is delayed or underperforms. M2 and M3A/M3B may proceed partially in parallel only when shared signal contracts are stable.

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
 -> optional VLM evidence explanation
 -> human-review dashboard
 -> stored outcome
 -> reproducible evaluation
```

and the final experiment compares the proposed incident-level method against the required independent baselines using event and operational metrics.

## Final implementation review checklist

- [ ] M1 remains stable and reusable.
- [ ] Offline and live/stream boundaries are clear.
- [ ] Detector/tracker are replaceable adapters.
- [ ] Crowd features are documented and unit tested.
- [ ] Violence model version/checkpoint/config is traceable.
- [ ] The system can run with the M3A pretrained violence baseline.
- [ ] M3B fine-tuning results are separately reproducible.
- [ ] Missing model evidence is represented explicitly, not as zero risk.
- [ ] Fusion can replay from stored feature streams.
- [ ] Incident state is deterministic and deduplicated.
- [ ] Every alert contains deterministic reason codes.
- [ ] VLM output is supplementary and never authoritative.
- [ ] Human action is persisted.
- [ ] Raw model score is not treated as final incident severity.
- [ ] False alerts per camera-hour can be computed.
- [ ] Detection delay can be computed.
- [ ] Baseline modes run through the same evaluation harness.
- [ ] No restricted media/checkpoints are committed.
- [ ] No facial recognition/demographic inference exists.
- [ ] Documentation avoids guaranteed-prediction claims.