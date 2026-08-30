# Codex Handoff Pack — AI-Based Real-Time Crowd Safety Monitoring

Status: **Milestone 1 complete. Milestone 2 is the next implementation milestone.**

This pack is the implementation handoff for the capstone project:

**AI-Based Real-Time Crowd Safety Monitoring for Early Detection of Crowd-Crush / Stampede Risk and Violent Incidents in Dense Public Gatherings**

The system is a human-in-the-loop CCTV safety-assistance platform. It detects people and motion, derives crowd-risk signals, produces temporal violence evidence, fuses those signals over time into one persistent incident, estimates severity, captures evidence, and presents the incident for operator review.

## Read order for Codex

1. `CODEX_HANDOFF.md`
2. `PRD.md`
3. `ARCHITECTURE.md`
4. `DATA_AND_MODELS.md`
5. `EVALUATION_PLAN.md`
6. `ROADMAP.md`
7. Relevant file under `milestones/`
8. `decisions/`
9. `RISK_REGISTER.md`

## Critical project boundary

This is **not** a generic “AI surveillance” product and is **not** a single end-to-end stampede classifier.

The research contribution is:

> Explicit incident-level temporal fusion of person-level violence evidence and surrounding crowd dynamics, with incident lifecycle/severity reasoning and operational alert evaluation.

## Updated implementation philosophy after M1

- Keep the completed deterministic offline runner as the stable foundation.
- Use a mature pretrained person detector and tracker; do not train them initially.
- Use a ready-made temporal violence checkpoint first so the complete pipeline does not depend on training.
- Treat violence-model fine-tuning as a bounded experiment, not a prerequisite for M4-M6.
- Derive interpretable crowd features before adding heavier crowd models.
- Build deterministic/rule-weighted temporal fusion first.
- Add learned fusion only after the baseline pipeline and evaluation dataset exist.
- Treat alerts as persistent incidents, not repeated threshold crossings.
- Add a non-authoritative VLM explanation layer after incident creation to make alerts easier for human operators to understand.
- Keep all incident creation, state, severity, and escalation decisions outside the VLM.
- Keep final external escalation under human control.
- Measure event-level and operational quality, not only classifier accuracy.

## Technology direction

Current preferred direction, subject to repository compatibility and measured results:

- pretrained Ultralytics YOLO26 person detector;
- ByteTrack multi-object tracking;
- engineered trajectory/ROI crowd features;
- pretrained VideoMAE-style binary violence model as the M3A integration baseline;
- X3D-S transfer learning as the bounded M3B training experiment;
- deterministic temporal incident fusion as the primary research implementation;
- optional/configurable VLM explanation of evidence clips/keyframes in M5;
- FastAPI + PostgreSQL + Next.js for the integrated product layer.

## Not included intentionally

This handoff pack does **not** contain repository-specific `tasks/plan.md` or `tasks/todo.md`.

Codex should create those only after inspecting the actual repository and executing the planning workflow for the selected milestone.