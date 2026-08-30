# Codex Handoff Pack — AI-Based Real-Time Crowd Safety Monitoring

Status: **Product direction locked; implementation not started in this pack.**

This pack is the implementation handoff for the capstone project:

**AI-Based Real-Time Crowd Safety Monitoring for Early Detection of Crowd-Crush / Stampede Risk and Violent Incidents in Dense Public Gatherings**

The system is a human-in-the-loop CCTV safety-assistance platform. It detects people and motion, derives crowd-risk signals, detects physical violence from temporal video, fuses those signals over time into one persistent incident, estimates severity, captures evidence, and presents the incident for operator review.

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

## Implementation philosophy

- Build offline recorded-video inference first.
- Use mature pretrained person detection and tracking.
- Fine-tune one temporal violence model.
- Derive interpretable crowd features before adding heavier crowd models.
- Build deterministic/rule-weighted temporal fusion first.
- Add learned fusion only after the baseline pipeline and evaluation dataset exist.
- Treat alerts as persistent incidents, not repeated threshold crossings.
- Keep final escalation under human control.
- Measure event-level and operational quality, not only classifier accuracy.

## Not included intentionally

This handoff pack does **not** contain repository-specific `tasks/plan.md` or `tasks/todo.md`.

Codex should create those only after inspecting the actual repository and executing the planning workflow for the selected milestone.
