# Repository Instructions

## Project

This repository is the implementation workspace for an AI-based real-time crowd-safety monitoring system. It is a human-in-the-loop safety-assistance system that combines crowd dynamics and violence evidence into persistent, explainable incidents.

The current repository contains the implementation handoff pack under `engg-work/crowd_safety_codex_handoff/`; implementation has not started yet.

## Read Before Editing

Read these documents in order:

1. `engg-work/crowd_safety_codex_handoff/README.md`
2. `engg-work/crowd_safety_codex_handoff/CODEX_HANDOFF.md`
3. `engg-work/crowd_safety_codex_handoff/PRD.md`
4. `engg-work/crowd_safety_codex_handoff/ARCHITECTURE.md`
5. `engg-work/crowd_safety_codex_handoff/ROADMAP.md`
6. One selected file under `engg-work/crowd_safety_codex_handoff/milestones/`
7. Relevant files under `engg-work/crowd_safety_codex_handoff/decisions/`
8. `engg-work/crowd_safety_codex_handoff/RISK_REGISTER.md`

Before changing code, inspect the actual repository structure, package/runtime configuration, existing conventions, tests, and available commands. Adapt the handoff to the repository; do not impose its suggested folder tree mechanically. Since implementation is not present yet, do not invent commands or claim runtime support that has not been verified.

## Delivery Workflow

- Select and implement one milestone at a time, beginning with M1: offline video foundation.
- Create repository-specific `tasks/plan.md` and `tasks/todo.md` only after inspecting the implementation repository and selecting the milestone.
- Do not begin a later milestone until the current milestone meets its acceptance criteria.
- Prefer the smallest working implementation and reuse existing project patterns.
- Keep model adapters replaceable and keep experimental thresholds, windows, cadence, and weights explicit in structured configuration.
- Use a deterministic, single-process offline runner before adding live inputs, queues, services, or other distributed infrastructure. Add infrastructure only after a measured need.
- Keep incident state separate from model-specific code. Use temporal fusion, persistence, lifecycle, severity, deduplication, reason codes, and evidence rather than repeated independent alerts.

## Safety and Privacy Boundaries

Do not add or imply:

- facial recognition, demographic inference, or persistent cross-camera identity;
- autonomous police, fire, medical, or emergency dispatch;
- guaranteed prediction of a future stampede;
- a single opaque end-to-end stampede classifier;
- detector or large video-model training from scratch;
- a hard dependency on RWF-2000.

Use wording such as “risk,” “warning,” or “detected indicators,” and require human review before external escalation.

Do not commit datasets, raw CCTV footage, checkpoints, or other large/private media. Use manifests and environment-configured external storage.

## Verification

For each milestone, run the repository’s applicable focused tests and configured lint/type checks, exercise at least one realistic integration path, inspect generated artifacts or visual output, record performance and failures, and review the result against the milestone acceptance criteria. Keep core tests runnable without a GPU where practical. Distinguish static, mocked, offline, and live/runtime evidence when reporting results.
