# Milestone 5 — Backend, Evidence and Dashboard

## Objective
Turn the incident engine into a usable human-reviewed product flow.

## Backend
Prefer FastAPI unless the existing repository dictates otherwise.

Implement:
- run/source records;
- incident persistence;
- state-transition history;
- evidence metadata;
- operator actions;
- REST API;
- optional server-sent events/WebSocket only if live update is needed.

## Evidence
- snapshot;
- pre-event buffer clip;
- post-event clip;
- contributing reasons;
- signal timeline.

## Frontend
Next.js:
- recent/active incidents;
- incident detail;
- severity/state;
- evidence;
- signal timeline;
- acknowledge;
- dismiss;
- escalate.

## Acceptance criteria
- UI does not infer incident state independently.
- Backend/domain engine is source of truth.
- Operator action is auditable.
- Dismissing an incident does not delete underlying evaluation evidence.
- External emergency dispatch is not implemented.
