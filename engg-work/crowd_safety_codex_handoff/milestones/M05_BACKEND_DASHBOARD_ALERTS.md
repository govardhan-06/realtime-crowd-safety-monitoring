# Milestone 5 — Backend, Evidence, Dashboard and Explainable Alerts

## Objective

Turn the incident engine into a usable human-reviewed product flow and add a modern VLM-based explanation layer without allowing generative AI to control incident decisions.

## Backend

Prefer FastAPI unless the existing repository dictates otherwise.

Implement:
- run/source records;
- incident persistence;
- state-transition history;
- evidence metadata;
- operator actions;
- REST API;
- optional server-sent events/WebSocket only if live update is needed;
- incident explanation record/status.

## Evidence

- snapshot;
- pre-event buffer clip;
- post-event clip;
- contributing deterministic reason codes;
- signal timeline;
- model/stage health where relevant.

## VLM explanation layer

Implement a provider-swappable `IncidentExplainer` interface.

Preferred first provider:
- Gemini video/image understanding, if API credentials are available.

Possible input:
- evidence clip and/or selected keyframes;
- incident time range;
- deterministic reason codes;
- compact feature summary.

Expected output:
- short description of observable activity;
- short operator-facing explanation.

### Hard constraints

The VLM:
- is invoked only after an incident exists;
- cannot create an incident;
- cannot close an incident;
- cannot modify severity;
- cannot modify lifecycle state;
- cannot trigger external escalation;
- cannot feed a generated score/text back into M4 fusion.

If VLM inference fails:
- store explanation status as failed/unavailable;
- continue the normal incident flow;
- show deterministic reason codes/evidence normally.

Unit/integration tests must be able to run without VLM credentials.

## Frontend

Next.js:
- recent/active incidents;
- incident detail;
- severity/state;
- evidence;
- deterministic reason codes;
- signal timeline;
- model/stage health where useful;
- AI-generated explanation clearly labelled as generated;
- explanation loading/unavailable/failure state;
- acknowledge;
- dismiss;
- escalate.

## Suggested demo presentation

Show both layers separately:

```text
Why the system alerted:
- violence score increased
- rapid local dispersal
- elevated direction disorder
- persistence threshold satisfied

AI-generated evidence summary:
"Possible physical altercation is visible near the centre of the frame,
followed by rapid outward movement of surrounding people."
```

The first block is authoritative system evidence.

The second block is supplementary generated text.

## Acceptance criteria

- UI does not infer incident state independently.
- Backend/domain engine is source of truth.
- Operator action is auditable.
- Dismissing an incident does not delete underlying evaluation evidence.
- External emergency dispatch is not implemented.
- VLM explanation is never required for alert delivery.
- VLM output is clearly labelled and stored separately from deterministic reason codes.
- Failure/timeout of the VLM does not change incident state.