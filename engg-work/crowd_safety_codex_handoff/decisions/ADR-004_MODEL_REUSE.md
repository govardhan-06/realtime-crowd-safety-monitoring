# ADR-004 — Reuse Mature Perception Models

## Status
Accepted — updated after M1 completion.

## Decision

- use pretrained YOLO26n for the first person-detection implementation;
- use ByteTrack for tracking;
- do not train either component initially;
- use a ready-made pretrained/fine-tuned temporal violence checkpoint for M3A so downstream delivery does not depend on training;
- perform X3D-S transfer learning as the bounded M3B academic experiment;
- keep engineered crowd features first;
- fine-tune the person detector only after measured dense-scene error analysis;
- keep learned fusion optional and gated;
- keep VLM explanation downstream of incident creation and non-authoritative.

## Why

Training every component is unnecessary and would dilute the research contribution.

The core research contribution is the incident-level temporal fusion and operational incident reasoning. The project still includes a legitimate ML training component through bounded X3D-S transfer learning, but unsuccessful fine-tuning must not prevent delivery of the complete system.