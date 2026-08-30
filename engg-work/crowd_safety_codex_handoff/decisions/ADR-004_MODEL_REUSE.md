# ADR-004 — Reuse Mature Perception Models

## Status
Accepted.

## Decision
- pretrained YOLO-family model for person detection;
- ByteTrack for tracking;
- fine-tuned pretrained X3D-S for violence;
- engineered crowd features first;
- optional detector fine-tuning only after measured dense-scene error analysis;
- optional VideoMAE-like comparison only after baseline delivery.

## Why
Training every component is unnecessary and would dilute the research contribution.
