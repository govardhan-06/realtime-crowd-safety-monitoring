# ADR-006 — Non-Authoritative VLM Explainable Alerts

## Status
Accepted.

## Context

The system already exposes deterministic reason codes and feature timelines, but operators and project reviewers benefit from a concise natural-language explanation of what is visible in the incident evidence.

A vision-language model is a useful modern technology for this purpose, but allowing it to control incident decisions would reduce determinism and introduce hallucination risk.

## Decision

Add a VLM explanation stage in M5 **after** incident creation/evidence capture.

Possible inputs:
- evidence clip;
- selected keyframes;
- incident timestamps;
- deterministic reason codes;
- compact signal summary.

Expected output:
- concise description of observable behavior;
- concise operator-facing explanation of why the incident warrants review.

## Hard boundary

The VLM:
- does not create incidents;
- does not close incidents;
- does not change state;
- does not change severity;
- does not enter the fusion feature vector;
- is not required for an alert to be delivered.

The UI must label generated text as AI-generated and show deterministic reason codes separately.

## Provider

Use a provider-swappable interface.

Gemini video/image understanding is the preferred first implementation if credentials are available. The project must retain a disabled/fallback mode for tests and offline operation.

## Why

This adds a current, useful technology that improves operator usability and demo value without compromising the deterministic research contribution.