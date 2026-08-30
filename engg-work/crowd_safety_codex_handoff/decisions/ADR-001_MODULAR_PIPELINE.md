# ADR-001 — Modular Perception + Incident Pipeline

## Status
Accepted.

## Decision
Use separate modules for ingestion, detection, tracking, crowd features, violence inference, temporal fusion, incident lifecycle, evidence, persistence, and UI.

## Why
The project contribution is fusion/incident reasoning. A monolithic end-to-end classifier would:
- obscure which signals cause decisions;
- require a dataset that does not exist at sufficient scale;
- make ablations difficult;
- weaken explainability;
- make operational evaluation harder.

## Consequences
More integration contracts are required, but components can be independently tested and replaced.
