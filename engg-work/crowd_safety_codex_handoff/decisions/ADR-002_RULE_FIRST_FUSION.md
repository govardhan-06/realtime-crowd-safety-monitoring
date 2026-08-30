# ADR-002 — Rule/Calibrated Fusion Before Learned Fusion

## Status
Accepted.

## Decision
Implement transparent deterministic temporal fusion before training a fusion model.

## Why
- produces an early working system;
- establishes required baselines;
- exposes feature quality issues;
- creates the feature/incident data needed to train a later model;
- supports interpretability and ablation.

## Learned fusion gate
A learned MLP/TCN/LSTM is optional and only follows a stable deterministic baseline plus leakage-safe labelled data.
