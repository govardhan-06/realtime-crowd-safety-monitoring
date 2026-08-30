# Milestone 2 — Person Tracking and Crowd Signals

## Objective
Convert video into persistent trajectories and interpretable crowd-risk features.

## Required work
- pretrained person detector adapter;
- ByteTrack integration;
- track history;
- ROI configuration;
- per-ROI occupancy/density proxy;
- density delta;
- speed/acceleration proxy;
- direction disorder;
- convergence;
- dispersal;
- counter-flow;
- congestion/stagnation;
- visual overlay/debug export;
- per-timestamp feature export.

## Test strategy
Unit-test feature functions with synthetic trajectories:
- all tracks moving right -> low directional disorder;
- half left/half right -> high counter-flow;
- tracks moving toward centroid -> convergence;
- tracks moving outward -> dispersal;
- stationary dense tracks -> congestion.

Integration-test on a short video.

## Acceptance criteria
- Features are deterministic for fixed detections/tracks.
- Missing/insufficient-track cases return explicit valid states.
- Feature definitions/window sizes are documented.
- No claims of physical density in people/m² without camera calibration.
- Dense-scene failures are logged for the later detector-fine-tune decision gate.
