# Milestone 2 — Person Tracking and Crowd Signals

## Objective

Convert the completed M1 video runner into persistent person trajectories and interpretable crowd-risk features using pretrained perception only.

## Default technology

### Person detector

- Ultralytics YOLO26n pretrained checkpoint.
- Filter predictions to `person`.
- Keep model name/path/config explicit.
- Keep detector behind the repository's stage/adapter interface.

### Tracker

- ByteTrack.
- No tracker training.
- Track IDs are temporary and source-local.

Do not fine-tune the detector in M2.

If YOLO26n shows material dense-scene misses, record the failure. YOLO26s may be tested as a measured accuracy/compute trade-off. Detector fine-tuning remains a later gated decision.

## Required work

- YOLO26 person-detector adapter;
- ByteTrack integration;
- persistent track history;
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
- per-timestamp feature export;
- detector/tracker latency metrics;
- explicit detector/tracker health status.

## Interface expectations

Detection output should remain model-agnostic.

Conceptually:

```python
PersonDetection(
    box_xyxy=(...),
    confidence=...,
)
```

Tracking output should remain tracker-agnostic.

Conceptually:

```python
TrackObservation(
    source_id=...,
    track_id=...,
    timestamp_s=...,
    center_xy=(...),
    box_xyxy=(...),
    confidence=...,
)
```

Do not leak Ultralytics-specific result objects into crowd-feature/domain logic.

## Test strategy

Unit-test feature functions with synthetic trajectories:

- all tracks moving right -> low directional disorder;
- half left/half right -> high counter-flow;
- tracks moving toward centroid -> convergence;
- tracks moving outward -> dispersal;
- stationary dense tracks -> congestion;
- insufficient track history -> explicit unavailable/insufficient state rather than fabricated zero movement.

Integration-test on at least:
- one normal pedestrian video;
- one dense crowd video;
- one video with strong directional change/running if available.

## Acceptance criteria

- M1 runner remains stable.
- YOLO26n runs through a replaceable detector adapter.
- ByteTrack produces source-local track IDs.
- Features are deterministic for fixed detections/tracks.
- Missing/insufficient-track cases return explicit valid states.
- Feature definitions/window sizes are documented.
- No claims of physical density in people/m² without camera calibration.
- Dense-scene failures are logged for the later detector-fine-tune decision gate.
- Generated overlay and machine-readable feature timeline are visually/structurally inspected.
- Core tests do not require a GPU.