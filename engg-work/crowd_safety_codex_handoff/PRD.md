# Product Requirements Document

## 1. Product

**AI-Based Real-Time Crowd Safety Monitoring**

A real-time CCTV safety-assistance system for dense public gatherings that identifies observable crowd-risk signals and violent interactions, fuses them over time into a coherent incident, estimates severity, and produces an evidence-backed alert for a human operator.

## 2. Problem

In dense gatherings, a safety incident often evolves through multiple weak but related visual signals:

- physical aggression;
- local density increase;
- sudden acceleration or crowd surge;
- convergence around a focal point;
- rapid dispersal;
- directional disorder or counter-flow;
- congestion/stagnation.

Standalone violence models and crowd-anomaly systems usually output independent scores. Operationally, the operator instead needs to know:

- whether these observations belong to the same event;
- whether the event is persistent or transient;
- whether it is escalating;
- how severe it is;
- why it was flagged;
- whether it has already been alerted.

## 3. Locked problem statement

Build an AI-based real-time crowd-safety monitoring system for dense public gatherings that detects early indicators of crowd-crush/stampede risk and violent incidents from CCTV, combines the signals as one evolving incident, estimates severity, and generates evidence-backed alerts for human-reviewed escalation.

## 4. Research question

Can joint temporal analysis of crowd dynamics and violent activity provide earlier and more reliable detection of dangerous crowd incidents than crowd-only, violence-only, or independently triggered surveillance systems?

## 5. Hypothesis

Temporal fusion of person-level violence evidence with local crowd density, flow, and motion abnormalities should improve event-level reliability and severity prioritisation while reducing nuisance and duplicate alerts compared with independent pipelines.

## 6. Primary users

- Control-room / event-safety operator
- Security personnel
- Event organiser / local authority
- Emergency-response coordinators after human verification

## 7. Core user journey

1. Operator registers or selects a camera/video source.
2. The system processes the stream/video continuously.
3. Person detections/tracks and crowd features are produced.
4. The violence model produces rolling temporal evidence.
5. Fusion logic creates or updates a single incident.
6. Incident state and severity change as evidence accumulates.
7. The system stores a snapshot and pre/post-event evidence clip.
8. Operator sees one alert with reason codes and timeline.
9. Operator acknowledges, dismisses, or escalates the incident.
10. The final outcome is retained for evaluation/audit.

## 8. Functional requirements

### FR-1 Video ingestion
- Support local video files in the first implementation.
- Preserve timestamps/frame index.
- Add webcam/RTSP only after offline stability.
- Allow configured frame sampling and resizing.

### FR-2 Person detection
- Detect `person` instances using a pretrained detector.
- Expose box, confidence, timestamp/frame, and camera/source ID.
- Detector should be swappable behind an interface.

### FR-3 Multi-object tracking
- Assign temporary track IDs.
- Maintain short-lived trajectories.
- Handle track expiry and re-association through the selected tracker.
- No cross-camera identity tracking.

### FR-4 Crowd feature extraction
At minimum compute per configured ROI/zone and/or incident neighbourhood:
- occupancy/person count;
- normalised density proxy;
- density change;
- mean speed and acceleration proxy;
- speed variance;
- directional dispersion/disorder;
- convergence;
- dispersal;
- counter-flow/opposing-direction evidence;
- congestion/stagnation proxy.

Features must be inspectable and exportable.

### FR-5 Temporal violence recognition
- Run on short rolling video clips/windows.
- Output a calibrated or calibratable violence/aggression score.
- Default model candidate: pretrained X3D-S fine-tuned for violent vs non-violent video.
- Model interface must allow a comparison model later.

### FR-6 Temporal buffer
- Maintain a rolling history of crowd and violence signals.
- Configurable window duration.
- Preserve raw and smoothed values needed for evaluation.

### FR-7 Incident association and fusion
- Associate evidence spatially and temporally.
- Avoid naive “OR” as the proposed method, though implement it as a baseline.
- Initial proposed method should use deterministic/rule-weighted fusion with persistence.
- Produce fused risk score, reason codes, and contributing feature values.

### FR-8 Incident lifecycle
Minimum states:
- `candidate`
- `active`
- `escalating`
- `critical`
- `resolving`
- `closed`

Transitions must be deterministic/configurable and testable.

### FR-9 Deduplication
- Repeated positive windows for the same event should update one incident.
- Do not create a new alert for every inference window.
- Persist incident start, last update, peak severity, and close time.

### FR-10 Severity
Minimum levels:
- low
- medium
- high
- critical

Severity must be based on multiple evidence signals rather than directly mirroring violence probability.

### FR-11 Evidence
Each alerted incident should support:
- camera/source ID;
- incident ID;
- start/update timestamps;
- severity;
- contributing reason codes;
- snapshot;
- short pre/post-event clip where possible;
- signal timeline;
- operator disposition.

### FR-12 Operator dashboard
Minimum MVP pages:
- source/camera list;
- current incident list;
- incident detail;
- evidence viewer;
- signal/severity timeline;
- acknowledge / dismiss / escalate controls.

### FR-13 Audit/evaluation export
- Persist machine evidence and operator outcome.
- Export incident/event records in structured format for offline evaluation.

## 9. Non-functional requirements

### NFR-1 Modularity
Detection, tracking, crowd features, violence inference, fusion, incident state, persistence, and UI must be independently testable.

### NFR-2 Reproducibility
Experiments must record:
- config;
- model checkpoint identifier;
- dataset split;
- thresholds;
- feature version;
- fusion version;
- metrics.

### NFR-3 Performance
Prioritise sustained processing and deterministic timing before adding complex models. Record:
- effective FPS;
- model inference latency;
- end-to-end alert latency;
- dropped/skipped frames.

### NFR-4 Explainability
Every incident escalation must expose machine-readable reason codes and feature values.

### NFR-5 Privacy
No facial recognition, demographic inference, or persistent identity tracking.

### NFR-6 Safety
No autonomous police/fire/medical dispatch. Human review is required before external escalation.

## 10. Explicitly out of scope

- Guaranteed prediction of a future stampede before observable cues.
- Face recognition.
- Demographic profiling.
- Weapon detection as a core requirement.
- Cross-camera person re-identification.
- City-scale multi-camera orchestration.
- Training a person detector from scratch.
- Training a large video transformer from scratch.
- One opaque end-to-end “stampede classifier.”
- Autonomous emergency-service dispatch.
- Audio analysis in the MVP.

## 11. Product success criteria

The project succeeds if the final prototype:

1. Processes a dense-crowd video end-to-end.
2. Visualises person tracks and crowd signals.
3. Produces rolling violence evidence.
4. Creates one persistent incident from related evidence.
5. Changes state/severity as evidence evolves.
6. Generates one evidence-backed alert rather than repeated duplicates.
7. Allows human disposition.
8. Stores an auditable incident timeline.
9. Supports direct experiments against the required baselines.
10. Demonstrates a better operational trade-off than independent detectors on the curated evaluation suite.

## 12. Required baselines

- B1: violence-only
- B2: crowd-only
- B3: naive OR
- B4: deterministic/rule fusion
- Proposed: calibrated temporal incident fusion with spatial association, persistence, severity, and lifecycle

If learned fusion is added, it becomes a separate proposed/experimental variant rather than replacing the deterministic baseline.
