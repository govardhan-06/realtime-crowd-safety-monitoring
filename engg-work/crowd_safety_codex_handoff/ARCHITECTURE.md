# System Architecture

## 1. End-to-end flow

```text
Recorded video / RTSP / webcam
            |
            v
      Video Ingestion
            |
            v
   Frame/Clip Scheduler
            |
      +-----+------+
      |            |
      v            v
Person Detection   Temporal Video Buffer
      |            |
      v            v
   Tracking    Violence Recognizer
      |            |
      v            |
Crowd Feature      |
 Extraction        |
      |            |
      +------v-----+
        Signal Bus
            |
            v
 Temporal Window / Smoothing
            |
            v
 Spatial-Temporal Fusion
            |
            v
    Incident State Engine
            |
       +----+----+
       |         |
       v         v
 Evidence     Persistence
 Capture      / Audit
       |         |
       +----v----+
        Backend API
            |
            v
       Web Dashboard
            |
            v
      Human Operator
```

## 2. Architecture principle

Neural models detect **signals**. Deterministic project-specific logic converts those signals into an **incident**.

Do not let model-specific code own incident state.

## 3. Recommended repository shape

Adapt this to the existing repository rather than forcing it mechanically.

```text
/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── domain/
│   │   │   ├── incidents/
│   │   │   ├── cameras/
│   │   │   └── evidence/
│   │   ├── pipeline/
│   │   │   ├── ingestion/
│   │   │   ├── detection/
│   │   │   ├── tracking/
│   │   │   ├── crowd/
│   │   │   ├── violence/
│   │   │   ├── fusion/
│   │   │   └── orchestration/
│   │   └── persistence/
│   └── tests/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── ml/
│   ├── configs/
│   ├── datasets/
│   ├── scripts/
│   ├── training/
│   ├── evaluation/
│   └── checkpoints/   # ignored / external
├── configs/
│   ├── cameras/
│   ├── pipeline/
│   └── fusion/
├── evaluation/
│   ├── manifests/
│   ├── annotations/
│   └── reports/
├── scripts/
└── docs/
```

Do not commit raw datasets, large video files, or model checkpoints.

## 4. Core domain contracts

These names are conceptual. Match local style.

### FramePacket

```python
FramePacket(
    source_id: str,
    frame_index: int,
    timestamp_s: float,
    image: ndarray,
)
```

### PersonDetection

```python
PersonDetection(
    box_xyxy: tuple[float, float, float, float],
    confidence: float,
)
```

### TrackObservation

```python
TrackObservation(
    source_id: str,
    track_id: int,
    timestamp_s: float,
    center_xy: tuple[float, float],
    box_xyxy: tuple[float, float, float, float],
    confidence: float,
)
```

### CrowdFeatureVector

```python
CrowdFeatureVector(
    source_id: str,
    region_id: str,
    timestamp_s: float,
    person_count: int,
    density: float,
    density_delta: float,
    mean_speed: float,
    speed_variance: float,
    direction_disorder: float,
    convergence: float,
    dispersal: float,
    counter_flow: float,
    congestion: float,
)
```

### ViolenceEvidence

```python
ViolenceEvidence(
    source_id: str,
    region_id: str | None,
    clip_start_s: float,
    clip_end_s: float,
    score: float,
    model_version: str,
)
```

### FusionEvidence

```python
FusionEvidence(
    source_id: str,
    region_id: str,
    timestamp_s: float,
    violence_score: float,
    crowd_features: CrowdFeatureVector,
    persistence_s: float,
    fused_risk: float,
    reason_codes: list[str],
)
```

### Incident

```python
Incident(
    id: UUID,
    source_id: str,
    region_id: str,
    state: IncidentState,
    severity: Severity,
    started_at: datetime,
    last_updated_at: datetime,
    closed_at: datetime | None,
    peak_risk: float,
    reason_codes: list[str],
    evidence_refs: list[str],
    operator_status: str | None,
)
```

## 5. Video scheduling

V1 should use a **single-process deterministic offline runner** before introducing queues.

Recommended conceptual loop:

```text
decode frame
  -> run detector at configured cadence
  -> update tracker
  -> update crowd feature state
  -> append frame to clip buffer
  -> when violence inference cadence is reached, infer clip
  -> emit timestamp-aligned signals
  -> update fusion window
  -> update incident state
  -> persist/export outputs
```

Do not introduce Celery/Kafka/Redis purely for architectural appearance.

## 6. Regions of interest

Support rectangular/polygonal configured ROIs.

Use ROIs for:
- zone occupancy;
- local density;
- exits/bottlenecks;
- incident neighbourhood aggregation.

Pixel-space quantities are acceptable for MVP. Do not claim real-world people-per-square-metre density unless camera geometry is calibrated.

## 7. Tracking-derived movement

Derive movement from temporally smoothed track centres.

Potential feature definitions:

- speed proxy: pixel displacement / elapsed time;
- acceleration proxy: speed delta;
- direction: angle of displacement vector;
- direction disorder: circular dispersion / entropy-like measure;
- convergence: fraction of track velocity vectors pointing toward local centroid/focal region;
- dispersal: fraction pointing outward;
- counter-flow: simultaneous strong directional groups;
- congestion: high local occupancy with low mean motion.

Feature implementations must:
- declare the time window used;
- handle insufficient tracks;
- normalise where possible;
- be covered by synthetic unit tests.

## 8. Violence inference

Default path:
- pretrained X3D-S backbone;
- replace classification head;
- fine-tune violent vs non-violent;
- fixed clip duration/sample rate defined by config;
- export score plus model/checkpoint version.

Do not couple the rest of the system to X3D-specific tensor shapes.

## 9. Signal alignment

Crowd and violence signals run at different cadences.

Create a common timestamp-aligned signal record. Missing values should be explicit rather than silently copied.

Use smoothing only where configured and preserve raw values for evaluation.

## 10. Fusion V1

Start with transparent fusion.

Example conceptual formulation:

```text
risk =
    w_v * smoothed_violence
  + w_d * density_risk
  + w_m * movement_abnormality
  + w_c * convergence_or_dispersal
  + w_p * persistence
```

Exact weights/thresholds are experimental configuration, not hard-coded constants.

Important:
- a severe crowd-only incident must be possible;
- violence should not be mandatory for crowd-risk escalation;
- brief isolated signals should decay/suppress;
- spatially related crowd response may increase the severity of a violent event.

## 11. Incident lifecycle

Suggested transitions:

```text
closed -> candidate
candidate -> active | closed
active -> escalating | resolving
escalating -> critical | resolving
critical -> resolving
resolving -> active | closed
```

Transition policy should consider:
- fused risk thresholds;
- minimum persistence;
- hysteresis;
- quiet/decay interval;
- evidence continuity.

State transitions must be logged.

## 12. Evidence buffer

Maintain a ring buffer of recent encoded frames or references sufficient for a pre-event clip.

On alert:
- capture representative snapshot;
- save N seconds before;
- continue N seconds after;
- attach to the incident rather than creating a new incident.

Use configurable retention.

## 13. API boundary

Suggested endpoints for the MVP:

```text
GET  /health
GET  /sources
POST /runs
GET  /runs/{id}
GET  /incidents
GET  /incidents/{id}
POST /incidents/{id}/acknowledge
POST /incidents/{id}/dismiss
POST /incidents/{id}/escalate
GET  /incidents/{id}/timeline
GET  /incidents/{id}/evidence
```

For offline development, a `run` can represent processing one input video.

## 14. Persistence

Prefer PostgreSQL for the final integrated prototype.

During early ML-only milestones, structured JSON/Parquet/CSV outputs are sufficient. Do not block model/pipeline work on database setup.

Suggested persistent entities:
- source;
- processing run;
- incident;
- incident event/state transition;
- evidence artifact;
- operator action;
- model/config metadata.

## 15. Frontend

Use Next.js for:
- source/run status;
- active/recent incidents;
- incident detail;
- evidence;
- reason codes;
- signal timeline;
- operator actions.

Do not build the dashboard before the pipeline emits stable incident contracts.

## 16. Observability

At minimum record:
- frame decode time;
- detector latency;
- tracker update latency;
- crowd feature latency;
- violence model latency;
- fusion latency;
- effective FPS;
- skipped/dropped frames;
- incident/alert timestamps.

This is necessary for the final latency analysis.
