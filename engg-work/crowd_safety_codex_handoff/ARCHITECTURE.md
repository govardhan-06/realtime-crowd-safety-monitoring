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
   (YOLO26)        |
      |            v
      v       Violence Recognizer
   ByteTrack      (generic adapter)
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
       +----+----------------+
       |                     |
       v                     v
 Evidence                Persistence
 Capture                 / Audit
       |
       +----------+
                  |
                  v
         Optional VLM Explainer
         (non-authoritative)
                  |
                  v
             Backend API
                  |
                  v
             Web Dashboard
                  |
                  v
            Human Operator
```

## 2. Architecture principles

1. Neural models detect **signals**.
2. Deterministic project-specific logic converts those signals into an **incident**.
3. Model availability/health is explicit.
4. A missing signal is not silently converted into a normal/zero-risk signal.
5. The VLM explains already-created incidents; it does not own incident creation, severity, state, or escalation.

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
│   │   │   ├── explanation/
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
    score: float | None,
    model_version: str,
    status: str,  # available | degraded | unavailable
)
```

### FusionEvidence

```python
FusionEvidence(
    source_id: str,
    region_id: str,
    timestamp_s: float,
    violence_score: float | None,
    violence_status: str,
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

### IncidentExplanation

```python
IncidentExplanation(
    incident_id: UUID,
    generated_at: datetime,
    provider: str,
    model_version: str,
    text: str,
    status: str,  # generated | unavailable | failed
)
```

`IncidentExplanation` is supplementary. It must not be consumed by the incident state engine.

## 5. Video scheduling

V1 should use the completed **single-process deterministic offline runner** from M1 before introducing queues.

Recommended conceptual loop:

```text
decode frame
  -> run detector at configured cadence
  -> update tracker
  -> update crowd feature state
  -> append frame to clip buffer
  -> when violence inference cadence is reached, infer clip
  -> emit timestamp-aligned signals + stage health
  -> update fusion window
  -> update incident state
  -> capture/persist evidence
  -> optionally request VLM explanation for created/alerted incident
  -> export outputs
```

Do not introduce Celery/Kafka/Redis purely for architectural appearance.

## 6. Person detection and tracking

Default detector:
- Ultralytics YOLO26n pretrained checkpoint;
- class filter restricted to `person`;
- use YOLO26s only if measured dense-scene misses justify the additional compute.

Default tracker:
- ByteTrack.

Rules:
- no detector fine-tuning in M2;
- no tracker training;
- detector/tracker remain replaceable adapters;
- track IDs are camera/source-local and temporary;
- no cross-camera re-identification.

## 7. Regions of interest

Support rectangular/polygonal configured ROIs.

Use ROIs for:
- zone occupancy;
- local density;
- exits/bottlenecks;
- incident neighbourhood aggregation.

Pixel-space quantities are acceptable for MVP. Do not claim real-world people-per-square-metre density unless camera geometry is calibrated.

## 8. Tracking-derived movement

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

## 9. Violence inference

The rest of the application must depend on a generic binary temporal-video interface, not on one architecture.

### M3A — training-independent delivery baseline

Integrate a ready-made violence-classification checkpoint first.

Initial development candidate:
- `mitegvg/videomae-small-kinetics-binary-finetuned-xd-violence`

Why:
- compact VideoMAE-style transformer checkpoint;
- standard Hugging Face video-classification interface;
- suitable for unblocking end-to-end integration.

Rules:
- verify license and label mapping before use;
- pin checkpoint revision when the integration is stable;
- benchmark it on the project's own dev clips;
- do not present the community checkpoint's published metrics as our result;
- do not let poor M3A accuracy block pipeline engineering.

### M3B — bounded transfer-learning experiment

Default academic training experiment:
- pretrained X3D-S backbone;
- replace classification head with violent/non-violent head;
- train head with backbone frozen;
- optionally unfreeze later blocks only if validation justifies it;
- fixed clip duration/sample rate defined by config;
- record checkpoint, config, split, and metrics.

M3B is important as an experiment, but M4-M6 must remain runnable with M3A if M3B is delayed or underperforms.

## 10. Model/stage health

Each ML stage emits:
- model/checkpoint identifier;
- inference timestamp;
- latency;
- status: `available`, `degraded`, or `unavailable`;
- score/output only when valid.

Never convert an unavailable stage into a fabricated zero score.

Fusion may:
- continue with available evidence;
- reduce confidence;
- emit a health-related reason/status;
- or suppress a decision if configured evidence requirements are not met.

The behavior must be deterministic and testable.

## 11. Signal alignment

Crowd and violence signals run at different cadences.

Create a common timestamp-aligned signal record. Missing values should be explicit rather than silently copied.

Use smoothing only where configured and preserve raw values for evaluation.

## 12. Fusion V1

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
- spatially related crowd response may increase the severity of a violent event;
- unavailable violence evidence is different from low violence evidence.

## 13. Incident lifecycle

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
- evidence continuity;
- required signal availability where configured.

State transitions must be logged.

## 14. Evidence buffer

Maintain a ring buffer of recent encoded frames or references sufficient for a pre-event clip.

On alert:
- capture representative snapshot;
- save N seconds before;
- continue N seconds after;
- attach to the incident rather than creating a new incident.

Use configurable retention.

## 15. VLM explanation layer

Purpose:
- convert already-captured incident evidence into a concise operator-facing explanation.

Inputs may include:
- evidence clip;
- selected keyframes;
- incident time range;
- deterministic reason codes;
- selected numerical signal summaries.

Output:
- short description of observable behavior;
- optional concise explanation of why the alert warrants review.

Hard boundary:
- VLM output does not feed back into fusion;
- VLM output cannot create/close an incident;
- VLM output cannot alter severity/state;
- deterministic reason codes remain authoritative;
- VLM failure must not block alert creation.

Default provider may be Gemini video/image understanding if API credentials are available, but the interface must remain provider-swappable and disabled in unit tests.

## 16. API boundary

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
GET  /incidents/{id}/explanation
```

For offline development, a `run` can represent processing one input video.

## 17. Persistence

Prefer PostgreSQL for the final integrated prototype.

During early ML-only milestones, structured JSON/Parquet/CSV outputs are sufficient. Do not block model/pipeline work on database setup.

Suggested persistent entities:
- source;
- processing run;
- incident;
- incident event/state transition;
- evidence artifact;
- optional incident explanation;
- operator action;
- model/config metadata.

## 18. Frontend

Use Next.js for:
- source/run status;
- active/recent incidents;
- incident detail;
- evidence;
- deterministic reason codes;
- signal timeline;
- optional generated explanation labelled as AI-generated;
- operator actions.

Do not build the dashboard before the pipeline emits stable incident contracts.

## 19. Observability

At minimum record:
- frame decode time;
- detector latency and health;
- tracker update latency and health;
- crowd feature latency;
- violence model latency and health;
- fusion latency;
- evidence-generation latency;
- VLM latency/status when enabled;
- effective FPS;
- skipped/dropped frames;
- incident/alert timestamps.

This is necessary for final latency and degraded-mode analysis.