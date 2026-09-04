# Crowd Safety Monitoring

M4 extends the deterministic, single-process offline runner with timestamp-aligned temporal fusion and a deduplicated incident lifecycle. M5 adds bounded evidence capture, durable import/API records, a human-review dashboard, audited dispositions, and an optional-but-disabled explanation state. Live input, autonomous dispatch, and VLM credentials remain out of scope.

## Setup

```bash
python3.11 -m venv venv
venv/bin/python -m pip install -e '.[dev]'

# Required for the M3A pretrained violence adapter:
venv/bin/python -m pip install -e '.[violence]'
```

The approved M2 perception path is `ultralytics==8.4.135` with `lap==0.5.13`, YOLO26n person detection, and Ultralytics' `bytetrack.yaml`. M3A uses `Nikeytas/videomae-crime-detector-fixed-format` at revision `5d6d18cf0cabd4bd01c98edc9d68288590afd24f` as the current fallback after the original `mitegvg/videomae-small-kinetics-binary-finetuned-xd-violence` candidate returned 401/repository-not-found during preflight. The fallback model card documents MIT licensing and `Non-Violent Incident`/`Violent Crime` labels; its reported metrics are not project results.

## Run

Validate the example TOML configuration:

```bash
venv/bin/python -m crowd_safety validate-config --config configs/pipeline/dev.toml
```

Process a local video:

```bash
venv/bin/python -m crowd_safety process-video \
  --config configs/pipeline/dev.toml \
  --input path/to/video.mp4
```

Record a timestamped offline benchmark artifact for an authorised local video:

```bash
venv/bin/python -m crowd_safety benchmark \
  --config configs/pipeline/dev.toml \
  --input path/to/video.mp4
```

The benchmark records success/failure, effective FPS, decode/write timing, and processed/skipped frame counts. It does not represent live or model performance.

Each run creates an ignored `artifacts/<run-id>/` directory containing:

- `annotated.mp4` — resized output at the configured processing FPS;
- `frames.jsonl` — source frame index/timestamp and processed/skipped decision;
- `config.json` — resolved settings and SHA-256 config hash;
- `metadata.json` — run/source metadata and artifact names;
- `metrics.json` — counts, decode/write timing, total time, and effective FPS.
- `tracks.jsonl` — one record per processed timestamp with project-owned source-local track observations and tracker health when M2 is enabled;
- `features.jsonl` — one record per processed timestamp and configured ROI with crowd features and feature-stage health when M2 is enabled.
- `violence.jsonl` — timestamp-aligned rolling clip evidence with score, status, label mapping, latency, and stage health when M3A is enabled.
- `fusion.jsonl` — raw crowd/violence health, normalized and smoothed values, fused risk, strategy, and deterministic reason codes when M4 is active.
- `incidents.jsonl` — incident snapshots keyed by source/ROI, with lifecycle state, severity, peak risk, and accumulated reason codes.
- `transitions.jsonl` — ordered lifecycle/severity transition records with timestamp, cause, and reason codes.

M2 uses the configured ROI polygon in resized pixel coordinates. `density_proxy` is an occupancy-per-pixel-area proxy, not people/m²; physical density requires camera calibration. Feature records use `insufficient` for short/empty trajectory history and `unavailable` when perception health is not usable. Violence evidence uses `available`, `degraded`, or `unavailable`; an unavailable/degraded result retains `score = null` when no valid model output exists and is never converted to a normal zero-risk signal.

Feature definitions use the configured `window_s`: occupancy is current track count; density proxy is occupancy divided by ROI pixel area; density delta compares current occupancy with the latest pre-window occupancy; speed is centre displacement divided by elapsed timestamp; acceleration is the change in speed divided by elapsed timestamp; speed variance is the population variance of speed; direction disorder is one minus the resultant length of unit motion vectors; convergence/dispersal are fractions moving toward/away from the current ROI centroid; counter-flow is the normalized smaller opposing directional group; congestion is high occupancy with mean speed below the configured threshold. These are pixel/time proxies and return `null` when the required history is unavailable.

For the authorised development videos in this repository:

```bash
venv/bin/python -m crowd_safety process-video \
  --config configs/pipeline/dev.toml \
  --input videos/fighting1.mp4
```

The generated overlay shows ROI outlines, person boxes, temporary track IDs/trails, occupancy, crowd-feature health, and the latest violence status/score. `metrics.json` and `metadata.json` include detector/tracker/feature/violence timing, health, model/device, revision, label mapping, and checkpoint provenance. These are offline/model measurements and do not establish live readiness.

M3A uses 3.2-second clips sampled to 16 unique buffered frames at the example runner's 5 FPS cadence and runs at the configured one-second cadence. The clip bounds in `violence.jsonl` are the actual first/last packet timestamps. The adapter validates the model's binary label mapping and exports only the generic unsafe probability; model-specific tensors do not cross the adapter boundary. The configured threshold is recorded for downstream evaluation and does not create incidents by itself.

M4 is configured in `[fusion]` in TOML. `strategy` selects `violence-only`, `crowd-only`, `naive-or`, `rule-fusion`, or `temporal`; smoothing, normalization bounds, weights, persistence, hysteresis, decay, quiet time, lifecycle thresholds, severity boundaries, source/ROI association, and crowd-only degraded-mode policy are all recorded in `config.json` and its hash. A missing violence result has `score = null`; stale evidence retains its raw score/status but exposes `effective_violence_score = null`, with explicit health/reason codes. Neither is treated as negative evidence. The proposed temporal strategy updates one in-memory incident per source/ROI and writes ordered transitions rather than one alert per window. All strategies consume the same stored signal stream.

Replay stored signals without rerunning models:

```bash
venv/bin/python -m crowd_safety replay \
  --run-directory artifacts/<run-id> \
  --config configs/pipeline/dev.toml
```

This writes `replay/<strategy>/{fusion,incidents,transitions}.jsonl` plus replay metadata for all five strategies, or one strategy with `--strategy`. Replay rejects a changed resolved-config hash. Fusion and incidents are deterministic safety-assistance evidence only; no VLM or generated text participates in state, severity, or escalation.

M2 evidence on the authorised `fighting1.mp4` clip: 510 source frames, 107 processed frames, 86 detector calls, 107 tracker calls, and a playable 640×360 output. With the direct detector-to-ByteTrack handoff, the clip produced one retained track observation and insufficient feature history at all processed timestamps. The observed sparse/fragmented tracks are recorded as insufficient evidence for a YOLO26s decision gate rather than as a detector-quality claim.

M2 evidence on the authorised normal-pedestrian `walking.mp4` clip: 1,723 source frames, 173 processed frames, 173 detector calls, 173 tracker calls, 678 track observations, 92 available and 81 insufficient feature rows, and a playable 640×360 output. The inspected overlay showed scaled person boxes, temporary IDs, ROI annotation, and feature health; the feature timeline produced movement values such as mean speed, direction disorder, convergence/dispersal, counter-flow, and congestion when history was sufficient. The run completed on CPU at 132.9 effective FPS. No separately authorised dense-normal or strong directional-change/running videos are available, so those acceptance categories remain unvalidated.

## M5 review flow

After a video run emits an incident, M5 writes `snapshot.jpg`, bounded `pre_event.mp4`/`post_event.mp4` clips, and a `manifest.json` below the configured `m5.evidence_root`. Evidence failures remain explicit in the manifest and do not remove the incident. Retention is applied to old run directories under that root.

Import an offline run into PostgreSQL by setting the configured environment variable:

```bash
export DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:5432/crowd_safety'
venv/bin/python -m crowd_safety import-run \
  --config configs/pipeline/dev.toml \
  --run-directory artifacts/<run-id>
```

Run the local review API and dashboard with PostgreSQL. For a disposable local demo only, add `--ephemeral` to keep records in memory:

```bash
venv/bin/python -m crowd_safety serve-api --config configs/pipeline/dev.toml
(cd frontend && npm run dev)
```

The API keeps deterministic incident state, transitions, reason codes, timeline, and evidence separate from generated explanation text and append-only human actions. `escalate` records an internal human request; it never contacts emergency services. The configured development VLM path is disabled.

Source media, model files, environments, and generated artifacts are intentionally ignored by Git.

The specified `mitegvg` checkpoint was rejected during preflight with a current 401/repository-not-found response, so it remains documented as an unsuitable candidate. The approved fallback checkpoint was downloaded externally and ran successfully on all 11 MP4s currently under `videos/` with zero process failures and `available` evidence on every emitted window. The sweep is engineering/preflight evidence, not a labelled accuracy evaluation: the clips do not have a checked-in ground-truth manifest. The provisional `0.5` threshold overlaps the observed score ranges (for example, `walking.mp4` mean `0.5524`, `fighting1.mp4` mean `0.4221`), so calibration and false-positive/false-negative claims remain deferred to M6A. M3B X3D-S training remains deferred until the M6A POC gate.

The fallback model file remains in the external Hugging Face cache, not Git. Its downloaded safetensors SHA-256 is `ff542a5aa37d4c447584523545996d7c186d87c71b70decae0a773a02f212e5c`. The configured benchmark on `walking.mp4` recorded 31 violence windows, `available` health, 403.3 ms last-call latency, and 13.27 s total violence-model time; these are host-specific offline measurements.
