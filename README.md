# Crowd Safety Monitoring

M2 extends the deterministic, single-process offline runner with optional person detection, source-local ByteTrack observations, and inspectable ROI crowd-signal timelines. It still has no violence model, fusion, incident engine, dashboard, or live-input path.

## Setup

```bash
python3.11 -m venv venv
venv/bin/python -m pip install -e '.[dev]'
```

The approved M2 perception path is `ultralytics==8.4.135` with `lap==0.5.13`, YOLO26n person detection, and Ultralytics' `bytetrack.yaml`. The model file is external and ignored by Git; record its SHA-256 in run metadata rather than committing it.

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

M2 uses the configured ROI polygon in resized pixel coordinates. `density_proxy` is an occupancy-per-pixel-area proxy, not people/m²; physical density requires camera calibration. Feature records use `insufficient` for short/empty trajectory history and `unavailable` when perception health is not usable. Those states are never converted to a normal zero-risk signal.

Feature definitions use the configured `window_s`: occupancy is current track count; density proxy is occupancy divided by ROI pixel area; density delta compares current occupancy with the latest pre-window occupancy; speed is centre displacement divided by elapsed timestamp; acceleration is the change in speed divided by elapsed timestamp; speed variance is the population variance of speed; direction disorder is one minus the resultant length of unit motion vectors; convergence/dispersal are fractions moving toward/away from the current ROI centroid; counter-flow is the normalized smaller opposing directional group; congestion is high occupancy with mean speed below the configured threshold. These are pixel/time proxies and return `null` when the required history is unavailable.

For the authorised development videos in this repository:

```bash
venv/bin/python -m crowd_safety process-video \
  --config configs/pipeline/dev.toml \
  --input videos/fighting1.mp4
```

The generated overlay shows ROI outlines, person boxes, temporary track IDs/trails, occupancy, and feature health. `metrics.json` and `metadata.json` include detector/tracker/feature timing, health, model/device, and checkpoint provenance. These are offline/model measurements and do not establish live readiness.

M2 evidence on the authorised `fighting1.mp4` clip: 510 source frames, 107 processed frames, 86 detector calls, 107 tracker calls, and a playable 640×360 output. With the direct detector-to-ByteTrack handoff, the clip produced one retained track observation and insufficient feature history at all processed timestamps. The observed sparse/fragmented tracks are recorded as insufficient evidence for a YOLO26s decision gate rather than as a detector-quality claim.

M2 evidence on the authorised normal-pedestrian `walking.mp4` clip: 1,723 source frames, 173 processed frames, 173 detector calls, 173 tracker calls, 678 track observations, 92 available and 81 insufficient feature rows, and a playable 640×360 output. The inspected overlay showed scaled person boxes, temporary IDs, ROI annotation, and feature health; the feature timeline produced movement values such as mean speed, direction disorder, convergence/dispersal, counter-flow, and congestion when history was sufficient. The run completed on CPU at 132.9 effective FPS. No separately authorised dense-normal or strong directional-change/running videos are available, so those acceptance categories remain unvalidated.

Source media, model files, environments, and generated artifacts are intentionally ignored by Git.
