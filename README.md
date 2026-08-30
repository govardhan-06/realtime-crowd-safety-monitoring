# Crowd Safety Monitoring

M1 provides a deterministic, single-process offline video runner. It has no detector, tracker, model, incident engine, dashboard, or live-input dependency.

## Setup

```bash
python3.11 -m venv venv
venv/bin/python -m pip install -e '.[dev]'
```

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

Source media, model files, environments, and generated artifacts are intentionally ignored by Git.
