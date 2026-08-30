# Milestone 1 — Offline Video Foundation

## Objective
Create the deterministic execution backbone for all later perception and incident work.

## Required work
- Python project/environment setup.
- Config system for input path, output path, resize, target processing FPS, logging.
- Video reader with reliable frame index and timestamps.
- Video writer/annotation abstraction.
- Pipeline interface/stage abstraction without overengineering.
- Run metadata (`run_id`, input metadata, start/end, config hash).
- Structured JSONL/CSV output.
- Timing instrumentation.
- CLI entry point.
- Unit tests for timestamp/frame scheduling logic.
- Small integration test using a synthetic generated video.

## CLI target
Example only; fit repo conventions:

```bash
python -m app process-video --config configs/pipeline/dev.yaml --input path/video.mp4
```

## Acceptance criteria
- Same config/input produces deterministic frame scheduling.
- Can process start-to-finish without detector/model dependencies.
- Output media preserves meaningful timestamps/frame count.
- Processing metrics are written.
- No raw dataset media added to Git.
- Tests run without GPU.

## Do not do yet
- Do not build dashboard.
- Do not train models.
- Do not introduce distributed queues.
- Do not implement incident rules.
