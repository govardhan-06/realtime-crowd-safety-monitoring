from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from uuid import uuid4

from .artifacts import config_hash, resolved_config, write_json
from .config import PipelineConfig
from .scheduling import FrameScheduler
from .video import VideoReader, VideoWriter


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config_hash: str
    run_directory: Path
    video_path: Path
    frames_path: Path
    metadata_path: Path
    metrics_path: Path


def benchmark_video(config: PipelineConfig, input_override: str | Path | None = None) -> Path:
    benchmark_directory = config.output_directory / "benchmarks"
    benchmark_directory.mkdir(parents=True, exist_ok=True)
    benchmark_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    benchmark_path = benchmark_directory / f"{benchmark_id}.json"
    started_at = _utc_now()
    try:
        result = process_video(config, input_override)
        metrics = json.loads(result.metrics_path.read_text())
        values = {
            "benchmark_id": benchmark_id,
            "status": "success",
            "started_at": started_at,
            "ended_at": _utc_now(),
            "run_id": result.run_id,
            "input_path": str(input_override or config.input_path),
            **{key: metrics[key] for key in (
                "effective_fps",
                "decode_seconds",
                "write_seconds",
                "processed_frame_count",
                "skipped_frame_count",
            )},
        }
    except Exception as exc:
        values = {
            "benchmark_id": benchmark_id,
            "status": "failed",
            "started_at": started_at,
            "ended_at": _utc_now(),
            "input_path": str(input_override or config.input_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json(benchmark_path, values)
    return benchmark_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_video(config: PipelineConfig, input_override: str | Path | None = None) -> RunResult:
    input_path = Path(input_override).expanduser().resolve() if input_override else config.input_path
    resolved = resolved_config(config, input_path)
    digest = config_hash(resolved)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_directory = config.output_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    video_path = run_directory / "annotated.mp4"
    frames_path = run_directory / "frames.jsonl"
    metadata_path = run_directory / "metadata.json"
    metrics_path = run_directory / "metrics.json"
    started_at = _utc_now()
    start_monotonic = time.perf_counter()
    processed_count = 0
    skipped_count = 0
    decode_seconds = 0.0
    write_seconds = 0.0

    with frames_path.open("w") as frames_output, VideoReader(input_path) as reader:
        source_metadata = {
            "path": str(input_path),
            "source_id": reader.source_id,
            "width": reader.width,
            "height": reader.height,
            "fps": reader.fps,
            "frame_count": reader.frame_count,
        }
        scheduler = FrameScheduler(config.target_fps)
        with VideoWriter(video_path, config.resize, config.target_fps, config.annotation_enabled) as writer:
            packets = iter(reader)
            while True:
                decode_start = time.perf_counter()
                try:
                    packet = next(packets)
                except StopIteration:
                    break
                decode_seconds += time.perf_counter() - decode_start
                process, schedule_time = scheduler.decide(packet.timestamp_s)
                if process:
                    write_start = time.perf_counter()
                    writer.write(packet)
                    write_seconds += time.perf_counter() - write_start
                    processed_count += 1
                else:
                    skipped_count += 1
                frames_output.write(json.dumps({
                    "source_id": packet.source_id,
                    "frame_index": packet.frame_index,
                    "timestamp_s": packet.timestamp_s,
                    "processed": process,
                    "schedule_time_s": schedule_time,
                }, sort_keys=True) + "\n")

    elapsed_seconds = time.perf_counter() - start_monotonic
    ended_at = _utc_now()
    write_json(
        run_directory / "config.json",
        {"config_hash": digest, "config": resolved},
    )
    write_json(
        metadata_path,
        {
            "run_id": run_id,
            "config_hash": digest,
            "started_at": started_at,
            "ended_at": ended_at,
            "input": source_metadata,
            "artifacts": {
                "video": video_path.name,
                "frames": frames_path.name,
                "config": "config.json",
                "metrics": metrics_path.name,
            },
        },
    )
    write_json(
        metrics_path,
        {
            "run_id": run_id,
            "config_hash": digest,
            "source_frame_count": processed_count + skipped_count,
            "processed_frame_count": processed_count,
            "skipped_frame_count": skipped_count,
            "output_frame_count": processed_count,
            "decode_seconds": decode_seconds,
            "write_seconds": write_seconds,
            "total_seconds": elapsed_seconds,
            "effective_fps": (processed_count + skipped_count) / elapsed_seconds if elapsed_seconds else 0.0,
        },
    )
    return RunResult(run_id, digest, run_directory, video_path, frames_path, metadata_path, metrics_path)
