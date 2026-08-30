from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from uuid import uuid4
from importlib.metadata import PackageNotFoundError, version

import cv2

from .annotations import annotate_frame
from .artifacts import config_hash, resolved_config, write_json
from .config import PipelineConfig
from .crowd_features import compute_crowd_features
from .detection import PersonDetector, UltralyticsPersonDetector
from .scheduling import FrameScheduler
from .tracking import ByteTrackTracker, Tracker
from .types import CrowdFeatureRecord, PersonDetection, StageHealth, TrackObservation
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
    tracks_path: Path | None = None
    features_path: Path | None = None


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
                "detector_seconds",
                "tracker_seconds",
                "crowd_feature_seconds",
                "detector_calls",
                "tracker_calls",
                "crowd_feature_calls",
            )},
            "stage_health": metrics["stage_health"],
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


def _resize_detections(
    detections: tuple[PersonDetection, ...],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[PersonDetection, ...]:
    x_scale = target_width / source_width
    y_scale = target_height / source_height
    return tuple(
        PersonDetection(
            detection.source_id,
            detection.frame_index,
            detection.timestamp_s,
            (
                detection.box_xyxy[0] * x_scale,
                detection.box_xyxy[1] * y_scale,
                detection.box_xyxy[2] * x_scale,
                detection.box_xyxy[3] * y_scale,
            ),
            detection.confidence,
        )
        for detection in detections
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def process_video(
    config: PipelineConfig,
    input_override: str | Path | None = None,
    *,
    detector: PersonDetector | None = None,
    tracker: Tracker | None = None,
) -> RunResult:
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
    m2_enabled = config.perception.enabled or detector is not None or tracker is not None
    tracks_path = run_directory / "tracks.jsonl" if m2_enabled else None
    features_path = run_directory / "features.jsonl" if m2_enabled else None
    started_at = _utc_now()
    start_monotonic = time.perf_counter()
    processed_count = 0
    skipped_count = 0
    decode_seconds = 0.0
    write_seconds = 0.0
    detector_seconds = 0.0
    tracker_seconds = 0.0
    feature_seconds = 0.0
    detector_calls = 0
    tracker_calls = 0
    feature_calls = 0
    detector_health: StageHealth | None = None
    tracker_health: StageHealth | None = None
    feature_health: StageHealth | None = None
    track_history: dict[int, list[TrackObservation]] = {}
    detection_interval = 1.0 / config.perception.cadence_fps
    last_detection_timestamp: float | None = None
    last_tracks: tuple[TrackObservation, ...] = ()
    if m2_enabled:
        detector = detector or UltralyticsPersonDetector(
            model=config.perception.model,
            confidence=config.perception.confidence,
            device=config.perception.device,
            person_class_id=config.perception.person_class_id,
        )
        tracker = tracker or ByteTrackTracker(
            model=getattr(detector, "model", None),
            tracker_config=config.tracking.config,
            track_buffer=config.tracking.track_buffer,
            confidence=config.perception.confidence,
            device=getattr(detector, "device", config.perception.device),
            person_class_id=config.perception.person_class_id,
        )

    tracks_output = tracks_path.open("w") if tracks_path else None
    features_output = features_path.open("w") if features_path else None
    try:
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
        with VideoWriter(video_path, config.resize, config.target_fps, annotate=False) as writer:
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
                    image = packet.image
                    if (image.shape[1], image.shape[0]) != config.resize:
                        image = cv2.resize(image, config.resize, interpolation=cv2.INTER_AREA)
                    tracking_packet = packet.__class__(
                        packet.source_id, packet.frame_index, packet.timestamp_s, image
                    )
                    tracks: tuple[TrackObservation, ...] = ()
                    features: tuple[CrowdFeatureRecord, ...] = ()
                    if m2_enabled:
                        if last_detection_timestamp is None or packet.timestamp_s - last_detection_timestamp >= detection_interval - 1e-9:
                            detector_result = detector.detect(packet)
                            detector_health = detector_result.health
                            detector_seconds += (detector_health.latency_ms or 0.0) / 1000.0
                            detector_calls += 1
                            last_detections = _resize_detections(
                                detector_result.detections,
                                packet.image.shape[1], packet.image.shape[0],
                                config.resize[0], config.resize[1],
                            )
                            last_detection_timestamp = packet.timestamp_s
                            tracking_result = tracker.update(tracking_packet, last_detections)
                            tracker_health = tracking_result.health
                            tracker_seconds += (tracker_health.latency_ms or 0.0) / 1000.0
                            tracker_calls += 1
                            last_tracks = tracking_result.observations
                        else:
                            last_detections = ()
                        tracks = last_tracks if last_detection_timestamp == packet.timestamp_s else ()
                        cutoff = packet.timestamp_s - config.crowd.window_s
                        for track_id, history in list(track_history.items()):
                            recent = [item for item in history if item.timestamp_s >= cutoff]
                            prior = [item for item in history if item.timestamp_s < cutoff]
                            if recent:
                                track_history[track_id] = ([prior[-1]] if prior else []) + recent
                            else:
                                del track_history[track_id]
                        for observation in tracks:
                            history = track_history.setdefault(observation.track_id, [])
                            history.append(observation)
                        history_values = tuple(tuple(items) for items in track_history.values())
                        feature_values: list[CrowdFeatureRecord] = []
                        feature_start = time.perf_counter()
                        for roi in config.crowd.rois:
                            if detector_health and detector_health.status != "available" or tracker_health.status != "available":
                                feature = CrowdFeatureRecord(
                                    packet.source_id, roi.name, packet.timestamp_s, "unavailable",
                                    detail="perception stage unavailable or degraded",
                                )
                            else:
                                feature = compute_crowd_features(
                                    tuple(item for items in history_values for item in items),
                                    roi,
                                    packet.timestamp_s,
                                    window_s=config.crowd.window_s,
                                    min_track_history=config.crowd.min_track_history,
                                    min_speed_px_s=config.crowd.min_speed_px_s,
                                    congestion_occupancy=config.crowd.congestion_occupancy,
                                    congestion_speed_px_s=config.crowd.congestion_speed_px_s,
                                    source_id=packet.source_id,
                                )
                            feature_values.append(feature)
                        features = tuple(feature_values)
                        feature_seconds += time.perf_counter() - feature_start
                        feature_calls += 1
                        feature_health = StageHealth(
                            "crowd_features",
                            "degraded" if any(item.status == "unavailable" for item in features) else "available",
                            detail="ROI features are pixel-space proxies; calibration is not applied",
                            latency_ms=(time.perf_counter() - feature_start) * 1000.0,
                        )
                        if tracks_output:
                            tracks_output.write(json.dumps({
                                "source_id": packet.source_id,
                                "frame_index": packet.frame_index,
                                "timestamp_s": packet.timestamp_s,
                                "health": asdict(tracker_health),
                                "observations": [asdict(item) for item in tracks],
                            }, sort_keys=True) + "\n")
                        if features_output:
                            features_output.write(json.dumps({
                                "source_id": packet.source_id,
                                "frame_index": packet.frame_index,
                                "timestamp_s": packet.timestamp_s,
                                "health": asdict(feature_health),
                                "features": [asdict(item) for item in features],
                            }, sort_keys=True) + "\n")
                    write_start = time.perf_counter()
                    if config.annotation_enabled:
                        image = annotate_frame(
                            image,
                            packet.frame_index,
                            packet.timestamp_s,
                            tracks=tracks,
                            histories=history_values if m2_enabled else (),
                            rois=config.crowd.rois if m2_enabled else (),
                            features=features,
                        )
                    writer.write(packet, image=image)
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
    finally:
        if tracks_output:
            tracks_output.close()
        if features_output:
            features_output.close()
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
                **({
                    "tracks": tracks_path.name,
                    "features": features_path.name,
                } if m2_enabled else {}),
            },
            "stages": {
                "detector": asdict(detector_health) if detector_health else None,
                "tracker": asdict(tracker_health) if tracker_health else None,
                "crowd_features": asdict(feature_health) if feature_health else None,
            },
            "provenance": ({
                "ultralytics": _package_version("ultralytics"),
                "lap": _package_version("lap"),
                "tracker_config": config.tracking.config,
                "track_buffer": config.tracking.track_buffer,
                "checkpoint_sha256": detector_health.checkpoint_sha256 if detector_health else None,
            } if m2_enabled else None),
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
            "detector_seconds": detector_seconds,
            "tracker_seconds": tracker_seconds,
            "crowd_feature_seconds": feature_seconds,
            "detector_calls": detector_calls,
            "tracker_calls": tracker_calls,
            "crowd_feature_calls": feature_calls,
            "stage_health": {
                "detector": asdict(detector_health) if detector_health else {"status": "disabled"},
                "tracker": asdict(tracker_health) if tracker_health else {"status": "disabled"},
                "crowd_features": asdict(feature_health) if feature_health else {"status": "disabled"},
            },
            "total_seconds": elapsed_seconds,
            "effective_fps": (processed_count + skipped_count) / elapsed_seconds if elapsed_seconds else 0.0,
        },
    )
    return RunResult(run_id, digest, run_directory, video_path, frames_path, metadata_path, metrics_path, tracks_path, features_path)
