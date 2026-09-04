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
from .evidence import capture_run_evidence
from .fusion import FUSION_VERSION, FusionBuilder
from .incidents import IncidentEngine
from .scheduling import FrameScheduler
from .tracking import ByteTrackTracker, Tracker
from .types import CrowdFeatureRecord, PersonDetection, StageHealth, TrackObservation, ViolenceEvidence
from .video import VideoReader, VideoWriter
from .violence import RollingClipBuffer, VideoMAEViolenceClassifier, ViolenceCadence, ViolenceClassifier


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
    violence_path: Path | None = None
    fusion_path: Path | None = None
    incidents_path: Path | None = None
    transitions_path: Path | None = None
    evidence_paths: tuple[Path, ...] = ()


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
                "violence_seconds",
                "detector_calls",
                "tracker_calls",
                "crowd_feature_calls",
                "violence_calls",
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


def _violence_failure_evidence(
    window,
    classifier: ViolenceClassifier,
    config: PipelineConfig,
    detail: str,
    latency_ms: float,
) -> ViolenceEvidence:
    labels = tuple(getattr(classifier, "label_mapping", ())) or tuple(
        (label, index) for index, label in enumerate(config.violence.labels)
    )
    return ViolenceEvidence(
        window.packets[0].source_id,
        None,
        window.start_s,
        window.end_s,
        None,
        str(getattr(classifier, "model_name", config.violence.model)),
        str(getattr(classifier, "revision", config.violence.revision)),
        labels,
        "degraded",
        latency_ms,
        detail,
    )


def _violence_provenance(
    classifier: ViolenceClassifier,
    evidence: ViolenceEvidence | None = None,
) -> dict[str, object]:
    adapter_provenance = getattr(classifier, "provenance", {})
    provenance = dict(adapter_provenance) if isinstance(adapter_provenance, dict) else {}
    if evidence is not None:
        provenance.update({
            "model": evidence.model,
            "revision": evidence.revision,
            "label_mapping": [list(item) for item in evidence.label_mapping],
        })
    return provenance


def process_video(
    config: PipelineConfig,
    input_override: str | Path | None = None,
    *,
    detector: PersonDetector | None = None,
    tracker: Tracker | None = None,
    violence_classifier: ViolenceClassifier | None = None,
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
    m3_enabled = config.violence.enabled or violence_classifier is not None
    tracks_path = run_directory / "tracks.jsonl" if m2_enabled else None
    features_path = run_directory / "features.jsonl" if m2_enabled else None
    violence_path = run_directory / "violence.jsonl" if m3_enabled else None
    m4_enabled = m2_enabled or m3_enabled
    fusion_path = run_directory / "fusion.jsonl" if m4_enabled else None
    incidents_path = run_directory / "incidents.jsonl" if m4_enabled else None
    transitions_path = run_directory / "transitions.jsonl" if m4_enabled else None
    started_at = _utc_now()
    start_monotonic = time.perf_counter()
    processed_count = 0
    skipped_count = 0
    decode_seconds = 0.0
    write_seconds = 0.0
    detector_seconds = 0.0
    tracker_seconds = 0.0
    feature_seconds = 0.0
    violence_seconds = 0.0
    detector_calls = 0
    tracker_calls = 0
    feature_calls = 0
    violence_calls = 0
    detector_health: StageHealth | None = None
    tracker_health: StageHealth | None = None
    feature_health: StageHealth | None = None
    track_history: dict[int, list[TrackObservation]] = {}
    detection_interval = 1.0 / config.perception.cadence_fps
    last_detection_timestamp: float | None = None
    last_tracks: tuple[TrackObservation, ...] = ()
    violence_health: StageHealth | None = None
    violence_provenance: dict[str, object] | None = None
    latest_violence_evidence: ViolenceEvidence | None = None
    fusion_builder = FusionBuilder(config.fusion) if m4_enabled else None
    incident_engine = IncidentEngine(config.fusion) if m4_enabled else None
    fusion_seconds = 0.0
    fusion_calls = 0
    last_fusion_timestamp: float | None = None
    clip_buffer = RollingClipBuffer(config.violence.clip_duration_s, config.violence.sample_count) if m3_enabled else None
    violence_cadence = ViolenceCadence(config.violence.cadence_s) if m3_enabled else None
    if m3_enabled:
        violence_classifier = violence_classifier or VideoMAEViolenceClassifier(
            config.violence.model,
            config.violence.revision,
            device=config.violence.device,
            labels=config.violence.labels,
            license_name=config.violence.license,
            known_limitations=config.violence.known_limitations,
            checkpoint_sha256=config.violence.checkpoint_sha256,
        )
        violence_health = getattr(violence_classifier, "health", None)
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
    violence_output = violence_path.open("w") if violence_path else None
    fusion_output = fusion_path.open("w") if fusion_path else None
    incidents_output = incidents_path.open("w") if incidents_path else None
    transitions_output = transitions_path.open("w") if transitions_path else None
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
                    if m3_enabled:
                        clip_buffer.append(tracking_packet)
                        window = clip_buffer.complete_window()
                        if window is not None and violence_cadence.is_due(packet.timestamp_s):
                            inference_start = time.perf_counter()
                            classifier_health = None
                            try:
                                evidence = violence_classifier.infer(window)
                                if not isinstance(evidence, ViolenceEvidence):
                                    raise TypeError("violence classifier must return ViolenceEvidence")
                            except Exception as exc:
                                evidence = _violence_failure_evidence(
                                    window,
                                    violence_classifier,
                                    config,
                                    str(exc),
                                    (time.perf_counter() - inference_start) * 1000.0,
                                )
                                classifier_health = StageHealth(
                                    "violence", "degraded", model=evidence.model,
                                    device=config.violence.device, latency_ms=evidence.latency_ms,
                                    detail=evidence.detail,
                                )
                            measured_ms = (time.perf_counter() - inference_start) * 1000.0
                            violence_seconds += measured_ms / 1000.0
                            violence_calls += 1
                            latest_violence_evidence = evidence
                            violence_provenance = _violence_provenance(violence_classifier, evidence)
                            violence_health = classifier_health or getattr(violence_classifier, "health", StageHealth(
                                "violence", evidence.status, model=evidence.model,
                                device=config.violence.device, latency_ms=evidence.latency_ms,
                                detail=evidence.detail,
                            ))
                            if evidence.status != "available" and violence_health.status == "available":
                                violence_health = StageHealth(
                                    "violence", evidence.status, model=evidence.model,
                                    device=config.violence.device, latency_ms=evidence.latency_ms,
                                    detail=evidence.detail,
                                )
                            if violence_output:
                                violence_output.write(json.dumps({
                                    "source_id": packet.source_id,
                                    "frame_index": packet.frame_index,
                                    "timestamp_s": packet.timestamp_s,
                                    "health": asdict(violence_health),
                                    "evidence": asdict(evidence),
                                }, sort_keys=True) + "\n")
                    if m4_enabled:
                        fusion_start = time.perf_counter()
                        fusion_features = features or tuple(
                            CrowdFeatureRecord(
                                packet.source_id, roi.name, packet.timestamp_s, "unavailable",
                                detail="crowd feature branch is disabled",
                            ) for roi in config.crowd.rois
                        )
                        for feature in fusion_features:
                            point = fusion_builder.add(feature, latest_violence_evidence)
                            if point is None:
                                continue
                            fusion_calls += 1
                            last_fusion_timestamp = point.timestamp_s
                            if fusion_output:
                                fusion_output.write(json.dumps(asdict(point), sort_keys=True) + "\n")
                            incident, transitions = incident_engine.update(point)
                            if incidents_output and incident is not None:
                                incidents_output.write(json.dumps(asdict(incident), sort_keys=True) + "\n")
                            if transitions_output:
                                for transition in transitions:
                                    transitions_output.write(json.dumps(asdict(transition), sort_keys=True) + "\n")
                        fusion_seconds += time.perf_counter() - fusion_start
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
                            violence=latest_violence_evidence,
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

        if m4_enabled and last_fusion_timestamp is not None:
            for incident in incident_engine.flush(last_fusion_timestamp + config.fusion.quiet_period_s):
                if incidents_output:
                    incidents_output.write(json.dumps(asdict(incident), sort_keys=True) + "\n")
            if transitions_output:
                for transition in incident_engine.transitions:
                    if transition.timestamp_s > last_fusion_timestamp:
                        transitions_output.write(json.dumps(asdict(transition), sort_keys=True) + "\n")
      elapsed_seconds = time.perf_counter() - start_monotonic
    finally:
        if tracks_output:
            tracks_output.close()
        if features_output:
            features_output.close()
        if violence_output:
            violence_output.close()
        if fusion_output:
            fusion_output.close()
        if incidents_output:
            incidents_output.close()
        if transitions_output:
            transitions_output.close()
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
                **({"violence": violence_path.name} if m3_enabled else {}),
                **({
                    "fusion": fusion_path.name,
                    "incidents": incidents_path.name,
                    "transitions": transitions_path.name,
                } if m4_enabled else {}),
            },
            "stages": {
                "detector": asdict(detector_health) if detector_health else None,
                "tracker": asdict(tracker_health) if tracker_health else None,
                "crowd_features": asdict(feature_health) if feature_health else None,
                "violence": asdict(violence_health) if violence_health else None,
                "fusion": {
                    "status": "available",
                    "version": FUSION_VERSION,
                    "strategy": config.fusion.strategy,
                    "calls": fusion_calls,
                } if m4_enabled else None,
            },
            "provenance": ({
                "fusion_version": FUSION_VERSION,
                **({
                    "ultralytics": _package_version("ultralytics"),
                    "lap": _package_version("lap"),
                    "tracker_config": config.tracking.config,
                    "track_buffer": config.tracking.track_buffer,
                    "checkpoint_sha256": detector_health.checkpoint_sha256 if detector_health else None,
                } if m2_enabled else {}),
                **({
                    "violence_model": violence_provenance.get("model") if violence_provenance else None,
                    "violence_revision": violence_provenance.get("revision") if violence_provenance else None,
                    "violence_label_mapping": violence_provenance.get("label_mapping") if violence_provenance else [],
                    "violence_license": violence_provenance.get("license") if violence_provenance else None,
                    "violence_known_limitations": violence_provenance.get("known_limitations") if violence_provenance else None,
                    "violence_checkpoint_sha256": violence_provenance.get("checkpoint_sha256") if violence_provenance else None,
                } if m3_enabled else {}),
            } if m2_enabled or m3_enabled else None),
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
            "violence_seconds": violence_seconds,
            "detector_calls": detector_calls,
            "tracker_calls": tracker_calls,
            "crowd_feature_calls": feature_calls,
            "violence_calls": violence_calls,
            "fusion_seconds": fusion_seconds,
            "fusion_calls": fusion_calls,
            "incident_count": len({
                json.loads(line)["incident_id"] for line in incidents_path.read_text().splitlines()
            }) if incidents_path and incidents_path.exists() else 0,
            "transition_count": len(transitions_path.read_text().splitlines()) if transitions_path and transitions_path.exists() else 0,
            "stage_health": {
                "detector": asdict(detector_health) if detector_health else {"status": "disabled"},
                "tracker": asdict(tracker_health) if tracker_health else {"status": "disabled"},
                "crowd_features": asdict(feature_health) if feature_health else {"status": "disabled"},
                "violence": asdict(violence_health) if violence_health else {"status": "disabled"},
                "fusion": {
                    "status": "available",
                    "version": FUSION_VERSION,
                    "strategy": config.fusion.strategy,
                    "calls": fusion_calls,
                } if m4_enabled else {"status": "disabled"},
            },
            "total_seconds": elapsed_seconds,
            "effective_fps": (processed_count + skipped_count) / elapsed_seconds if elapsed_seconds else 0.0,
        },
    )
    evidence_manifests = capture_run_evidence(run_directory, config) if m4_enabled else ()
    if evidence_manifests:
        metadata = json.loads(metadata_path.read_text())
        metadata["artifacts"]["evidence_manifests"] = [
            f"{run_directory.name}/{manifest.incident_id}/manifest.json" for manifest in evidence_manifests
        ]
        metadata["evidence_count"] = len(evidence_manifests)
        write_json(metadata_path, metadata)
    return RunResult(
        run_id, digest, run_directory, video_path, frames_path, metadata_path, metrics_path,
        tracks_path, features_path, violence_path, fusion_path, incidents_path, transitions_path,
        tuple(config.m5.evidence_root / run_directory.name / manifest.incident_id / "manifest.json" for manifest in evidence_manifests),
    )
