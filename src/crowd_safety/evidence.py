from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from typing import Any

import cv2

from .config import PipelineConfig
from .types import EvidenceManifest, EvidenceReference


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_clip(
    video_path: Path,
    timestamps: list[float],
    output_path: Path,
    lower: float,
    upper: float,
    fps: float,
) -> EvidenceReference:
    selected = [index for index, timestamp in enumerate(timestamps) if lower <= timestamp <= upper]
    if not selected:
        return EvidenceReference(output_path.stem, "", lower, upper, "failed", "no annotated frames in requested bound")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return EvidenceReference(output_path.stem, "", lower, upper, "failed", "annotated video could not be opened")
    writer = None
    written_times: list[float] = []
    try:
        selected_set = set(selected)
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected_set:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        return EvidenceReference(output_path.stem, "", lower, upper, "failed", "evidence clip could not be opened")
                writer.write(frame)
                written_times.append(timestamps[frame_index])
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not written_times:
        return EvidenceReference(output_path.stem, "", lower, upper, "failed", "annotated video ended before requested bound")
    return EvidenceReference(
        output_path.stem, output_path.name, written_times[0], written_times[-1], "available"
    )


def _prune_evidence(root: Path, retention_s: float) -> None:
    cutoff = time.time() - retention_s
    if not root.exists():
        return
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink() or child.stat().st_mtime >= cutoff:
            continue
        manifests = list(child.glob("*/manifest.json"))
        valid = bool(manifests)
        for path in manifests:
            try:
                manifest = json.loads(path.read_text())
            except (OSError, ValueError):
                valid = False
                break
            if manifest.get("run_id") != child.name or not manifest.get("incident_id"):
                valid = False
                break
        if not valid:
            continue
        shutil.rmtree(child)


def capture_run_evidence(run_directory: str | Path, config: PipelineConfig) -> tuple[EvidenceManifest, ...]:
    run_directory = Path(run_directory).resolve()
    incidents = _jsonl(run_directory / "incidents.jsonl")
    if not incidents:
        return ()
    latest: dict[str, dict[str, Any]] = {}
    for incident in incidents:
        latest[incident["incident_id"]] = incident
    timeline = _jsonl(run_directory / "fusion.jsonl")
    metadata = json.loads((run_directory / "metadata.json").read_text()) if (run_directory / "metadata.json").exists() else {}
    stage_health = metadata.get("stages", {})
    frame_rows = [row for row in _jsonl(run_directory / "frames.jsonl") if row.get("processed")]
    timestamps = [float(row["timestamp_s"]) for row in frame_rows]
    video_path = run_directory / "annotated.mp4"
    root = config.m5.evidence_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _prune_evidence(root, config.m5.retention_s)
    manifests: list[EvidenceManifest] = []
    for incident_id, incident in latest.items():
        source_id = str(incident["source_id"])
        region_id = str(incident["region_id"])
        start_s = float(incident["started_at_s"])
        end_s = float(incident["last_updated_at_s"])
        output_directory = root / run_directory.name / incident_id
        output_directory.mkdir(parents=True, exist_ok=True)
        relative_prefix = Path(run_directory.name) / incident_id
        references: list[EvidenceReference] = []
        if video_path.exists() and timestamps:
            target_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - start_s))
            target_frame = None
            capture = cv2.VideoCapture(str(video_path))
            for frame_index in range(target_index + 1):
                ok, target_frame = capture.read()
                if not ok:
                    target_frame = None
                    break
            capture.release()
            snapshot_path = output_directory / "snapshot.jpg"
            if target_frame is not None and cv2.imwrite(str(snapshot_path), target_frame):
                references.append(EvidenceReference(
                    "snapshot", (relative_prefix / snapshot_path.name).as_posix(),
                    timestamps[target_index], timestamps[target_index], "available",
                ))
            else:
                references.append(EvidenceReference("snapshot", "", start_s, start_s, "failed", "snapshot capture failed"))
            pre_path = output_directory / "pre_event.mp4"
            pre = _write_clip(video_path, timestamps, pre_path, max(0.0, start_s - config.m5.pre_event_s), start_s, config.target_fps)
            references.append(EvidenceReference(
                "pre_event_clip", (relative_prefix / pre_path.name).as_posix() if pre.status == "available" else "",
                pre.start_s, pre.end_s, pre.status, pre.detail,
            ))
            post_path = output_directory / "post_event.mp4"
            post = _write_clip(video_path, timestamps, post_path, end_s, end_s + config.m5.post_event_s, config.target_fps)
            references.append(EvidenceReference(
                "post_event_clip", (relative_prefix / post_path.name).as_posix() if post.status == "available" else "",
                post.start_s, post.end_s, post.status, post.detail,
            ))
        else:
            detail = "annotated video or processed frame timestamps are unavailable"
            references = [
                EvidenceReference("snapshot", "", start_s, start_s, "failed", detail),
                EvidenceReference("pre_event_clip", "", max(0.0, start_s - config.m5.pre_event_s), start_s, "failed", detail),
                EvidenceReference("post_event_clip", "", end_s, end_s + config.m5.post_event_s, "failed", detail),
            ]
        incident_timeline = tuple(
            point for point in timeline
            if point.get("source_id") == source_id and point.get("region_id") == region_id
            and start_s <= float(point.get("timestamp_s", -1)) <= end_s
        )
        manifest = EvidenceManifest(
            run_directory.name, source_id, incident_id, start_s, end_s,
            config.m5.pre_event_s, config.m5.post_event_s,
            tuple(incident.get("reason_codes", ())), incident_timeline,
            stage_health, tuple(references),
        )
        (output_directory / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        manifests.append(manifest)
    return tuple(manifests)
