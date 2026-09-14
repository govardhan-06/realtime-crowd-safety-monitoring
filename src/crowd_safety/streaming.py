from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Lock
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import PipelineConfig
from .persistence import Persistence, import_run
from .runner import process_video


class EventBuffer:
    """Small replay buffer; old frame events are discarded first when busy."""

    def __init__(self, limit: int = 200) -> None:
        self.limit = limit
        self._events: deque[tuple[int, dict[str, Any]]] = deque()
        self._next = 0
        self._lock = Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append((self._next, event))
            self._next += 1
            while len(self._events) > self.limit:
                frame_index = next((i for i, (_, item) in enumerate(self._events) if item.get("type") == "frame"), None)
                if frame_index is None:
                    self._events.popleft()
                else:
                    del self._events[frame_index]

    def since(self, cursor: int) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            if not self._events:
                return self._next, []
            cursor = max(cursor, self._events[0][0])
            items = [event for sequence, event in self._events if sequence >= cursor]
            return self._next, items


class RunManager:
    def __init__(self, config: PipelineConfig, store: Persistence) -> None:
        self.config = config
        self.store = store
        self._jobs: dict[str, dict[str, Any]] = {}
        self._buffers: dict[str, EventBuffer] = {}
        self._lock = Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crowd-safety-run")

    def submit(self, content: bytes, filename: str, content_type: str | None) -> str:
        run_id = f"upload-{uuid4().hex}"
        upload_dir = self.config.output_directory / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        input_path = upload_dir / f"{run_id}-{Path(filename).name}"
        input_path.write_bytes(content)
        buffer = EventBuffer()
        with self._lock:
            self._buffers[run_id] = buffer
            self._jobs[run_id] = {
                "run_id": run_id,
                "state": "queued",
                "input": {"filename": Path(filename).name, "content_type": content_type or ""},
                "artifacts": {},
            }
        self._worker.submit(self._process, run_id, input_path, buffer)
        return run_id

    def _update(self, run_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[run_id].update(values)

    def _process(self, run_id: str, input_path: Path, buffer: EventBuffer) -> None:
        self._update(run_id, state="processing")
        completion: dict[str, Any] | None = None

        def publish(event: dict[str, Any]) -> None:
            nonlocal completion
            if event.get("type") == "complete":
                completion = event
            else:
                buffer.publish(event)

        try:
            result = process_video(self.config, input_path, event_sink=publish, run_id=run_id)
            import_run(result.run_directory, self.store, self.config.m5.evidence_root)
            self._update(run_id, state="completed", artifacts={"annotated_video": str(result.video_path), "run_directory": str(result.run_directory)})
            buffer.publish(completion or {"type": "complete", "run_id": run_id, "artifact_url": f"/runs/{run_id}/artifact"})
        except Exception as exc:
            self._update(run_id, state="failed", error=str(exc))
            buffer.publish({"type": "error", "message": str(exc)})

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(run_id)
        if job:
            return dict(job)
        return self.store.get_run(run_id)

    def events(self, run_id: str, cursor: int) -> tuple[int, list[dict[str, Any]]]:
        buffer = self._buffers.get(run_id)
        return buffer.since(cursor) if buffer else (cursor, [])

    def known(self, run_id: str) -> bool:
        return run_id in self._jobs
