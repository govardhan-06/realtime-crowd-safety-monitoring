from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import asyncio

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .explanations import DisabledExplainer, IncidentExplainer, explain_incident
from .persistence import Persistence
from .streaming import RunManager
from .config import PipelineConfig


IncidentStateParam = Literal["candidate", "active", "escalating", "critical", "resolving", "closed"]


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=1, max_length=120)
    timestamp: datetime
    note: str | None = Field(default=None, max_length=1000)


def _public_run(run: dict) -> dict:
    result = dict(run)
    result["input"] = {key: value for key, value in result.get("input", {}).items() if key != "path"}
    return result


def _public_incident(record: dict) -> dict:
    incident = record["incident"]
    return {
        "record_id": record.get("record_id", incident["incident_id"]),
        "incident": incident,
        "deterministic": {
            "reason_codes": incident.get("reason_codes", []),
            "transitions": record.get("transitions", []),
            "timeline": record.get("timeline", []),
            "evidence": record.get("evidence", []),
        },
        "explanation": record.get("explanation", {"status": "disabled", "provider": "disabled", "text": ""}),
        "actions": record.get("actions", []),
    }


def create_app(store: Persistence, evidence_root: str | Path | None = None, explainer: IncidentExplainer | None = None, processing_config: PipelineConfig | None = None) -> FastAPI:
    app = FastAPI(title="Crowd Safety Human Review API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-filename"],
    )
    explainer = explainer or DisabledExplainer()
    run_manager = RunManager(processing_config, store) if processing_config else None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "explanation_provider": getattr(explainer, "provider", "unknown")}

    @app.get("/sources")
    def sources() -> list[dict]:
        return store.list_sources()

    @app.get("/runs")
    def runs() -> list[dict]:
        imported = {run["run_id"]: _public_run(run) for run in store.list_runs()}
        if run_manager:
            imported.update({run_id: _public_run(run_manager.get(run_id) or {}) for run_id in run_manager._jobs})
        return list(imported.values())

    @app.post("/runs", status_code=201)
    async def create_run(request: Request) -> dict:
        if run_manager is None:
            raise HTTPException(503, "processing is not configured")
        filename = request.headers.get("x-filename", "upload.mp4")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise HTTPException(415, "unsupported video filename")
        content = await request.body()
        if not content:
            raise HTTPException(422, "video upload is required")
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(413, "video upload exceeds 100 MB")
        return {"run_id": run_manager.submit(content, filename, request.headers.get("content-type"))}

    @app.get("/runs/{run_id}/artifact")
    def run_artifact(run_id: str) -> FileResponse:
        run = run_manager.get(run_id) if run_manager else None
        path = Path(run.get("artifacts", {}).get("annotated_video", "")) if run else None
        if path is None or not path.is_file():
            raise HTTPException(404, "artifact not found")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        run = run_manager.get(run_id) if run_manager else store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return _public_run(run)

    @app.websocket("/runs/{run_id}/stream")
    async def run_stream(websocket: WebSocket, run_id: str) -> None:
        if run_manager is None or not run_manager.known(run_id):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        cursor = 0
        while True:
            cursor, events = run_manager.events(run_id, cursor)
            for event in events:
                await websocket.send_json(event)
                if event["type"] in {"complete", "error"}:
                    return
            state = run_manager.get(run_id).get("state")
            if state in {"completed", "failed"}:
                return
            await asyncio.sleep(0.03)

    @app.get("/incidents")
    def incidents(
        source_id: str | None = None,
        state: IncidentStateParam | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> list[dict]:
        return [
            {
                "incident": record["incident"],
                "record_id": record.get("record_id", record["incident"]["incident_id"]),
                "reason_codes": record["incident"].get("reason_codes", []),
                "evidence_available": any(item.get("status") == "available" for manifest in record.get("evidence", []) for item in manifest.get("artifacts", [])),
                "explanation_status": record.get("explanation", {}).get("status", "disabled"),
            }
            for record in store.list_incidents(source_id, state)
        ][:limit]

    @app.get("/incidents/{incident_id}")
    def incident_detail(incident_id: str) -> dict:
        record = store.get_incident(incident_id)
        if record is None:
            raise HTTPException(404, "incident not found")
        return _public_incident(record)

    @app.get("/incidents/{incident_id}/timeline")
    def incident_timeline(incident_id: str) -> list[dict]:
        record = store.get_incident(incident_id)
        if record is None:
            raise HTTPException(404, "incident not found")
        return record.get("timeline", [])

    @app.get("/incidents/{incident_id}/evidence")
    def incident_evidence(incident_id: str) -> list[dict]:
        record = store.get_incident(incident_id)
        if record is None:
            raise HTTPException(404, "incident not found")
        return record.get("evidence", [])

    @app.get("/incidents/{incident_id}/evidence/{kind}")
    def incident_evidence_file(incident_id: str, kind: Literal["snapshot", "pre_event_clip", "post_event_clip"]) -> FileResponse:
        record = store.get_incident(incident_id)
        root = Path(evidence_root).resolve() if evidence_root else None
        if record is None or root is None:
            raise HTTPException(404, "evidence not found")
        for manifest in record.get("evidence", []):
            for artifact in manifest.get("artifacts", []):
                if artifact.get("kind") != kind or artifact.get("status") != "available":
                    continue
                relative_path = artifact.get("relative_path", "")
                if not isinstance(relative_path, str) or not relative_path:
                    continue
                candidate = (root / relative_path).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    raise HTTPException(404, "evidence not found")
                if not candidate.is_file():
                    raise HTTPException(404, "evidence not found")
                media_type = "image/jpeg" if kind == "snapshot" else "video/mp4"
                return FileResponse(candidate, media_type=media_type, filename=candidate.name)
        raise HTTPException(404, "evidence not found")

    @app.get("/incidents/{incident_id}/explanation")
    def incident_explanation(incident_id: str) -> dict:
        record = store.get_incident(incident_id)
        if record is None:
            raise HTTPException(404, "incident not found")
        return record.get("explanation", {"status": "disabled", "provider": "disabled", "text": ""})

    @app.post("/incidents/{incident_id}/explain")
    def generate_explanation(incident_id: str) -> dict:
        try:
            return explain_incident(store, incident_id, explainer).__dict__
        except KeyError:
            raise HTTPException(404, "incident not found")

    def record_action(incident_id: str, action: str, payload: ActionRequest) -> dict:
        try:
            return store.record_action(incident_id, action, payload.actor.strip(), payload.timestamp.isoformat(), payload.note)
        except KeyError:
            raise HTTPException(404, "incident not found")
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/incidents/{incident_id}/acknowledge")
    def acknowledge(incident_id: str, payload: ActionRequest) -> dict:
        return record_action(incident_id, "acknowledge", payload)

    @app.post("/incidents/{incident_id}/dismiss")
    def dismiss(incident_id: str, payload: ActionRequest) -> dict:
        return record_action(incident_id, "dismiss", payload)

    @app.post("/incidents/{incident_id}/escalate")
    def escalate(incident_id: str, payload: ActionRequest) -> dict:
        return record_action(incident_id, "escalate", payload)
    return app
