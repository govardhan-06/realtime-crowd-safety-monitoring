from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .types import IncidentExplanation, OperatorAction


class PersistenceConflict(RuntimeError):
    """Raised when an imported run ID has different authoritative M4 data."""


class Persistence(Protocol):
    def list_sources(self) -> list[dict[str, Any]]: ...
    def list_runs(self) -> list[dict[str, Any]]: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def list_incidents(self, source_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]: ...
    def get_incident(self, incident_id: str) -> dict[str, Any] | None: ...
    def record_action(self, incident_id: str, action: str, actor: str, timestamp: str, note: str | None = None) -> dict[str, Any]: ...
    def save_explanation(self, explanation: IncidentExplanation) -> None: ...


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_no_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in str(key).lower() for token in ("password", "secret", "token", "api_key", "database_url")):
                raise ValueError("persisted metadata must not contain secrets")
            _assert_no_secrets(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_secrets(nested)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _deterministic_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": bundle["run"],
        "incidents": bundle["incidents"],
        "transitions": bundle["transitions"],
        "fusion": bundle["fusion"],
    }


def _record_id(run_id: str, incident_id: str) -> str:
    return f"{run_id}--{incident_id}"


def _load_bundle(run_directory: Path, evidence_root: Path | None = None) -> dict[str, Any]:
    metadata = _read_json(run_directory / "metadata.json")
    run_id = str(metadata.get("run_id") or run_directory.name)
    incidents = _read_jsonl(run_directory / "incidents.jsonl")
    latest_incidents = {}
    for incident in incidents:
        latest_incidents[incident["incident_id"]] = incident
    transitions = _read_jsonl(run_directory / "transitions.jsonl")
    fusion = _read_jsonl(run_directory / "fusion.jsonl")
    root = evidence_root or (run_directory / "evidence")
    manifests = []
    for incident_id in latest_incidents:
        manifest_path = root / run_id / incident_id / "manifest.json"
        if not manifest_path.exists():
            manifest_path = root / incident_id / "manifest.json"
        if manifest_path.exists():
            manifests.append(_read_json(manifest_path))
    bundle = {
        "run": {
            "run_id": run_id,
            "source_id": metadata.get("input", {}).get("source_id", "unknown"),
            "config_hash": metadata.get("config_hash"),
            "started_at": metadata.get("started_at"),
            "ended_at": metadata.get("ended_at"),
            "input": metadata.get("input", {}),
            "artifacts": metadata.get("artifacts", {}),
            "stages": metadata.get("stages", {}),
            "provenance": metadata.get("provenance", {}),
            "metrics": _read_json(run_directory / "metrics.json") if (run_directory / "metrics.json").exists() else {},
        },
        "incidents": list(latest_incidents.values()),
        "transitions": transitions,
        "fusion": fusion,
        "evidence": manifests,
    }
    _assert_no_secrets(bundle)
    return bundle


def import_run(run_directory: str | Path, store: "MemoryPersistence | PostgresPersistence", evidence_root: str | Path | None = None) -> bool:
    bundle = _load_bundle(Path(run_directory).resolve(), Path(evidence_root).resolve() if evidence_root else None)
    return store.import_bundle(bundle)


def configured_store(config: Any, *, allow_ephemeral: bool = False) -> "MemoryPersistence | PostgresPersistence":
    dsn = os.environ.get(config.m5.database_url_env)
    if dsn:
        store = PostgresPersistence(dsn)
        store.ensure_schema()
        return store
    if allow_ephemeral:
        return MemoryPersistence()
    raise RuntimeError(f"{config.m5.database_url_env} is not set; pass --ephemeral only for a local demo")


class MemoryPersistence:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.incidents: dict[str, dict[str, Any]] = {}
        self.transitions: dict[str, list[dict[str, Any]]] = {}
        self.evidence: dict[str, list[dict[str, Any]]] = {}
        self.timelines: dict[str, list[dict[str, Any]]] = {}
        self.actions: dict[str, list[dict[str, Any]]] = {}
        self.explanations: dict[str, dict[str, Any]] = {}
        self.incident_aliases: dict[str, list[str]] = {}

    def import_bundle(self, bundle: dict[str, Any]) -> bool:
        run = bundle["run"]
        run_id = run["run_id"]
        deterministic = _deterministic_payload(bundle)
        if run_id in self.runs:
            if self.runs[run_id]["deterministic_hash"] != _fingerprint(deterministic):
                raise PersistenceConflict(f"run {run_id} conflicts with existing authoritative records")
            return False
        for incident in bundle["incidents"]:
            incident_id = incident["incident_id"]
            record_id = _record_id(run_id, incident_id)
            if record_id in self.incidents and self.incidents[record_id] != incident:
                raise PersistenceConflict(f"incident {incident_id} conflicts with existing run record")
        self.runs[run_id] = {**run, "deterministic_hash": _fingerprint(deterministic)}
        for incident in bundle["incidents"]:
            incident_id = incident["incident_id"]
            record_id = _record_id(run_id, incident_id)
            self.incidents[record_id] = incident
            self.incident_aliases.setdefault(incident_id, []).append(record_id)
            self.transitions[record_id] = [item for item in bundle["transitions"] if item.get("incident_id") == incident_id]
            self.evidence[record_id] = [item for item in bundle["evidence"] if item.get("incident_id") == incident_id]
            self.timelines[record_id] = [
                item for item in bundle["fusion"]
                if item.get("source_id") == incident.get("source_id") and item.get("region_id") == incident.get("region_id")
                and incident["started_at_s"] <= item.get("timestamp_s", -1) <= incident["last_updated_at_s"]
            ]
        return True

    def list_sources(self) -> list[dict[str, Any]]:
        sources = {}
        for run in self.runs.values():
            sources[run["source_id"]] = {"source_id": run["source_id"], "latest_run_id": run["run_id"]}
        return list(sources.values())

    def list_runs(self) -> list[dict[str, Any]]:
        return [{key: value for key, value in run.items() if key != "deterministic_hash"} for run in self.runs.values()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.runs.get(run_id)
        return {key: value for key, value in run.items() if key != "deterministic_hash"} if run else None

    def list_incidents(self, source_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
        values = [self.get_incident(incident_id) for incident_id in self.incidents]
        return [item for item in values if item and (source_id is None or item["incident"]["source_id"] == source_id) and (state is None or item["incident"]["state"] == state)]

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        record_id = self._resolve_record_id(incident_id)
        incident = self.incidents.get(record_id) if record_id else None
        if incident is None:
            return None
        return {
            "record_id": record_id,
            "incident": incident,
            "transitions": self.transitions.get(record_id, []),
            "timeline": self.timelines.get(record_id, []),
            "evidence": self.evidence.get(record_id, []),
            "explanation": self.explanations.get(record_id, {"status": "disabled", "provider": "disabled", "text": ""}),
            "actions": self.actions.get(record_id, []),
        }

    def _resolve_record_id(self, incident_id: str) -> str | None:
        if incident_id in self.incidents:
            return incident_id
        aliases = self.incident_aliases.get(incident_id, [])
        return aliases[0] if len(aliases) == 1 else None

    def record_action(self, incident_id: str, action: str, actor: str, timestamp: str, note: str | None = None) -> dict[str, Any]:
        OperatorAction(incident_id, action, actor, timestamp, note)
        record_id = self._resolve_record_id(incident_id)
        if record_id is None:
            raise KeyError(incident_id)
        record = {"action_id": str(uuid4()), "incident_id": incident_id, "action": action, "actor": actor, "timestamp": timestamp, "note": note}
        self.actions.setdefault(record_id, []).append(record)
        return record

    def save_explanation(self, explanation: IncidentExplanation) -> None:
        record_id = self._resolve_record_id(explanation.incident_id)
        if record_id is None:
            raise KeyError(explanation.incident_id)
        self.explanations[record_id] = {
            "incident_id": explanation.incident_id, "status": explanation.status,
            "provider": explanation.provider, "model": explanation.model,
            "text": explanation.text, "detail": explanation.detail,
        }


class PostgresPersistence:
    def __init__(self, dsn: str, connection: Any | None = None) -> None:
        if connection is None:
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError("PostgreSQL persistence requires the psycopg package") from exc
            connection = psycopg.connect(dsn)
        self.connection = connection

    def ensure_schema(self) -> None:
        sql = files("crowd_safety.migrations").joinpath("001_m5.sql").read_text()
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()

    def import_bundle(self, bundle: dict[str, Any]) -> bool:
        run = bundle["run"]
        run_hash = _fingerprint(_deterministic_payload(bundle))
        payload = json.dumps(run)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT deterministic_hash FROM m5_runs WHERE run_id = %s FOR UPDATE", (run["run_id"],))
                row = cursor.fetchone()
                if row:
                    self.connection.rollback()
                    if row[0] != run_hash:
                        raise PersistenceConflict(f"run {run['run_id']} conflicts with existing authoritative records")
                    return False
                cursor.execute("INSERT INTO m5_sources (source_id, payload) VALUES (%s, %s::jsonb) ON CONFLICT DO NOTHING", (run["source_id"], json.dumps({"source_id": run["source_id"]})))
                cursor.execute("INSERT INTO m5_runs (run_id, source_id, config_hash, deterministic_hash, payload) VALUES (%s, %s, %s, %s, %s::jsonb)", (run["run_id"], run["source_id"], run["config_hash"], run_hash, payload))
                for incident in bundle["incidents"]:
                    cursor.execute("INSERT INTO m5_incidents (incident_id, run_id, payload) VALUES (%s, %s, %s::jsonb)", (_record_id(run["run_id"], incident["incident_id"]), run["run_id"], json.dumps(incident)))
                for sequence, transition in enumerate(bundle["transitions"]):
                    cursor.execute("INSERT INTO m5_transitions (run_id, incident_id, sequence, payload) VALUES (%s, %s, %s, %s::jsonb)", (run["run_id"], _record_id(run["run_id"], transition["incident_id"]), sequence, json.dumps(transition)))
                for incident in bundle["incidents"]:
                    points = [point for point in bundle["fusion"] if point.get("source_id") == incident.get("source_id") and point.get("region_id") == incident.get("region_id") and incident["started_at_s"] <= point.get("timestamp_s", -1) <= incident["last_updated_at_s"]]
                    for sequence, point in enumerate(points):
                        cursor.execute("INSERT INTO m5_timelines (incident_id, sequence, payload) VALUES (%s, %s, %s::jsonb)", (_record_id(run["run_id"], incident["incident_id"]), sequence, json.dumps(point)))
                for manifest in bundle["evidence"]:
                    cursor.execute("INSERT INTO m5_evidence (incident_id, payload) VALUES (%s, %s::jsonb)", (_record_id(run["run_id"], manifest["incident_id"]), json.dumps(manifest)))
            self.connection.commit()
            return True
        except PersistenceConflict:
            raise
        except Exception:
            self.connection.rollback()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT deterministic_hash FROM m5_runs WHERE run_id = %s", (run["run_id"],))
                row = cursor.fetchone()
            if row and row[0] == run_hash:
                return False
            if row:
                raise PersistenceConflict(f"run {run['run_id']} conflicts with existing authoritative records")
            raise

    def list_sources(self) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT source_id, payload FROM m5_sources ORDER BY source_id")
            return [{**row[1], "source_id": row[0]} for row in cursor.fetchall()]

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM m5_runs ORDER BY run_id DESC")
            return [row[0] for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM m5_runs WHERE run_id = %s", (run_id,))
            row = cursor.fetchone()
        return row[0] if row else None

    def list_incidents(self, source_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
        conditions, params = [], []
        if source_id:
            conditions.append("payload->>'source_id' = %s")
            params.append(source_id)
        if state:
            conditions.append("payload->>'state' = %s")
            params.append(state)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT incident_id FROM m5_incidents{where} ORDER BY payload->>'last_updated_at_s' DESC", params)
            ids = [row[0] for row in cursor.fetchall()]
        return [self.get_incident(incident_id) for incident_id in ids]

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM m5_incidents WHERE incident_id = %s", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return None
            incident = row[0]
            cursor.execute("SELECT payload FROM m5_transitions WHERE incident_id = %s ORDER BY sequence", (incident_id,))
            transitions = [item[0] for item in cursor.fetchall()]
            cursor.execute("SELECT payload FROM m5_timelines WHERE incident_id = %s ORDER BY sequence", (incident_id,))
            timeline = [item[0] for item in cursor.fetchall()]
            cursor.execute("SELECT payload FROM m5_evidence WHERE incident_id = %s ORDER BY evidence_id", (incident_id,))
            evidence = [item[0] for item in cursor.fetchall()]
            cursor.execute("SELECT action_id, action, actor, timestamp, note FROM m5_actions WHERE incident_id = %s ORDER BY sequence", (incident_id,))
            actions = [{"action_id": row[0], "action": row[1], "actor": row[2], "timestamp": row[3], "note": row[4]} for row in cursor.fetchall()]
            cursor.execute("SELECT payload FROM m5_explanations WHERE incident_id = %s", (incident_id,))
            explanation_row = cursor.fetchone()
        return {"record_id": incident_id, "incident": incident, "transitions": transitions, "timeline": timeline, "evidence": evidence, "explanation": explanation_row[0] if explanation_row else {"status": "disabled", "provider": "disabled", "text": ""}, "actions": actions}

    def record_action(self, incident_id: str, action: str, actor: str, timestamp: str, note: str | None = None) -> dict[str, Any]:
        record = OperatorAction(incident_id, action, actor, timestamp, note)
        if self.get_incident(incident_id) is None:
            raise KeyError(incident_id)
        action_id = str(uuid4())
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO m5_actions (action_id, incident_id, action, actor, timestamp, note) VALUES (%s, %s, %s, %s, %s, %s)", (action_id, record.incident_id, record.action, record.actor, record.timestamp, record.note))
        self.connection.commit()
        return {"action_id": action_id, "incident_id": incident_id, "action": action, "actor": actor, "timestamp": timestamp, "note": note}

    def save_explanation(self, explanation: IncidentExplanation) -> None:
        if self.get_incident(explanation.incident_id) is None:
            raise KeyError(explanation.incident_id)
        payload = {"incident_id": explanation.incident_id, "status": explanation.status, "provider": explanation.provider, "model": explanation.model, "text": explanation.text, "detail": explanation.detail}
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO m5_explanations (incident_id, payload) VALUES (%s, %s::jsonb) ON CONFLICT (incident_id) DO UPDATE SET payload = EXCLUDED.payload", (explanation.incident_id, json.dumps(payload)))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
