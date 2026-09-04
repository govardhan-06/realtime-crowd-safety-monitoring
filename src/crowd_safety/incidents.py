from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .config import FusionConfig
from .types import Incident, IncidentState, IncidentTransition, FusionPoint, Severity


@dataclass
class _OpenIncident:
    incident_id: str
    source_id: str
    region_id: str
    started_at_s: float
    last_updated_at_s: float
    state: IncidentState
    severity: Severity
    peak_risk: float
    reason_codes: tuple[str, ...]
    quiet_since_s: float | None = None
    last_positive_s: float | None = None


def _severity(risk: float, config: FusionConfig) -> Severity:
    if risk >= config.severity_critical:
        return "critical"
    if risk >= config.severity_high:
        return "high"
    if risk >= config.severity_medium:
        return "medium"
    return "low"


def _id(source_id: str, region_id: str, started_at_s: float) -> str:
    value = f"{source_id}\0{region_id}\0{started_at_s:.9f}".encode()
    return f"incident-{hashlib.sha256(value).hexdigest()[:20]}"


class IncidentEngine:
    def __init__(self, config: FusionConfig):
        self.config = config
        self._open: dict[tuple[str, str], _OpenIncident] = {}
        self.transitions: list[IncidentTransition] = []

    def _snapshot(self, incident: _OpenIncident, timestamp_s: float, closed_at_s: float | None = None) -> Incident:
        return Incident(
            incident.incident_id, incident.source_id, incident.region_id, incident.state, incident.severity,
            incident.started_at_s, timestamp_s, closed_at_s, incident.peak_risk, incident.reason_codes,
        )

    def _transition(self, incident: _OpenIncident, timestamp_s: float, state: IncidentState, cause: str, reasons: tuple[str, ...]) -> IncidentTransition:
        transition = IncidentTransition(
            incident.incident_id, incident.source_id, incident.region_id, timestamp_s,
            incident.state, state, incident.severity, cause, reasons,
        )
        incident.state = state
        self.transitions.append(transition)
        return transition

    def update(self, point: FusionPoint) -> tuple[Incident | None, tuple[IncidentTransition, ...]]:
        key = (point.source_id, point.region_id)
        before = len(self.transitions)
        incident = self._open.get(key)
        positive = point.fused_risk >= self.config.candidate_threshold
        if incident is None:
            if not positive:
                return None, ()
            incident = _OpenIncident(
                _id(point.source_id, point.region_id, point.timestamp_s), point.source_id, point.region_id,
                point.timestamp_s, point.timestamp_s, "candidate", _severity(point.fused_risk, self.config),
                point.fused_risk, point.reason_codes, last_positive_s=point.timestamp_s,
            )
            self._open[key] = incident
            self._transition(incident, point.timestamp_s, "candidate", "candidate_started", point.reason_codes)
        else:
            incident.last_updated_at_s = point.timestamp_s
            incident.peak_risk = max(incident.peak_risk, point.fused_risk)
            incident.reason_codes = tuple(dict.fromkeys(incident.reason_codes + point.reason_codes))
            if positive:
                incident.last_positive_s = point.timestamp_s
                if incident.state == "resolving" and point.fused_risk < self.config.active_threshold + self.config.hysteresis:
                    incident.quiet_since_s = incident.quiet_since_s or point.timestamp_s
                else:
                    incident.quiet_since_s = None
            elif incident.last_positive_s is not None and point.timestamp_s - incident.last_positive_s > self.config.decay_s:
                self._transition(incident, point.timestamp_s, "closed", "decay", incident.reason_codes)
                self._open.pop(key)
                return self._snapshot(incident, point.timestamp_s, point.timestamp_s), tuple(self.transitions[before:])

        previous_severity = incident.severity
        incident.severity = max((incident.severity, _severity(point.fused_risk, self.config)), key=("low", "medium", "high", "critical").index)
        if incident.severity != previous_severity:
            self._transition(incident, point.timestamp_s, incident.state, "severity_increased", incident.reason_codes)

        if incident.state == "candidate" and positive and point.timestamp_s - incident.started_at_s >= self.config.persistence_s and point.fused_risk >= self.config.active_threshold:
            self._transition(incident, point.timestamp_s, "active", "persistence_reached", incident.reason_codes)
        elif incident.state == "active" and point.fused_risk >= self.config.escalating_threshold:
            self._transition(incident, point.timestamp_s, "escalating", "risk_increased", incident.reason_codes)
        elif incident.state == "escalating" and point.fused_risk >= self.config.critical_threshold:
            self._transition(incident, point.timestamp_s, "critical", "critical_threshold", incident.reason_codes)
        elif incident.state in {"active", "escalating", "critical"} and point.fused_risk < self.config.active_threshold - self.config.hysteresis:
            incident.quiet_since_s = point.timestamp_s
            self._transition(incident, point.timestamp_s, "resolving", "risk_decayed", incident.reason_codes)
        elif incident.state == "resolving" and positive and point.fused_risk >= self.config.active_threshold + self.config.hysteresis:
            self._transition(incident, point.timestamp_s, "active", "risk_recovered", incident.reason_codes)
        elif incident.state == "resolving" and incident.quiet_since_s is not None and point.timestamp_s - incident.quiet_since_s >= self.config.quiet_period_s:
            self._transition(incident, point.timestamp_s, "closed", "quiet_period_elapsed", incident.reason_codes)
            self._open.pop(key)
            return self._snapshot(incident, point.timestamp_s, point.timestamp_s), tuple(self.transitions[before:])
        return self._snapshot(incident, point.timestamp_s), tuple(self.transitions[before:])

    def flush(self, timestamp_s: float) -> tuple[Incident, ...]:
        closed: list[Incident] = []
        for key, incident in list(self._open.items()):
            if incident.state == "candidate" and incident.last_positive_s is not None and timestamp_s - incident.last_positive_s >= self.config.decay_s:
                self._transition(incident, timestamp_s, "closed", "decay", incident.reason_codes)
                self._open.pop(key)
                closed.append(self._snapshot(incident, timestamp_s, timestamp_s))
            elif incident.quiet_since_s is not None and timestamp_s - incident.quiet_since_s >= self.config.quiet_period_s:
                self._transition(incident, timestamp_s, "closed", "quiet_period_elapsed", incident.reason_codes)
                self._open.pop(key)
                closed.append(self._snapshot(incident, timestamp_s, timestamp_s))
        return tuple(closed)


@dataclass(frozen=True)
class IncidentReplay:
    incidents: tuple[Incident, ...]
    transitions: tuple[IncidentTransition, ...]


def replay_incidents(points: list[FusionPoint] | tuple[FusionPoint, ...], config: FusionConfig) -> IncidentReplay:
    engine = IncidentEngine(config)
    incidents: list[Incident] = []
    for point in points:
        incident, _ = engine.update(point)
        if incident is not None:
            incidents.append(incident)
    if points:
        incidents.extend(engine.flush(points[-1].timestamp_s + config.quiet_period_s))
    return IncidentReplay(tuple(incidents), tuple(engine.transitions))
