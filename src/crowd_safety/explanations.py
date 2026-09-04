from __future__ import annotations

from typing import Any, Protocol

from .persistence import Persistence
from .types import IncidentExplanation


class IncidentExplainer(Protocol):
    provider: str
    model: str

    def explain(self, incident: dict[str, Any], evidence: list[dict[str, Any]]) -> IncidentExplanation: ...


class DisabledExplainer:
    provider = "disabled"
    model = ""

    def explain(self, incident: dict[str, Any], evidence: list[dict[str, Any]]) -> IncidentExplanation:
        return IncidentExplanation(incident["incident_id"], "disabled", self.provider, self.model, detail="VLM explanation is disabled")


class UnavailableExplainer:
    provider = "unavailable"
    model = ""

    def __init__(self, detail: str) -> None:
        self.detail = detail

    def explain(self, incident: dict[str, Any], evidence: list[dict[str, Any]]) -> IncidentExplanation:
        return IncidentExplanation(incident["incident_id"], "unavailable", self.provider, self.model, detail=self.detail)


class FakeExplainer:
    provider = "fake"
    model = "fake-model"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def explain(self, incident: dict[str, Any], evidence: list[dict[str, Any]]) -> IncidentExplanation:
        if self.error:
            raise self.error
        return IncidentExplanation(incident["incident_id"], "generated", self.provider, self.model, self.text)


def explain_incident(store: Persistence, incident_id: str, explainer: IncidentExplainer) -> IncidentExplanation:
    record = store.get_incident(incident_id)
    if record is None:
        raise KeyError(incident_id)
    try:
        explanation = explainer.explain(record["incident"], record["evidence"])
    except TimeoutError as exc:
        explanation = IncidentExplanation(incident_id, "unavailable", getattr(explainer, "provider", "unknown"), getattr(explainer, "model", ""), detail=str(exc))
    except Exception as exc:
        explanation = IncidentExplanation(incident_id, "failed", getattr(explainer, "provider", "unknown"), getattr(explainer, "model", ""), detail=str(exc))
    if explanation.incident_id != incident_id:
        explanation = IncidentExplanation(
            incident_id, explanation.status, explanation.provider, explanation.model,
            explanation.text, explanation.detail,
        )
    store.save_explanation(explanation)
    return explanation


def configured_explainer(enabled: bool, provider: str, model: str) -> IncidentExplainer:
    if not enabled or provider == "disabled":
        return DisabledExplainer()
    if provider == "fake":
        return FakeExplainer("")
    return UnavailableExplainer(f"{provider} explanation provider is not configured for this disabled M5 deployment")
