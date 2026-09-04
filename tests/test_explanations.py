import unittest

from crowd_safety.explanations import DisabledExplainer, FakeExplainer, explain_incident
from crowd_safety.persistence import MemoryPersistence
from crowd_safety.types import IncidentExplanation


class ExplanationTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryPersistence()
        self.store.runs["run-1"] = {"run_id": "run-1", "source_id": "camera-1"}
        self.store.incidents["incident-1"] = {
            "incident_id": "incident-1", "source_id": "camera-1", "region_id": "zone",
            "state": "active", "severity": "high", "started_at_s": 1.0,
            "last_updated_at_s": 2.0, "closed_at_s": None, "peak_risk": 0.8,
            "reason_codes": ["violence_high"],
        }

    def test_disabled_explanation_is_separate_and_does_not_change_incident(self):
        before = dict(self.store.incidents["incident-1"])
        result = explain_incident(self.store, "incident-1", DisabledExplainer())

        self.assertEqual(result.status, "disabled")
        self.assertEqual(self.store.incidents["incident-1"], before)
        self.assertEqual(self.store.get_incident("incident-1")["explanation"]["status"], "disabled")

    def test_fake_success_and_provider_failures_are_stored_without_blocking(self):
        generated = explain_incident(self.store, "incident-1", FakeExplainer("observable movement near zone"))
        self.assertEqual(generated.status, "generated")
        self.assertEqual(generated.text, "observable movement near zone")

        for error, status in ((TimeoutError("timed out"), "unavailable"), (RuntimeError("provider failed"), "failed")):
            with self.subTest(status=status):
                result = explain_incident(self.store, "incident-1", FakeExplainer(error=error))
                self.assertEqual(result.status, status)
                self.assertEqual(self.store.get_incident("incident-1")["incident"]["severity"], "high")

    def test_scoped_record_id_is_used_when_saving_explanation(self):
        store = MemoryPersistence()
        record_id = "run-1--incident-1"
        store.incidents[record_id] = {**self.store.incidents["incident-1"]}
        store.incident_aliases["incident-1"] = [record_id]

        explanation = explain_incident(store, record_id, FakeExplainer("scoped explanation"))

        self.assertEqual(explanation.incident_id, record_id)
        self.assertEqual(store.get_incident(record_id)["explanation"]["text"], "scoped explanation")

    def test_generated_explanation_contract_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            IncidentExplanation("incident-1", "generated", "fake", "model", "")


if __name__ == "__main__":
    unittest.main()
