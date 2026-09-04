import unittest

from crowd_safety.config import FusionConfig
from crowd_safety.incidents import IncidentEngine, replay_incidents
from crowd_safety.types import CrowdFeatureRecord, FusionPoint


def point(timestamp, risk, *, strategy="temporal"):
    crowd = CrowdFeatureRecord("camera-1", "zone", timestamp, "available", occupancy=4)
    return FusionPoint(
        "camera-1", "zone", timestamp, strategy, crowd, "available", None, "unavailable", None, None,
        None, False, {}, None, None, 0.0, risk, ("crowd_context",) if risk else (),
    )


class IncidentTest(unittest.TestCase):
    def test_one_spike_does_not_become_active_and_persistent_signal_is_one_incident(self):
        config = FusionConfig(persistence_s=2.0, quiet_period_s=1.0, decay_s=1.0)
        engine = IncidentEngine(config)
        spike = engine.update(point(0.0, 0.8))
        self.assertEqual(spike[0].state, "candidate")
        self.assertEqual(spike[1][0].to_state, "candidate")
        replay = replay_incidents([point(i, 0.8) for i in range(5)], config)
        ids = {item.incident_id for item in replay.incidents if item.state != "closed"}
        self.assertEqual(len(ids), 1)
        self.assertEqual(replay.incidents[-1].state, "escalating")

    def test_brief_signal_closes_after_decay_and_quiet_period(self):
        config = FusionConfig(persistence_s=3.0, decay_s=1.0, quiet_period_s=2.0)
        engine = IncidentEngine(config)
        engine.update(point(0.0, 0.6))
        closed, transitions = engine.update(point(2.0, 0.0))
        self.assertEqual(closed.state, "closed")
        self.assertEqual(transitions[-1].cause, "decay")
        self.assertEqual(transitions[-1].from_state, "candidate")

    def test_resolving_requires_quiet_time_before_close(self):
        config = FusionConfig(persistence_s=0.0 + 1.0, quiet_period_s=2.0)
        engine = IncidentEngine(config)
        for timestamp in (0.0, 1.0, 2.0):
            engine.update(point(timestamp, 0.8))
        resolving, _ = engine.update(point(3.0, 0.0))
        self.assertEqual(resolving.state, "resolving")
        closed, _ = engine.update(point(5.0, 0.0))
        self.assertEqual(closed.state, "closed")

    def test_low_risk_during_resolution_does_not_cancel_quiet_timer(self):
        config = FusionConfig(persistence_s=1.0, quiet_period_s=2.0, hysteresis=0.05)
        engine = IncidentEngine(config)
        for timestamp in (0.0, 1.0, 2.0):
            engine.update(point(timestamp, 0.8))
        engine.update(point(3.0, 0.0))
        still_resolving, _ = engine.update(point(4.0, 0.3))
        self.assertEqual(still_resolving.state, "resolving")
        closed, _ = engine.update(point(5.0, 0.0))
        self.assertEqual(closed.state, "closed")

    def test_replay_is_deterministic_and_severity_uses_boundaries(self):
        config = FusionConfig(persistence_s=1.0)
        stream = [point(0.0, 0.6), point(1.0, 0.9), point(2.0, 0.9)]
        first = replay_incidents(stream, config)
        second = replay_incidents(stream, config)
        self.assertEqual(first, second)
        self.assertIn(first.incidents[-1].severity, {"high", "critical"})
        self.assertTrue(all(item.timestamp_s <= 2.0 for item in first.transitions))


if __name__ == "__main__":
    unittest.main()
