import unittest

from experiments.transient_fault_quarantine import (
    BLACKHOLES,
    FAULT_RELAY,
    GATEWAY,
    QuarantinePolicy,
    shortest_path,
    topology,
)


class TransientFaultQuarantineTests(unittest.TestCase):
    def test_fault_is_quarantined_and_re_admitted_after_two_normal_rounds(self):
        policy = QuarantinePolicy(quarantine_rounds=2, normal_rounds=2)
        self.assertEqual(policy.update(6, "fault"), "quarantined")
        self.assertEqual(policy.update(7, "fault"), "quarantined")
        self.assertEqual(policy.update(8, "normal"), "quarantined")
        self.assertEqual(policy.update(9, "normal"), "active")
        self.assertEqual(policy.readmitted_round, 9)

    def test_blackhole_is_never_on_selected_route(self):
        for scenario in ("sparse", "bottleneck"):
            adjacency, _ = topology(scenario)
            route = shortest_path(0, adjacency)
            self.assertIsNotNone(route)
            self.assertEqual(route[-1], GATEWAY)
            self.assertTrue(BLACKHOLES.isdisjoint(route))

    def test_quarantining_fault_bridge_leaves_bypass_available(self):
        adjacency, _ = topology("bottleneck")
        route = shortest_path(0, adjacency, {FAULT_RELAY})
        self.assertIsNotNone(route)
        self.assertNotIn(FAULT_RELAY, route)


if __name__ == "__main__":
    unittest.main()
