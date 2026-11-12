import unittest

from experiments.recovery_policy_v2 import (
    BRIDGE,
    GATEWAY,
    FaultState,
    assign_labels,
    make_topology,
    shortest_path,
)


class RecoveryPolicyV2Tests(unittest.TestCase):
    def test_bottleneck_exclusion_disconnects_sources_but_sparse_has_detour(self):
        bottleneck, _ = make_topology("bottleneck")
        sparse, _ = make_topology("sparse")
        self.assertIsNone(shortest_path(0, bottleneck, excluded={BRIDGE}))
        self.assertIsNotNone(shortest_path(0, sparse, excluded={BRIDGE}))
        self.assertEqual(shortest_path(0, sparse, excluded={BRIDGE})[-1], GATEWAY)

    def test_fault_quarantine_re_admits_and_boosts_after_consecutive_normals(self):
        state = FaultState()
        self.assertEqual(state.update(4, "fault")[0], "quarantined")
        self.assertEqual(state.update(5, "normal")[0], "quarantined")
        status, readmitted = state.update(6, "normal")
        self.assertTrue(readmitted)
        self.assertEqual(status, "active_boosted")
        self.assertEqual(state.readmission_delay, 2)
        self.assertTrue(state.power_boost)

    def test_blackhole_isolation_is_permanent(self):
        state = FaultState()
        self.assertEqual(state.update(2, "blackhole")[0], "isolated")
        self.assertEqual(state.update(3, "normal")[0], "isolated")

    def test_high_fault_load_increases_fault_nodes_and_keeps_bridge_faulty(self):
        low = assign_labels(17, 0.20)
        high = assign_labels(17, 0.40)
        self.assertEqual(sum(label == "fault" for label in low.values()), 2)
        self.assertEqual(sum(label == "fault" for label in high.values()), 4)
        self.assertEqual(low[BRIDGE], "fault")
        self.assertEqual(high[BRIDGE], "fault")
        self.assertEqual(sum(label == "blackhole" for label in high.values()), 2)

    def test_failed_edge_can_be_removed_for_packet_rerouting(self):
        graph, _ = make_topology("bottleneck")
        route = shortest_path(0, graph, blocked_edges={(0, 10), (10, 0)})
        self.assertIsNotNone(route)
        self.assertEqual(route[0], 0)
        self.assertNotEqual(route[1], 10)


if __name__ == "__main__":
    unittest.main()
