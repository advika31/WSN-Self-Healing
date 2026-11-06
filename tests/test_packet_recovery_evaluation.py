import unittest

from experiments.packet_recovery_evaluation import (
    GATEWAY_ID,
    route_cost,
    shortest_path,
    simulate_packet,
)


class PacketRecoveryTests(unittest.TestCase):
    def test_policy_costs_encode_diagnosis_difference(self):
        predictions = {1: "fault", 2: "blackhole"}
        self.assertEqual(route_cost(0, 1, 10, predictions, "no_recovery"), 10)
        self.assertEqual(route_cost(0, 1, 10, predictions, "generic_rerouting"), float("inf"))
        self.assertEqual(route_cost(0, 1, 10, predictions, "diagnosis_aware"), 15)
        self.assertEqual(route_cost(0, 2, 10, predictions, "diagnosis_aware"), float("inf"))

    def test_shortest_path_routes_around_generic_exclusions(self):
        adjacency = {
            0: [(1, 1.0), (2, 2.0)],
            1: [(0, 1.0), (GATEWAY_ID, 1.0)],
            2: [(0, 2.0), (GATEWAY_ID, 2.0)],
            GATEWAY_ID: [(1, 1.0), (2, 2.0)],
        }
        predictions = {1: "blackhole", 2: "normal"}
        self.assertEqual(shortest_path(0, adjacency, predictions, "no_recovery"),
                         [0, 1, GATEWAY_ID])
        self.assertEqual(shortest_path(0, adjacency, predictions, "generic_rerouting"),
                         [0, 2, GATEWAY_ID])

    def test_recovery_reroutes_after_retries_and_no_recovery_stops(self):
        adjacency = {
            0: [(1, 1.0), (2, 2.0)],
            1: [(0, 1.0), (GATEWAY_ID, 1.0)],
            2: [(0, 2.0), (GATEWAY_ID, 2.0)],
            GATEWAY_ID: [(1, 1.0), (2, 2.0)],
        }
        positions = {0: (0.0, 0.0), 1: (1.0, 0.0),
                     2: (2.0, 0.0), GATEWAY_ID: (2.0, 1.0)}
        predictions = {1: "normal", 2: "normal"}
        labels = {0: "normal", 1: "normal", 2: "normal"}
        trials = {(0, 1): [0.99, 0.99, 0.99], (1, GATEWAY_ID): [0.0] * 3,
                  (0, 2): [0.0] * 3, (2, GATEWAY_ID): [0.0] * 3}
        initial = [0, 1, GATEWAY_ID]
        baseline = simulate_packet(0, "no_recovery", initial, adjacency,
                                   positions, predictions, labels, trials)
        recovering = simulate_packet(0, "generic_rerouting", initial, adjacency,
                                     positions, predictions, labels, trials)
        self.assertFalse(baseline["delivered"])
        self.assertTrue(recovering["delivered"])
        self.assertGreater(recovering["route_changes"], 0)
        self.assertEqual(baseline["retransmissions"], 3)


if __name__ == "__main__":
    unittest.main()
