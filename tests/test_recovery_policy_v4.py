import unittest

import numpy as np

from experiments import recovery_policy_v4 as v4

GW = v4.GATEWAY_ID


def line_graph():
    # 0 -> 1 -> GW (short, relay 1) and 0 -> 2 -> GW (long, relay 2)
    pos = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (10.0, 10.0), GW: (20.0, 0.0)}
    return pos, v4.build_graph(pos, 15.0)


class V4Tests(unittest.TestCase):
    def test_exclusion_tier_avoids_flagged_relay(self):
        _, graph = line_graph()
        path, fb = v4.find_route(0, graph, [{1: np.inf}])
        self.assertEqual(path, [0, 2, GW])
        self.assertFalse(fb)

    def test_sole_bridge_falls_back_instead_of_dropping(self):
        # Only relay 1 connects 0 to the gateway; excluding it must not strand the
        # packet: relaxed tier (penalty) or last-resort fallback must route.
        pos = {0: (0.0, 0.0), 1: (15.0, 0.0), GW: (30.0, 0.0)}
        graph = v4.build_graph(pos, 16.0)
        path, fb = v4.find_route(0, graph, [{1: np.inf}, {1: 1.5}])
        self.assertEqual(path, [0, 1, GW])
        path, fb = v4.find_route(0, graph, [{1: np.inf}])
        self.assertEqual(path, [0, 1, GW])
        self.assertTrue(fb)

    def test_silent_blackhole_is_invisible_to_hop_retries(self):
        pos, graph = line_graph()
        labels = {0: "normal", 1: "blackhole", 2: "normal"}
        scenario = dict(radio=30.0, drop="silent", fault="persistent", crash=0.0)
        table = v4._LinkTable(np.zeros((v4.NODES + 1, v4.NODES + 1, v4.MAX_ATTEMPTS)))
        drop_u = np.ones(v4.NODES + 1)       # always >= forwarding p -> dropped
        energy = np.zeros(v4.NODES + 100)
        out = v4.send_packet(0, "blind_reroute", [0, 1, GW], False, graph, pos,
                             labels, scenario, {n: True for n in range(3)}, set(),
                             [], table, drop_u, energy)
        self.assertFalse(out["delivered"])
        self.assertTrue(out["silent_loss"])
        self.assertEqual(out["retx"], 0)     # sender believed the hop succeeded
        self.assertEqual(out["changes"], 0)  # so no reroute was triggered

    def test_detectable_blackhole_triggers_reroute(self):
        pos, graph = line_graph()
        labels = {0: "normal", 1: "blackhole", 2: "normal"}
        scenario = dict(radio=30.0, drop="detectable", fault="persistent", crash=0.0)
        arr = np.full((v4.NODES + 1, v4.NODES + 1, v4.MAX_ATTEMPTS), 0.5)
        out = v4.send_packet(0, "blind_reroute", [0, 1, GW], False, graph, pos,
                             labels, scenario, {n: True for n in range(3)}, set(),
                             [], v4._LinkTable(arr), np.zeros(v4.NODES + 1),
                             np.zeros(v4.NODES + 100))
        self.assertTrue(out["delivered"])
        self.assertGreater(out["changes"], 0)

    def test_memory_isolates_after_two_blackhole_predictions_and_readmits_faults(self):
        mem = v4.Memory()
        base = {n: "normal" for n in range(v4.NODES)}
        for rnd in range(3):
            preds = dict(base)
            preds[3] = "blackhole" if rnd < 2 else "normal"
            preds[4] = "fault" if rnd == 0 else "normal"
            mem.update(preds)
        self.assertIn(3, mem.isolated)       # permanent after confirmation
        self.assertNotIn(4, mem.quarantined())  # fault readmitted after 2 normals

    def test_holm_is_monotone_and_bounded(self):
        adj = v4.holm([0.01, 0.04, 0.03])
        self.assertTrue(np.all(adj <= 1.0))
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertGreaterEqual(adj[1], adj[2])

    def test_random_labels_prevalence(self):
        labels = v4.random_labels(311)
        counts = {c: list(labels.values()).count(c) for c in v4.CLASSES}
        self.assertEqual(counts, {"normal": 12, "fault": 4, "blackhole": 4})

    def test_bottleneck_layout_designed_labels(self):
        pos, labels = v4.bottleneck_layout(311)
        inner = [labels[n] for n in range(4)]
        self.assertEqual(sorted(inner), ["blackhole", "fault", "fault", "normal"])

    def test_recovery_requires_three_consecutive_rounds(self):
        pdr = np.array([0.9] * 14 + [0.95, 0.94, 0.95, 0.95, 0.95] + [0.9] * 5)
        self.assertEqual(v4.sustained_recovery_rounds(pdr, 1.0, 15), 2.0)

    def test_non_recovery_is_censored_at_remaining_horizon(self):
        pdr = np.array([0.8] * 40)
        self.assertEqual(v4.sustained_recovery_rounds(pdr, 0.9, 15), 26.0)


if __name__ == "__main__":
    unittest.main()
