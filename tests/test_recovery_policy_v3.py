import unittest

import numpy as np

from experiments.recovery_policy_v3 import GATEWAY, NODE_COUNT, _transmit, make_topology


class RecoveryPolicyV3Tests(unittest.TestCase):
    def test_failed_hop_uses_and_counts_an_alternate_route(self):
        graph, distances = make_topology("sparse")
        uniforms = np.ones((NODE_COUNT + 1, NODE_COUNT + 1, 3), dtype=float)
        # Force the original 0->10 hop to fail, then let the 0->11 detour succeed.
        for attempt in range(3):
            uniforms[0, 11, attempt] = 0.0
            uniforms[11, 16, attempt] = 0.0
            uniforms[16, 19, attempt] = 0.0
            uniforms[19, GATEWAY, attempt] = 0.0
        energy = {node: 10_000.0 for node in range(NODE_COUNT + 1)}
        labels = {node: "normal" for node in range(NODE_COUNT)}
        result = _transmit([0, 10, 16, 19, GATEWAY], 0, "blind_reroute", {},
            labels, set(), set(), distances, uniforms, energy,
            {"graph": graph, "excluded": set()})
        delivered, _, _, _, trace, fallback_routes = result
        self.assertTrue(delivered)
        self.assertEqual(fallback_routes, 1)
        self.assertIn("11", trace.split("-"))


if __name__ == "__main__":
    unittest.main()
