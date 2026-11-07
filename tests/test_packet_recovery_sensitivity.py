import unittest

import pandas as pd

from experiments.packet_recovery_sensitivity import (
    assign_labels,
    out_of_fold_predictions,
    scenarios,
)


class SensitivityDesignTests(unittest.TestCase):
    def test_scenario_grid_is_one_factor_at_a_time(self):
        rows = scenarios()
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["radio_range"] for r in rows}, {35.0, 48.0, 60.0, 75.0})
        self.assertEqual({r["num_nodes"] for r in rows}, {20, 30, 50})
        self.assertEqual({r["blackhole_p"] for r in rows}, {0.05, 0.08, 0.20})
        self.assertEqual({r["fault_p"] for r in rows}, {0.40, 0.62, 0.80})

    def test_node_behavior_prevalence_scales(self):
        for n in (20, 30, 50):
            labels = assign_labels(n, 102)
            counts = pd.Series(labels).value_counts().to_dict()
            self.assertEqual(counts["fault"], round(n * 0.20))
            self.assertEqual(counts["blackhole"], round(n * 0.20))
            self.assertEqual(counts["normal"], n - round(n * 0.40))

    def test_crossfit_predicts_every_node_round(self):
        rows = []
        classes = ["normal", "fault", "blackhole"]
        for node in range(15):
            label = classes[node % 3]
            for rnd in range(1, 4):
                rows.append(dict(node_id=node, round=rnd, label=label,
                                 feature=float(node % 3) + rnd / 10))
        predictions = out_of_fold_predictions(pd.DataFrame(rows), seed=7)
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(set(predictions), {(n, r) for n in range(15) for r in range(1, 4)})


if __name__ == "__main__":
    unittest.main()
