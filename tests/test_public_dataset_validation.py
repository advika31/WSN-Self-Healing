import unittest

import numpy as np
import pandas as pd

from experiments.public_dataset_validation import (
    CLASSES_INTEL, cluster_bootstrap_metrics, grouped_oof_metrics,
    inject_intel_faults, intel_features,
)


class PublicDatasetValidationTests(unittest.TestCase):
    def test_grouped_oof_keeps_node_groups_separate(self):
        groups, labels, values = [], [], []
        for node in range(20):
            label = "a" if node % 2 == 0 else "b"
            for sample in range(4):
                groups.append(node)
                labels.append(label)
                values.append([node % 2 + sample * 0.001])
        features = pd.DataFrame(values, columns=["signal"])
        summary, class_metrics, fold_metrics, predictions = grouped_oof_metrics(
            features, labels, groups, ["a", "b"], folds=5)
        self.assertEqual(summary["n_rows"], 80)
        self.assertEqual(summary["n_groups"], 20)
        self.assertEqual(len(predictions), 80)
        self.assertEqual(len(class_metrics), 2)
        self.assertEqual(len(fold_metrics), 5)
        self.assertTrue((fold_metrics.n_test_nodes == 4).all())

    def test_intel_fault_injection_is_reproducible_and_covers_classes(self):
        rows = []
        for mote_id in range(1, 4):
            for index in range(500):
                rows.append({"mote_id": mote_id,
                    "time_bin": pd.Timestamp("2004-01-01") + pd.Timedelta(minutes=5 * index),
                    "temperature": 20.0 + 0.002 * index + np.sin(index / 20),
                    "reading_count": 1})
        raw = pd.DataFrame(rows)
        first = inject_intel_faults(raw, seed=19, episodes_per_type=3,
                                    episode_bins=4, spikes_per_node=5)
        second = inject_intel_faults(raw, seed=19, episodes_per_type=3,
                                     episode_bins=4, spikes_per_node=5)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first.label.unique()), set(CLASSES_INTEL))
        self.assertEqual(len(raw), len(first))
        self.assertEqual(set(raw.columns), set(first.columns) - {"label"})

    def test_intel_features_do_not_use_mote_id_as_a_predictor(self):
        rows = []
        for mote_id in range(1, 4):
            for index in range(500):
                rows.append({"mote_id": mote_id,
                    "time_bin": pd.Timestamp("2004-01-01") + pd.Timedelta(minutes=5 * index),
                    "temperature": 20.0 + np.sin(index / 10), "reading_count": 1})
        injected = inject_intel_faults(pd.DataFrame(rows), seed=23,
            episodes_per_type=2, episode_bins=4, spikes_per_node=5)
        features, labels, groups = intel_features(injected)
        self.assertNotIn("mote_id", features.columns)
        self.assertNotIn("time_bin", features.columns)
        self.assertEqual(len(features), len(labels))
        self.assertEqual(len(groups), len(labels))

    def test_cluster_bootstrap_intervals_resample_groups(self):
        labels = ["a", "a", "b", "b"] * 3
        predictions = ["a", "b", "b", "a"] * 3
        groups = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        metrics, summary = cluster_bootstrap_metrics(
            labels, predictions, groups, ["a", "b"], replicates=200, seed=7)
        self.assertEqual(summary["cluster_bootstrap_unit"], "node/mote")
        self.assertEqual(summary["cluster_bootstrap_replicates"], 200)
        self.assertEqual(len(metrics), 2)
        self.assertTrue((metrics.f1_ci95_low <= metrics.f1).all())
        self.assertTrue((metrics.f1 <= metrics.f1_ci95_high).all())


if __name__ == "__main__":
    unittest.main()
