"""Evaluate the Random Forest across reproducible node-wise simulation seeds.

Each run simulates a fresh dataset, applies the existing rolling feature
extractor, and holds out whole nodes while guaranteeing all classes occur in
both the training and test partitions.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.feature_extraction import extract_features
from src.simulator import WSN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "results" / "multi_seed_nodewise"
CLASSES = ("normal", "fault", "blackhole")
NUM_NODES = 20
ROUNDS = 30
TEST_NODES_PER_CLASS = {"normal": 4, "fault": 1, "blackhole": 1}
FAULT_NODES = (1, 5, 14, 18)
BLACKHOLE_NODES = (2, 8, 15, 19)
EXCLUDED_COLUMNS = {"node_id", "x", "y", "round", "label"}


def choose_node_split(data: pd.DataFrame, seed: int):
    """Choose test nodes by class, leaving at least one train node per class."""
    node_labels = data.groupby("node_id")["label"].agg(lambda labels: set(labels))
    if any(len(labels) != 1 for labels in node_labels):
        raise ValueError("Expected each simulated node to have one fixed label.")

    nodes_by_class = {
        label: sorted(node for node, labels in node_labels.items() if label in labels)
        for label in CLASSES
    }
    rng = np.random.default_rng(seed)
    test_nodes = []
    for label in CLASSES:
        candidates = nodes_by_class[label]
        test_count = TEST_NODES_PER_CLASS[label]
        if len(candidates) <= test_count:
            raise ValueError(
                f"Need more than {test_count} {label} nodes to leave some for training; "
                f"found {len(candidates)}."
            )
        test_nodes.extend(rng.choice(candidates, size=test_count, replace=False).tolist())

    test_nodes = sorted(test_nodes)
    train_nodes = sorted(set(node_labels.index) - set(test_nodes))
    return train_nodes, test_nodes


def run_seed(seed: int) -> dict:
    """Run one full simulation, feature extraction, and node-wise evaluation."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RUNS_DIR / f"seed_{seed}_raw.csv"
    features_path = RUNS_DIR / f"seed_{seed}_features.csv"

    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=FAULT_NODES,
        blackhole_nodes=BLACKHOLE_NODES,
    )
    network.simulate(rounds=ROUNDS).to_csv(raw_path, index=False)
    data = extract_features(raw_path, features_path, window_size=3)

    train_nodes, test_nodes = choose_node_split(data, seed)
    train_data = data[data["node_id"].isin(train_nodes)]
    test_data = data[data["node_id"].isin(test_nodes)]
    train_classes = set(train_data["label"])
    test_classes = set(test_data["label"])
    if train_classes != set(CLASSES) or test_classes != set(CLASSES):
        raise ValueError(
            f"Seed {seed} split is missing classes: train={sorted(train_classes)}, "
            f"test={sorted(test_classes)}"
        )

    features = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(train_data[features], train_data["label"])
    predictions = model.predict(test_data[features])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_data["label"],
        predictions,
        labels=list(CLASSES),
        average="macro",
        zero_division=0,
    )

    return {
        "seed": seed,
        "train_nodes": ",".join(map(str, train_nodes)),
        "test_nodes": ",".join(map(str, test_nodes)),
        "train_records": len(train_data),
        "test_records": len(test_data),
        "accuracy": accuracy_score(test_data["label"], predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def main():
    seeds = list(range(42, 52))
    results = pd.DataFrame(run_seed(seed) for seed in seeds)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = pd.DataFrame(
        {"mean": results[metrics].mean(), "std": results[metrics].std(ddof=1)}
    )

    results_path = RUNS_DIR / "per_run_metrics.csv"
    summary_path = RUNS_DIR / "mean_std_metrics.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index_label="metric")

    print("\n=== Multiple-seed node-wise evaluation ===")
    print(f"Seeds: {seeds}")
    print("Metrics use macro averaging across normal, fault, and blackhole.")
    print("\nPer-run metrics:")
    print(results[["seed", *metrics]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMean and sample standard deviation across runs:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nPer-run results saved to: {results_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary saved to: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
