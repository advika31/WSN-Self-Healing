"""Measure node-wise Random Forest performance under sensor measurement noise.

This is a separate robustness experiment. It leaves the simulator, baseline
datasets, and original multi-seed results unchanged.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiments.multi_seed_nodewise_evaluation import (
    BLACKHOLE_NODES,
    CLASSES,
    EXCLUDED_COLUMNS,
    FAULT_NODES,
    NUM_NODES,
    ROUNDS,
    choose_node_split,
)
from src.feature_extraction import extract_features
from src.simulator import WSN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "noisy_robustness"
NOISE_LEVELS = {
    # Additive Gaussian standard deviations in the units of each measurement.
    "moderate": {"rssi": 10.0, "packet_delivery_ratio": 0.18, "delay_ms": 30.0},
    "strong": {"rssi": 18.0, "packet_delivery_ratio": 0.30, "delay_ms": 60.0},
}
SEEDS = tuple(range(42, 52))


def add_measurement_noise(data: pd.DataFrame, seed: int, level: str) -> pd.DataFrame:
    """Add reproducible, class-independent noise before rolling extraction."""
    noisy = data.copy()
    rng = np.random.default_rng(seed + 100_000)
    scales = NOISE_LEVELS[level]
    for column, standard_deviation in scales.items():
        noisy[column] += rng.normal(0.0, standard_deviation, size=len(noisy))

    noisy["packet_delivery_ratio"] = noisy["packet_delivery_ratio"].clip(0.0, 1.0)
    noisy["packet_loss"] = 1.0 - noisy["packet_delivery_ratio"]
    noisy["delay_ms"] = noisy["delay_ms"].clip(lower=1.0)
    return noisy


def run_one(seed: int, level: str) -> dict:
    run_dir = OUTPUT_DIR / level
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / f"seed_{seed}_raw.csv"
    feature_path = run_dir / f"seed_{seed}_features.csv"

    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=FAULT_NODES,
        blackhole_nodes=BLACKHOLE_NODES,
    )
    raw = network.simulate(rounds=ROUNDS)
    noisy = add_measurement_noise(raw, seed, level)
    noisy.to_csv(raw_path, index=False)
    # The existing extractor prints a sample for interactive use; suppress
    # that repeated preview while retaining its normal feature calculations.
    with redirect_stdout(StringIO()):
        features = extract_features(raw_path, feature_path, window_size=3)

    train_nodes, test_nodes = choose_node_split(features, seed)
    train = features[features["node_id"].isin(train_nodes)]
    test = features[features["node_id"].isin(test_nodes)]
    if set(train["label"]) != set(CLASSES) or set(test["label"]) != set(CLASSES):
        raise ValueError(f"Seed {seed} ({level}) does not contain all classes in both splits.")

    feature_columns = [column for column in features.columns if column not in EXCLUDED_COLUMNS]
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(train[feature_columns], train["label"])
    predictions = model.predict(test[feature_columns])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test["label"],
        predictions,
        labels=list(CLASSES),
        average="macro",
        zero_division=0,
    )
    return {
        "noise_level": level,
        "seed": seed,
        "train_nodes": ",".join(map(str, train_nodes)),
        "test_nodes": ",".join(map(str, test_nodes)),
        "train_records": len(train),
        "test_records": len(test),
        "accuracy": accuracy_score(test["label"], predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(
        run_one(seed, level)
        for level in NOISE_LEVELS
        for seed in SEEDS
    )
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = (
        runs.groupby("noise_level")[metrics]
        .agg(["mean", "std"])
        .sort_index()
    )
    per_run_path = OUTPUT_DIR / "per_run_metrics.csv"
    summary_path = OUTPUT_DIR / "mean_std_metrics.csv"
    runs.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path)

    print("=== Noisy node-wise robustness evaluation ===")
    print(f"Seeds: {list(SEEDS)}")
    print("Macro metrics are averaged equally over normal, fault, and blackhole.")
    print("\nPer-run metrics:")
    print(runs[["noise_level", "seed", *metrics]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    print("\nMean and sample standard deviation across seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nPer-run results: {per_run_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
