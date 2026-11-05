
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from src.simulator import WSN


RESULTS_DIR = Path("results")

NUM_SEEDS = 10
NUM_NODES = 20
ROUNDS = 30

TRAIN_NODES = list(range(14))
TEST_NODES = list(range(14, 20))

NOISE_LEVELS = [0.0, 0.10, 0.25, 0.50, 1.0]

RAW_FEATURES = [
    "rssi",
    "packet_delivery_ratio",
    "packet_loss",
    "delay_ms",
    "residual_energy",
    "energy_consumed",
]


def assign_behaviors(seed):
    """Assign fault and blackhole nodes within both partitions."""

    rng = np.random.default_rng(seed)

    train_faults = rng.choice(
        TRAIN_NODES, size=2, replace=False
    ).tolist()

    remaining_train = [
        n for n in TRAIN_NODES if n not in train_faults
    ]

    train_blackholes = rng.choice(
        remaining_train, size=2, replace=False
    ).tolist()

    test_faults = rng.choice(
        TEST_NODES, size=2, replace=False
    ).tolist()

    remaining_test = [
        n for n in TEST_NODES if n not in test_faults
    ]

    test_blackholes = rng.choice(
        remaining_test, size=2, replace=False
    ).tolist()

    return (
        train_faults + test_faults,
        train_blackholes + test_blackholes,
    )


def extract_features(data, window_size=3):
    """Extract rolling mean and standard deviation per node."""

    data = data.sort_values(["node_id", "round"]).copy()
    grouped = data.groupby("node_id", group_keys=False)

    for column in RAW_FEATURES:
        data[f"{column}_mean"] = grouped[column].transform(
            lambda values: values.rolling(
                window=window_size,
                min_periods=1,
            ).mean()
        )

        data[f"{column}_std"] = grouped[column].transform(
            lambda values: values.rolling(
                window=window_size,
                min_periods=1,
            ).std()
        )

    std_columns = [f"{column}_std" for column in RAW_FEATURES]
    data[std_columns] = data[std_columns].fillna(0)

    return data


def add_measurement_noise(data, noise_level, train_data, seed):
    """
    Add Gaussian noise to test telemetry only.

    Noise standard deviation = noise_level *
    corresponding clean training feature standard deviation.
    """

    noisy_data = data.copy()
    rng = np.random.default_rng(seed)

    for column in RAW_FEATURES:
        scale = train_data[column].std()

        if pd.isna(scale) or scale == 0:
            continue

        noise = rng.normal(
            loc=0,
            scale=noise_level * scale,
            size=len(noisy_data),
        )

        noisy_data[column] += noise

    # Keep naturally bounded measurements within valid limits.
    noisy_data["packet_delivery_ratio"] = (
        noisy_data["packet_delivery_ratio"].clip(0, 1)
    )

    noisy_data["packet_loss"] = (
        noisy_data["packet_loss"].clip(0, 1)
    )

    noisy_data["residual_energy"] = (
        noisy_data["residual_energy"].clip(lower=0)
    )

    noisy_data["delay_ms"] = (
        noisy_data["delay_ms"].clip(lower=0)
    )

    noisy_data["energy_consumed"] = (
        noisy_data["energy_consumed"].clip(lower=0)
    )

    return noisy_data


def evaluate_model(train_data, test_data, seed):
    """Train on clean features and evaluate on test features."""

    excluded_columns = {
        "node_id", "x", "y", "round", "label"
    }

    feature_columns = [
        col for col in train_data.columns
        if col not in excluded_columns
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["label"]
    y_test = test_data["label"]

    expected_classes = {"normal", "fault", "blackhole"}

    if set(y_train.unique()) != expected_classes:
        raise ValueError("Training partition is missing a class.")

    if set(y_test.unique()) != expected_classes:
        raise ValueError("Testing partition is missing a class.")

    encoder = LabelEncoder()
    encoder.fit(y_train)

    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
    )

    model.fit(X_train, y_train_encoded)
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test_encoded, predictions),
        "macro_precision": precision_score(
            y_test_encoded,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test_encoded,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test_encoded,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def run_one_seed(seed):
    fault_nodes, blackhole_nodes = assign_behaviors(seed)

    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=fault_nodes,
        blackhole_nodes=blackhole_nodes,
    )

    raw_data = network.simulate(rounds=ROUNDS)

    train_raw = raw_data[
        raw_data["node_id"].isin(TRAIN_NODES)
    ].copy()

    test_raw = raw_data[
        raw_data["node_id"].isin(TEST_NODES)
    ].copy()

    seed_results = []

    # Train on clean data once for each seed.
    train_features = extract_features(train_raw)

    for noise_level in NOISE_LEVELS:

        noisy_test_raw = add_measurement_noise(
            data=test_raw,
            noise_level=noise_level,
            train_data=train_raw,
            seed=seed + int(noise_level * 10000),
        )

        test_features = extract_features(noisy_test_raw)

        metrics = evaluate_model(
            train_features,
            test_features,
            seed,
        )

        seed_results.append({
            "seed": seed,
            "noise_level": noise_level,
            **metrics,
        })

    return seed_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(1, NUM_SEEDS + 1):
        seed_results = run_one_seed(seed)
        all_results.extend(seed_results)

        for result in seed_results:
            print(
                f"Seed {seed:02d} | "
                f"Noise: {result['noise_level'] * 100:>3.0f}% | "
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    results_path = RESULTS_DIR / "noise_robustness_results.csv"
    results_df.to_csv(results_path, index=False)

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    summary = results_df.groupby("noise_level")[
        metric_columns
    ].agg(["mean", "std"])

    summary_path = RESULTS_DIR / "noise_robustness_summary.csv"
    summary.to_csv(summary_path)

    print("\n=== Noise Robustness Summary ===")
    print(summary.round(4))

    print(f"\nDetailed results: {results_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
