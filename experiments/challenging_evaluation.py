
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
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

RAW_FEATURES = [
    "rssi",
    "packet_delivery_ratio",
    "packet_loss",
    "delay_ms",
    "residual_energy",
    "energy_consumed",
]


def assign_behaviors(seed):
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

    std_columns = [
        f"{column}_std" for column in RAW_FEATURES
    ]

    data[std_columns] = data[std_columns].fillna(0)

    return data


def simulate_dataset(
    seed,
    fault_nodes,
    blackhole_nodes,
    challenging_mode,
):
    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=fault_nodes,
        blackhole_nodes=blackhole_nodes,
        challenging_mode=challenging_mode,
    )

    return extract_features(network.simulate(rounds=ROUNDS))


def evaluate(train_data, test_data, seed, scenario):
    excluded = {
        "node_id", "x", "y", "round", "label"
    }

    feature_columns = [
        col for col in train_data.columns
        if col not in excluded
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["label"]
    y_test = test_data["label"]

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

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            y_test_encoded,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            zero_division=0,
        )
    )

    overall = {
        "seed": seed,
        "scenario": scenario,
        "accuracy": accuracy_score(
            y_test_encoded, predictions
        ),
        "macro_f1": f1_score(
            y_test_encoded,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    class_results = []

    for index, class_name in enumerate(encoder.classes_):
        class_results.append({
            "seed": seed,
            "scenario": scenario,
            "class": class_name,
            "precision": precision[index],
            "recall": recall[index],
            "f1": class_f1[index],
            "support": support[index],
        })

    return overall, class_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_results = []
    class_results = []

    scenarios = [
        "clean_in_domain",
        "challenging_in_domain",
        "clean_to_challenging",
    ]

    for seed in range(1, NUM_SEEDS + 1):

        fault_nodes, blackhole_nodes = assign_behaviors(seed)

        # Generate original clean and challenging datasets
        # with identical node behaviour assignments.
        clean_data = simulate_dataset(
            seed,
            fault_nodes,
            blackhole_nodes,
            challenging_mode=False,
        )

        challenging_data = simulate_dataset(
            seed,
            fault_nodes,
            blackhole_nodes,
            challenging_mode=True,
        )

        clean_train = clean_data[
            clean_data["node_id"].isin(TRAIN_NODES)
        ].copy()

        clean_test = clean_data[
            clean_data["node_id"].isin(TEST_NODES)
        ].copy()

        challenging_train = challenging_data[
            challenging_data["node_id"].isin(TRAIN_NODES)
        ].copy()

        challenging_test = challenging_data[
            challenging_data["node_id"].isin(TEST_NODES)
        ].copy()

        experiments = [
            (
                "clean_in_domain",
                clean_train,
                clean_test,
            ),
            (
                "challenging_in_domain",
                challenging_train,
                challenging_test,
            ),
            (
                "clean_to_challenging",
                clean_train,
                challenging_test,
            ),
        ]

        for scenario, train_data, test_data in experiments:

            overall, per_class = evaluate(
                train_data,
                test_data,
                seed,
                scenario,
            )

            overall_results.append(overall)
            class_results.extend(per_class)

            print(
                f"Seed {seed:02d} | "
                f"{scenario:24s} | "
                f"Accuracy: {overall['accuracy']:.4f} | "
                f"Macro F1: {overall['macro_f1']:.4f}"
            )

    overall_df = pd.DataFrame(overall_results)
    class_df = pd.DataFrame(class_results)

    overall_path = (
        RESULTS_DIR / "challenging_overall_results.csv"
    )

    class_path = (
        RESULTS_DIR / "challenging_per_class_results.csv"
    )

    summary_path = (
        RESULTS_DIR / "challenging_summary.csv"
    )

    overall_df.to_csv(overall_path, index=False)
    class_df.to_csv(class_path, index=False)

    summary = overall_df.groupby("scenario")[
        ["accuracy", "macro_f1"]
    ].agg(["mean", "std"])

    summary.to_csv(summary_path)

    class_summary = class_df.groupby(
        ["scenario", "class"]
    )[["precision", "recall", "f1"]].mean()

    print("\n=== Overall Scenario Summary ===")
    print(summary.round(4))

    print("\n=== Per-Class Mean Results ===")
    print(class_summary.round(4))

    print(f"\nOverall results: {overall_path}")
    print(f"Per-class results: {class_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
