
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

from src.simulator import WSN


RESULTS_DIR = Path("results")

NUM_SEEDS = 10
NUM_NODES = 20
ROUNDS = 30

TRAIN_NODES = list(range(14))
TEST_NODES = list(range(14, 20))

CLASSES = ["normal", "fault", "blackhole"]

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


def rule_based_predictions(data):
    """
    Fixed threshold baseline.

    These thresholds are illustrative, predeclared rules.
    They are not learned or tuned on the test set.
    """

    predictions = []

    for _, row in data.iterrows():

        pdr = row["packet_delivery_ratio"]
        rssi = row["rssi"]
        delay = row["delay_ms"]

        # First identify severe packet dropping.
        if pdr <= 0.45:
            predictions.append("blackhole")

        # Then identify possible link/sensor degradation.
        elif (
            rssi <= -72
            or delay >= 75
            or pdr <= 0.65
        ):
            predictions.append("fault")

        else:
            predictions.append("normal")

    return np.array(predictions)


def evaluate_predictions(
    y_true,
    y_pred,
    encoder,
    seed,
    model_name,
):
    y_true_encoded = encoder.transform(y_true)
    y_pred_encoded = encoder.transform(y_pred)

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            y_true_encoded,
            y_pred_encoded,
            labels=np.arange(len(encoder.classes_)),
            zero_division=0,
        )
    )

    overall = {
        "seed": seed,
        "model": model_name,
        "accuracy": accuracy_score(
            y_true_encoded, y_pred_encoded
        ),
        "macro_f1": f1_score(
            y_true_encoded,
            y_pred_encoded,
            average="macro",
            zero_division=0,
        ),
    }

    per_class = []

    for index, class_name in enumerate(encoder.classes_):
        per_class.append({
            "seed": seed,
            "model": model_name,
            "class": class_name,
            "precision": precision[index],
            "recall": recall[index],
            "f1": class_f1[index],
            "support": support[index],
        })

    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=np.arange(len(encoder.classes_)),
    )

    return overall, per_class, matrix


def run_one_seed(seed):
    fault_nodes, blackhole_nodes = assign_behaviors(seed)

    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=fault_nodes,
        blackhole_nodes=blackhole_nodes,
        challenging_mode=True,
    )

    data = extract_features(network.simulate(rounds=ROUNDS))

    train_data = data[
        data["node_id"].isin(TRAIN_NODES)
    ].copy()

    test_data = data[
        data["node_id"].isin(TEST_NODES)
    ].copy()

    excluded = {
        "node_id", "x", "y", "round", "label"
    }

    feature_columns = [
        col for col in data.columns
        if col not in excluded
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["label"]
    y_test = test_data["label"]

    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    y_train_encoded = encoder.transform(y_train)

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=seed,
        ),
    }

    results = []
    class_results = []
    matrices = {}

    # Evaluate rule-based baseline.
    rule_predictions = rule_based_predictions(test_data)

    overall, per_class, matrix = evaluate_predictions(
        y_test,
        rule_predictions,
        encoder,
        seed,
        "rule_based",
    )

    results.append(overall)
    class_results.extend(per_class)
    matrices["rule_based"] = matrix

    # Evaluate ML baselines.
    for model_name, model in models.items():

        model.fit(X_train, y_train_encoded)

        predictions_encoded = model.predict(X_test)

        predictions = encoder.inverse_transform(
            predictions_encoded
        )

        overall, per_class, matrix = evaluate_predictions(
            y_test,
            predictions,
            encoder,
            seed,
            model_name,
        )

        results.append(overall)
        class_results.extend(per_class)
        matrices[model_name] = matrix

    return results, class_results, matrices


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_class_results = []

    total_matrices = {
        model: np.zeros((3, 3), dtype=int)
        for model in [
            "rule_based",
            "decision_tree",
            "random_forest",
        ]
    }

    for seed in range(1, NUM_SEEDS + 1):

        results, class_results, matrices = run_one_seed(seed)

        all_results.extend(results)
        all_class_results.extend(class_results)

        for model_name, matrix in matrices.items():
            total_matrices[model_name] += matrix

        for result in results:
            print(
                f"Seed {seed:02d} | "
                f"{result['model']:16s} | "
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    class_df = pd.DataFrame(all_class_results)

    results_df.to_csv(
        RESULTS_DIR / "baseline_overall_results.csv",
        index=False,
    )

    class_df.to_csv(
        RESULTS_DIR / "baseline_per_class_results.csv",
        index=False,
    )

    summary = results_df.groupby("model")[
        ["accuracy", "macro_f1"]
    ].agg(["mean", "std"])

    class_summary = class_df.groupby(
        ["model", "class"]
    )[["precision", "recall", "f1"]].mean()

    summary.to_csv(
        RESULTS_DIR / "baseline_summary.csv"
    )

    print("\n=== Overall Baseline Comparison ===")
    print(summary.round(4))

    print("\n=== Per-Class Mean Results ===")
    print(class_summary.round(4))

    print("\n=== Aggregate Confusion Matrices ===")
    print("Class order:", CLASSES)

    for model_name, matrix in total_matrices.items():
        print(f"\n{model_name}:")
        print(matrix)

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
