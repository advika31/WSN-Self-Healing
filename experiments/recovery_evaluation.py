
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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

GATEWAY = (50, 50)

# Number of source nodes whose traffic is represented
# by the selected shared gateway relay.
OFFERED_PACKETS_PER_ROUND = len(TEST_NODES) - 1


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


def train_and_predict(data, seed):
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
        column for column in data.columns
        if column not in excluded
    ]

    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=seed,
    )

    model.fit(
        train_data[feature_columns],
        encoder.transform(train_data["label"]),
    )

    test_data["predicted_label"] = encoder.inverse_transform(
        model.predict(test_data[feature_columns])
    )

    return test_data


def distance_to_gateway(row):
    return float(
        np.sqrt(
            (row["x"] - GATEWAY[0]) ** 2
            + (row["y"] - GATEWAY[1]) ** 2
        )
    )


def choose_relay(candidates, strategy):
    """
    Select one gateway relay using the specified recovery policy.

    no_recovery:
        Every node remains eligible.

    generic_rerouting:
        Exclude all nodes predicted as anomalous.

    diagnosis_aware:
        Exclude predicted blackhole nodes.
        Keep predicted faulty nodes with a 1.5 cost penalty.
    """

    candidates = candidates.copy()

    candidates["distance"] = candidates.apply(
        distance_to_gateway,
        axis=1,
    )

    candidates["routing_penalty"] = 1.0

    if strategy == "no_recovery":
        eligible = candidates.copy()

    elif strategy == "generic_rerouting":
        eligible = candidates[
            candidates["predicted_label"] == "normal"
        ].copy()

    elif strategy == "diagnosis_aware":
        eligible = candidates[
            candidates["predicted_label"] != "blackhole"
        ].copy()

        eligible.loc[
            eligible["predicted_label"] == "fault",
            "routing_penalty",
        ] = 1.5

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    fallback_used = False

    # If every candidate is excluded, retain all nodes
    # as a documented emergency fallback.
    if eligible.empty:
        eligible = candidates.copy()
        fallback_used = True

    eligible["routing_cost"] = (
        eligible["distance"] * eligible["routing_penalty"]
    )

    selected = eligible.sort_values(
        ["routing_cost", "node_id"]
    ).iloc[0]

    return selected, fallback_used


def evaluate_seed(seed):
    fault_nodes, blackhole_nodes = assign_behaviors(seed)

    network = WSN(
        num_nodes=NUM_NODES,
        seed=seed,
        fault_nodes=fault_nodes,
        blackhole_nodes=blackhole_nodes,
        challenging_mode=True,
    )

    data = extract_features(
        network.simulate(rounds=ROUNDS)
    )

    test_data = train_and_predict(data, seed)

    strategies = [
        "no_recovery",
        "generic_rerouting",
        "diagnosis_aware",
    ]

    round_results = []

    for round_number in range(1, ROUNDS + 1):

        round_data = test_data[
            test_data["round"] == round_number
        ].copy()

        for strategy in strategies:

            selected, fallback_used = choose_relay(
                round_data,
                strategy,
            )

            delivered = (
                selected["packet_delivery_ratio"]
                * OFFERED_PACKETS_PER_ROUND
            )

            round_results.append({
                "seed": seed,
                "round": round_number,
                "strategy": strategy,
                "selected_node": int(selected["node_id"]),
                "predicted_class": selected["predicted_label"],
                "actual_class": selected["label"],
                "packet_delivery_ratio": selected[
                    "packet_delivery_ratio"
                ],
                "delay_ms": selected["delay_ms"],
                "energy_consumed": selected["energy_consumed"],
                "distance_to_gateway": selected["distance"],
                "routing_cost": selected["routing_cost"],
                "estimated_packets_delivered": delivered,
                "offered_packets": OFFERED_PACKETS_PER_ROUND,
                "fallback_used": fallback_used,
                "selected_fault_or_attack": (
                    selected["label"] != "normal"
                ),
                "selected_blackhole": (
                    selected["label"] == "blackhole"
                ),
            })

    return round_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(1, NUM_SEEDS + 1):

        results = evaluate_seed(seed)
        all_results.extend(results)

        seed_df = pd.DataFrame(results)

        seed_summary = seed_df.groupby("strategy")[
            [
                "packet_delivery_ratio",
                "delay_ms",
                "energy_consumed",
            ]
        ].mean()

        print(f"\nSeed {seed:02d}")
        print(seed_summary.round(4))

    results_df = pd.DataFrame(all_results)

    detailed_path = (
        RESULTS_DIR / "recovery_round_results.csv"
    )

    results_df.to_csv(detailed_path, index=False)

    # First aggregate rounds within each seed and strategy.
    per_seed = results_df.groupby(
        ["seed", "strategy"]
    ).agg(
        packet_delivery_ratio=("packet_delivery_ratio", "mean"),
        delay_ms=("delay_ms", "mean"),
        energy_consumed=("energy_consumed", "sum"),
        estimated_packets_delivered=(
            "estimated_packets_delivered", "sum"
        ),
        offered_packets=("offered_packets", "sum"),
        fallback_rounds=("fallback_used", "sum"),
        selected_fault_or_attack=(
            "selected_fault_or_attack", "mean"
        ),
        selected_blackhole=("selected_blackhole", "mean"),
    ).reset_index()

    per_seed["estimated_network_pdr"] = (
        per_seed["estimated_packets_delivered"]
        / per_seed["offered_packets"]
    )

    per_seed_path = (
        RESULTS_DIR / "recovery_per_seed_results.csv"
    )

    per_seed.to_csv(per_seed_path, index=False)

    metric_columns = [
        "packet_delivery_ratio",
        "estimated_network_pdr",
        "delay_ms",
        "energy_consumed",
        "fallback_rounds",
        "selected_fault_or_attack",
        "selected_blackhole",
    ]

    summary = per_seed.groupby("strategy")[
        metric_columns
    ].agg(["mean", "std"])

    summary_path = (
        RESULTS_DIR / "recovery_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n=== Recovery Strategy Summary ===")
    print(summary.round(4))

    print(f"\nDetailed rounds: {detailed_path}")
    print(f"Per-seed results: {per_seed_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
