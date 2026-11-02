import pandas as pd
import joblib

from src.diagnosis import diagnose_and_recover
from src.recovery import select_route, apply_recovery


FEATURES_PATH = "data/processed/wsn_features.csv"
MODEL_PATH = "results/anomaly_detector.pkl"
ENCODER_PATH = "results/label_encoder.pkl"


def main():
    print("\n=== Diagnosis-Aware Self-Healing WSN ===\n")

    # Load processed data and trained model.
    data = pd.read_csv(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    excluded_columns = {
        "node_id",
        "x",
        "y",
        "round",
        "label",
    }

    feature_columns = [
        column for column in data.columns
        if column not in excluded_columns
    ]

    # Use the most recent reading for each node.
    latest_data = (
        data.sort_values("round")
        .groupby("node_id")
        .tail(1)
        .sort_values("node_id")
        .copy()
    )

    X_latest = latest_data[feature_columns]

    predictions_encoded = model.predict(X_latest)
    predictions = label_encoder.inverse_transform(
        predictions_encoded
    )

    gateway = (50, 50)

    nodes = {
        int(row.node_id): (float(row.x), float(row.y))
        for row in latest_data.itertuples()
    }

    node_status = {
        node_id: "active"
        for node_id in nodes
    }

    print("Node predictions and recovery decisions:\n")

    for node_id, prediction in zip(
        latest_data["node_id"], predictions
    ):
        diagnosis_result = diagnose_and_recover(
            predicted_label=prediction,
            node_id=int(node_id),
        )

        apply_recovery(diagnosis_result, node_status)

        print(
            f"Node {node_id}: "
            f"prediction={prediction}, "
            f"diagnosis={diagnosis_result['diagnosis']}, "
            f"action={diagnosis_result['recovery_action']}"
        )

    print("\nFinal node status:")
    print(node_status)

    print("\nSelected route after recovery:")
    route = select_route(nodes, gateway, node_status)
    print(route)

    print("\n=== Pipeline completed ===")


if __name__ == "__main__":
    main()