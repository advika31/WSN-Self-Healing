import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/processed/wsn_nodewise_features_extracted.csv"

def main():
    data = pd.read_csv(DATA_PATH)

    # Keep each node's readings together.
    node_ids = sorted(data["node_id"].unique())

    if len(node_ids) < 4:
        raise ValueError(
            "At least 4 distinct nodes are needed for this evaluation."
        )

    # Use the first 70% of nodes for training and the rest for testing.
    split_index = int(len(node_ids) * 0.7)

    train_nodes = list(range(14))
    test_nodes = list(range(14, 20))

    train_data = data[data["node_id"].isin(train_nodes)].copy()
    test_data = data[data["node_id"].isin(test_nodes)].copy()

    excluded_columns = {
        "node_id",
        "x",
        "y",
        "round",
        "label",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["label"]
    y_test = test_data["label"]

    print("\n=== Node-wise Evaluation ===")
    print(f"Training node IDs: {train_nodes}")
    print(f"Testing node IDs: {test_nodes}")
    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Ensure every test class exists in the training set.
    unseen_test_classes = set(y_test.unique()) - set(y_train.unique())

    if unseen_test_classes:
        raise ValueError(
            "Some test classes are absent from training: "
            f"{sorted(unseen_test_classes)}. "
            "Expand or revise the simulation split."
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train_encoded)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test_encoded,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test_encoded,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
        )
    )


if __name__ == "__main__":
    main()
    