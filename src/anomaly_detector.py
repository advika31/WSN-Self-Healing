import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "data/processed/wsn_features.csv"
MODEL_PATH = "results/anomaly_detector.pkl"
ENCODER_PATH = "results/label_encoder.pkl"


def train_detector():
    data = pd.read_csv(DATA_PATH)

    # These columns describe identity or ground-truth labels,
    # so they must not be used as model input.
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

    X = data[feature_columns]
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAnomaly detector training completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
        )
    )

    # Save the model and label encoder for later integration.
    import joblib

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {ENCODER_PATH}")

    return model, label_encoder


if __name__ == "__main__":
    train_detector()