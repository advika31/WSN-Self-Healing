"""Compare Random Forest with a fixed threshold baseline on identical runs.

The thresholds are chosen from the simulator's nominal measurement ranges,
not from held-out test results. This provides a simple reference point rather
than a tuned or production-ready WSN detector.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiments.multi_seed_nodewise_evaluation import (
    CLASSES,
    EXCLUDED_COLUMNS,
    choose_node_split,
)
from experiments.noisy_robustness_evaluation import NOISE_LEVELS, SEEDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "noisy_robustness"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_comparison"
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def threshold_predictions(data: pd.DataFrame) -> pd.Series:
    """Apply fixed rules to three-round rolling mean measurements.

    First identify blackholes by very low packet delivery. Among remaining
    nodes, identify faults by weak RSSI or high delay; classify the rest normal.
    """
    predictions = pd.Series("normal", index=data.index, dtype="object")
    blackhole = data["packet_delivery_ratio_mean"] < 0.20
    fault = (
        (data["rssi_mean"] < -70.0)
        | (data["delay_ms_mean"] > 65.0)
    )
    predictions.loc[blackhole] = "blackhole"
    predictions.loc[~blackhole & fault] = "fault"
    return predictions


def metric_row(y_true, y_pred, *, noise_level: str, seed: int, method: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASSES),
        average="macro",
        zero_division=0,
    )
    return {
        "noise_level": noise_level,
        "seed": seed,
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def evaluate_run(noise_level: str, seed: int) -> list[dict]:
    feature_path = DATA_DIR / noise_level / f"seed_{seed}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_path.relative_to(PROJECT_ROOT)}. "
            "Run experiments.noisy_robustness_evaluation first."
        )
    data = pd.read_csv(feature_path)
    train_nodes, test_nodes = choose_node_split(data, seed)
    train = data[data["node_id"].isin(train_nodes)]
    test = data[data["node_id"].isin(test_nodes)]
    if set(train["label"]) != set(CLASSES) or set(test["label"]) != set(CLASSES):
        raise ValueError(f"Seed {seed} ({noise_level}) is missing a class in a split.")

    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(train[feature_columns], train["label"])
    rf_predictions = model.predict(test[feature_columns])
    rule_predictions = threshold_predictions(test)

    return [
        metric_row(
            test["label"], rf_predictions,
            noise_level=noise_level, seed=seed, method="random_forest",
        ),
        metric_row(
            test["label"], rule_predictions,
            noise_level=noise_level, seed=seed, method="rule_based",
        ),
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        row
        for level in NOISE_LEVELS
        for seed in SEEDS
        for row in evaluate_run(level, seed)
    )
    summary = (
        results.groupby(["noise_level", "method"])[list(METRICS)]
        .agg(["mean", "std"])
        .sort_index()
    )
    per_run_path = OUTPUT_DIR / "per_run_metrics.csv"
    summary_path = OUTPUT_DIR / "mean_std_metrics.csv"
    results.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path)

    print("=== Rule-based baseline comparison ===")
    print(f"Seeds: {list(SEEDS)}")
    print("All methods use the same noisy data and node-wise train/test splits.")
    print("Rules: PDR mean < 0.20 => blackhole; otherwise RSSI mean < -70 dBm "
          "or delay mean > 65 ms => fault; otherwise normal.")
    print("\nPer-run metrics:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMean and sample standard deviation across seeds:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nPer-run results: {per_run_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
