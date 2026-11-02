import pandas as pd


def extract_features(input_path, output_path, window_size=3):
    """
    Create rolling features for each sensor node.

    The rolling window uses the current reading and the previous
    window_size - 1 readings for that same node.
    """

    data = pd.read_csv(input_path)

    required_columns = {
        "node_id",
        "round",
        "rssi",
        "packet_delivery_ratio",
        "packet_loss",
        "delay_ms",
        "residual_energy",
        "energy_consumed",
        "label",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing columns: {sorted(missing_columns)}"
        )

    data = data.sort_values(["node_id", "round"]).copy()

    feature_columns = [
        "rssi",
        "packet_delivery_ratio",
        "packet_loss",
        "delay_ms",
        "residual_energy",
        "energy_consumed",
    ]

    grouped = data.groupby("node_id", group_keys=False)

    for column in feature_columns:
        data[f"{column}_mean"] = grouped[column].transform(
            lambda values: values.rolling(
                window=window_size, min_periods=1
            ).mean()
        )

        data[f"{column}_std"] = grouped[column].transform(
            lambda values: values.rolling(
                window=window_size, min_periods=1
            ).std()
        )

    # A standard deviation is undefined for the first reading.
    # Replace it with zero because there is no variation to estimate yet.
    std_columns = [
        f"{column}_std" for column in feature_columns
    ]
    data[std_columns] = data[std_columns].fillna(0)

    data.to_csv(output_path, index=False)

    print("\nFeature extraction completed!")
    print(f"Input records: {len(data)}")
    print(f"Rolling window size: {window_size}")
    print(f"Total columns after feature extraction: {len(data.columns)}")
    print(f"Dataset saved to: {output_path}")

    print("\nSample extracted features:")
    print(
        data[
            [
                "node_id",
                "round",
                "rssi_mean",
                "packet_delivery_ratio_mean",
                "delay_ms_mean",
                "label",
            ]
        ].head(10)
    )

    return data


if __name__ == "__main__":
    extract_features(
        input_path="data/raw/wsn_scenario_data.csv",
        output_path="data/processed/wsn_features.csv",
        window_size=3,
    )