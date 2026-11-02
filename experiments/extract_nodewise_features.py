from src.feature_extraction import extract_features


def main():
    extract_features(
        input_path="data/processed/wsn_nodewise_features.csv",
        output_path="data/processed/wsn_nodewise_features_extracted.csv",
        window_size=3,
    )


if __name__ == "__main__":
    main()