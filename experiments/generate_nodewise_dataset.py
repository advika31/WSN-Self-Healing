from pathlib import Path

from src.simulator import WSN


def main():
    # Behaviors are distributed across both training and test node IDs.
    fault_nodes = [1, 5, 14, 18]
    blackhole_nodes = [2, 8, 15, 19]

    network = WSN(
        num_nodes=20,
        seed=123,
        fault_nodes=fault_nodes,
        blackhole_nodes=blackhole_nodes,
    )

    data = network.simulate(rounds=30)

    output_path = Path("data/processed/wsn_nodewise_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path, index=False)

    print("Node-wise dataset generated.")
    print(f"Total records: {len(data)}")
    print("\nBehavior distribution:")
    print(data["label"].value_counts())

    print("\nBehavior distribution by node:")
    print(data.groupby(["node_id", "label"]).size())

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()