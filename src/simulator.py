import numpy as np
import pandas as pd


class SensorNode:
    def __init__(self, node_id, x, y, initial_energy=100.0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.initial_energy = initial_energy
        self.residual_energy = initial_energy

    def generate_reading(self, behavior="normal"):
        """Generate network metrics for a normal, faulty, or malicious node."""

        if behavior == "normal":
            rssi = np.random.normal(-60, 5)
            packet_delivery_ratio = np.random.uniform(0.94, 1.0)
            delay = np.random.normal(30, 5)
            energy_consumed = np.random.uniform(0.1, 0.5)

        elif behavior == "fault":
            # A faulty node has weak communication and irregular delays.
            rssi = np.random.normal(-82, 6)
            packet_delivery_ratio = np.random.uniform(0.35, 0.70)
            delay = np.random.normal(100, 25)
            energy_consumed = np.random.uniform(0.2, 0.7)

        elif behavior == "blackhole":
            # A blackhole node receives packets but drops most of them.
            rssi = np.random.normal(-60, 5)
            packet_delivery_ratio = np.random.uniform(0.0, 0.10)
            delay = np.random.normal(35, 8)
            energy_consumed = np.random.uniform(0.1, 0.5)

        else:
            raise ValueError(f"Unknown behavior: {behavior}")

        delay = max(delay, 1)
        packet_delivery_ratio = np.clip(packet_delivery_ratio, 0, 1)
        packet_loss = 1 - packet_delivery_ratio

        self.residual_energy = max(
            0, self.residual_energy - energy_consumed
        )

        return {
            "node_id": self.node_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "rssi": round(rssi, 2),
            "packet_delivery_ratio": round(packet_delivery_ratio, 4),
            "packet_loss": round(packet_loss, 4),
            "delay_ms": round(delay, 2),
            "residual_energy": round(self.residual_energy, 2),
            "energy_consumed": round(energy_consumed, 4),
            "label": behavior,
        }


class WSN:
    def __init__(self, num_nodes=10, seed=42):
        np.random.seed(seed)

        self.num_nodes = num_nodes
        self.gateway = (50, 50)
        self.nodes = []

        for node_id in range(num_nodes):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)

            self.nodes.append(
                SensorNode(node_id=node_id, x=x, y=y)
            )

        # Fixed node assignments make experiments reproducible.
        self.behavior_by_node = {}

        for node in self.nodes:
            self.behavior_by_node[node.node_id] = "normal"

        if num_nodes >= 3:
            self.behavior_by_node[1] = "fault"
            self.behavior_by_node[2] = "blackhole"

    def simulate(self, rounds=10):
        records = []

        for round_number in range(1, rounds + 1):
            for node in self.nodes:
                behavior = self.behavior_by_node[node.node_id]

                reading = node.generate_reading(behavior=behavior)
                reading["round"] = round_number

                records.append(reading)

        return pd.DataFrame(records)


if __name__ == "__main__":
    network = WSN(num_nodes=10, seed=42)
    data = network.simulate(rounds=10)

    print("\nWSN Simulation with Faults and Attacks Completed!")
    print(f"Total sensor nodes: {network.num_nodes}")
    print(f"Total records: {len(data)}")

    print("\nRecords by behavior:")
    print(data["label"].value_counts())

    print("\nAverage metrics by behavior:")
    print(
        data.groupby("label")[
            [
                "rssi",
                "packet_delivery_ratio",
                "packet_loss",
                "delay_ms",
                "residual_energy",
            ]
        ].mean().round(3)
    )

    output_path = "data/raw/wsn_scenario_data.csv"
    data.to_csv(output_path, index=False)

    print(f"\nDataset saved to {output_path}")