import math


def calculate_distance(node_a, node_b):
    """Calculate Euclidean distance between two (x, y) points."""
    return math.sqrt(
        (node_a[0] - node_b[0]) ** 2
        + (node_a[1] - node_b[1]) ** 2
    )


def select_route(nodes, gateway, node_status):
    """
    Select an available node with the lowest estimated routing cost.

    nodes: dictionary mapping node_id to (x, y)
    gateway: (x, y) coordinate
    node_status: dictionary mapping node_id to status
    """

    candidates = []

    for node_id, position in nodes.items():
        status = node_status.get(node_id, "active")

        if status not in ("active", "unreliable"):
            continue

        distance = calculate_distance(position, gateway)

        # Unreliable nodes remain usable, but receive a penalty.
        penalty = 1.5 if status == "unreliable" else 1.0
        cost = distance * penalty

        candidates.append((cost, node_id, distance, status))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    best_cost, best_node, distance, status = candidates[0]

    return {
        "selected_node": best_node,
        "distance_to_gateway": round(distance, 2),
        "routing_cost": round(best_cost, 2),
        "status": status,
    }


def apply_recovery(diagnosis_result, node_status):
    """Update the network's status table using a diagnosis result."""

    node_id = diagnosis_result["node_id"]
    new_status = diagnosis_result["node_status"]

    node_status[node_id] = new_status

    return node_status


if __name__ == "__main__":
    gateway = (50, 50)

    nodes = {
        0: (45, 48),
        1: (52, 53),
        2: (70, 80),
        3: (30, 40),
    }

    node_status = {
        0: "active",
        1: "active",
        2: "active",
        3: "active",
    }

    print("\nInitial route:")
    print(select_route(nodes, gateway, node_status))

    # Simulate a blackhole diagnosis for node 1.
    diagnosis_result = {
        "node_id": 1,
        "node_status": "isolated",
    }

    apply_recovery(diagnosis_result, node_status)

    print("\nRoute after isolating node 1:")
    print(select_route(nodes, gateway, node_status))

    print("\nUpdated node status:")
    print(node_status)