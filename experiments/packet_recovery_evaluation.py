"""Packet-forwarding recovery comparison with paired link-trial draws.

See docs/packet_recovery_assumptions.md before interpreting these synthetic results.
"""
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.recovery_evaluation import (
    NUM_NODES, NUM_SEEDS, ROUNDS, TEST_NODES, assign_behaviors,
    extract_features, train_and_predict,
)
from src.simulator import WSN

OUT = Path("results/recovery_v3_packet")
GATEWAY_ID = NUM_NODES
GATEWAY = (50.0, 50.0)
RADIO_RANGE = 48.0
PACKETS_PER_SOURCE_ROUND = 20
MAX_RETRIES_PER_HOP = 3
STRATEGIES = ("no_recovery", "generic_rerouting", "diagnosis_aware")
FAULT_ROUTE_PENALTY = 1.5
E_TX_BASE = 1.0
E_TX_DISTANCE = 0.0002
E_RX = 0.5


def euclidean(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def make_graph(network, node_ids=TEST_NODES):
    """Undirected disk graph; links are independently lossy by direction at runtime."""
    positions = {node_id: (float(network.nodes[node_id].x),
                           float(network.nodes[node_id].y))
                 for node_id in node_ids}
    positions[GATEWAY_ID] = GATEWAY
    adjacency = defaultdict(list)
    for u in positions:
        for v in positions:
            if u >= v:
                continue
            d = euclidean(positions[u], positions[v])
            if d <= RADIO_RANGE:
                adjacency[u].append((v, d))
                adjacency[v].append((u, d))
    return positions, adjacency


def edge_success_probability(distance, receiver, labels):
    """Synthetic link and forwarding model; values are assumptions, not measurements."""
    radio_success = max(0.55, 0.99 - 0.35 * (distance / RADIO_RANGE) ** 2)
    forwarding = {"normal": 0.98, "fault": 0.62, "blackhole": 0.08}
    node_acceptance = 1.0 if receiver == GATEWAY_ID else forwarding[labels[receiver]]
    return radio_success * node_acceptance


def route_cost(node, neighbor, distance, predictions, strategy):
    cost = distance
    if neighbor == GATEWAY_ID:
        return cost
    label = predictions.get(neighbor, "normal")
    if strategy == "generic_rerouting" and label != "normal":
        return np.inf
    if strategy == "diagnosis_aware":
        if label == "blackhole":
            return np.inf
        if label == "fault":
            cost *= FAULT_ROUTE_PENALTY
    return cost


def shortest_path(source, adjacency, predictions, strategy, blocked_edges=(), visited=()):
    """Dijkstra route under policy constraints and this packet's failed edges."""
    blocked = {frozenset(edge) for edge in blocked_edges}
    visited = set(visited)
    distances = {source: 0.0}
    previous = {}
    queue = [(0.0, source)]
    while queue:
        cost, u = heappop(queue)
        if cost != distances.get(u):
            continue
        if u == GATEWAY_ID:
            break
        for v, d in adjacency.get(u, []):
            if frozenset((u, v)) in blocked:
                continue
            if v in visited and v != GATEWAY_ID:
                continue
            weight = route_cost(u, v, d, predictions, strategy)
            candidate = cost + weight
            if np.isfinite(candidate) and candidate < distances.get(v, np.inf):
                distances[v] = candidate
                previous[v] = u
                heappush(queue, (candidate, v))
    if GATEWAY_ID not in distances:
        return None
    path = [GATEWAY_ID]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return list(reversed(path))


def draw_link_trials(seed, round_number, source, packet_index, positions):
    """Pre-draw shared uniforms for every directed edge and retry."""
    rng = np.random.default_rng(seed * 10_000_019 + round_number * 10_007 + source * 101 + packet_index)
    trials = {}
    node_ids = [*TEST_NODES, GATEWAY_ID]
    for u in node_ids:
        for v in node_ids:
            if u == v:
                continue
            d = euclidean(positions[u], positions[v])
            if d <= RADIO_RANGE:
                trials[(u, v)] = rng.random(MAX_RETRIES_PER_HOP)
    return trials


def simulate_packet(source, strategy, initial_path, adjacency, positions,
                    predictions, labels, shared_trials):
    if initial_path is None:
        return dict(delivered=False, delay_ms=0.0, energy_units=0.0,
                    hops=0, retransmissions=0, route_changes=0,
                    fallback=False, path="unreachable")
    current = source
    visited = {source}
    trace = [source]
    blocked = set()
    path = initial_path[:]
    index = 0
    delay = energy = 0.0
    hops = retransmissions = route_changes = 0
    fallback = False
    while current != GATEWAY_ID:
        if index + 1 >= len(path) or path[index] != current:
            alternative = shortest_path(current, adjacency, predictions,
                                        strategy, blocked, visited - {current})
            if alternative is None and strategy != "no_recovery":
                # Explicit emergency fallback when diagnosis excludes every route.
                alternative = shortest_path(current, adjacency,
                    {n: "normal" for n in predictions}, "no_recovery", blocked,
                    visited - {current})
                fallback = alternative is not None
            if alternative is None:
                break
            if path != alternative:
                route_changes += 1
            path = alternative
            index = 0
        nxt = path[index + 1]
        d = euclidean(positions[current], positions[nxt])
        success_p = edge_success_probability(d, nxt, labels)
        delivered_hop = False
        for attempt, u in enumerate(shared_trials.get((current, nxt), ())):
            energy += E_TX_BASE + E_TX_DISTANCE * d**2 + E_RX
            delay += 2.0 + d / RADIO_RANGE * 8.0
            if u < success_p:
                delivered_hop = True
                break
            retransmissions += 1
        hops += 1
        if delivered_hop:
            current = nxt
            visited.add(current)
            trace.append(current)
            index += 1
            continue
        blocked.add(frozenset((current, nxt)))
        if strategy == "no_recovery":
            break
        alternative = shortest_path(current, adjacency, predictions,
                                    strategy, blocked, visited - {current})
        if alternative is None:
            # If constraints strand this packet, make one logged emergency attempt.
            alternative = shortest_path(current, adjacency,
                {n: "normal" for n in predictions}, "no_recovery", blocked,
                visited - {current})
            fallback = alternative is not None
        if alternative is None:
            break
        route_changes += 1
        path = alternative
        index = 0
    return dict(delivered=current == GATEWAY_ID, delay_ms=delay,
                energy_units=energy, hops=hops, retransmissions=retransmissions,
                route_changes=route_changes, fallback=fallback,
                path="-".join(map(str, trace)))


def prepare_seed(seed):
    faults, blackholes = assign_behaviors(seed)
    network = WSN(num_nodes=NUM_NODES, seed=seed, fault_nodes=faults,
                  blackhole_nodes=blackholes, challenging_mode=True)
    data = extract_features(network.simulate(rounds=ROUNDS))
    heldout = train_and_predict(data, seed)
    heldout = heldout[heldout.node_id.isin(TEST_NODES)]
    labels = network.behavior_by_node
    positions, adjacency = make_graph(network)
    return network, heldout, labels, positions, adjacency


def run_seed(seed):
    network, predictions_frame, labels, positions, adjacency = prepare_seed(seed)
    records = []
    for rnd in range(1, ROUNDS + 1):
        round_data = predictions_frame[predictions_frame["round"] == rnd]
        predictions = dict(zip(round_data.node_id, round_data.predicted_label))
        for source in TEST_NODES:
            physical_path_exists = shortest_path(
                source, adjacency, {n: "normal" for n in predictions}, "no_recovery") is not None
            initial_paths = {}
            initial_fallbacks = {}
            for strategy in STRATEGIES:
                initial_paths[strategy] = shortest_path(
                    source, adjacency, predictions, strategy)
                initial_fallbacks[strategy] = False
                if initial_paths[strategy] is None:
                    initial_paths[strategy] = shortest_path(
                        source, adjacency,
                        {n: "normal" for n in predictions}, "no_recovery")
                    initial_fallbacks[strategy] = initial_paths[strategy] is not None
            for packet in range(PACKETS_PER_SOURCE_ROUND):
                shared_trials = draw_link_trials(seed, rnd, source, packet, positions)
                for strategy in STRATEGIES:
                    path = initial_paths[strategy]
                    fallback = initial_fallbacks[strategy]
                    outcome = simulate_packet(source, strategy, path, adjacency,
                        positions, predictions, labels, shared_trials)
                    outcome["fallback"] = outcome["fallback"] or fallback
                    records.append(dict(seed=seed, round=rnd, source=source,
                        packet=packet, strategy=strategy,
                        physical_route_available=physical_path_exists,
                        predicted_source=predictions[source],
                        actual_source=labels[source], **outcome))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([r for seed in range(1, NUM_SEEDS + 1)
                            for r in run_seed(seed)])
    results.to_csv(OUT / "packet_results.csv", index=False)
    per_seed = results.groupby(["seed", "strategy"]).agg(
        packet_delivery_ratio=("delivered", "mean"),
        physical_route_availability=("physical_route_available", "mean"),
        mean_delay_ms=("delay_ms", "mean"),
        energy_units_per_offered_packet=("energy_units", "mean"),
        mean_hops_per_offered_packet=("hops", "mean"),
        retransmissions_per_offered_packet=("retransmissions", "mean"),
        route_changes_per_offered_packet=("route_changes", "mean"),
        emergency_fallback_fraction=("fallback", "mean"),
    ).reset_index()
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    per_seed.groupby("strategy").agg(["mean", "std"]).to_csv(
        OUT / "mean_std_metrics.csv")
    print(per_seed.groupby("strategy").mean(numeric_only=True).round(4))
    print(f"\nSaved packet-level results to {OUT}")


if __name__ == "__main__":
    main()
