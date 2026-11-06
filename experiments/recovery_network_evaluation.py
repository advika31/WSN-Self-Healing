"""Topology-aware, proxy recovery experiment; see docs/recovery_v2_assumptions.md."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.recovery_evaluation import (
    CLASSES, NUM_NODES, NUM_SEEDS, ROUNDS, TEST_NODES,
    assign_behaviors, extract_features, train_and_predict,
)
from src.simulator import WSN

OUT = Path("results/recovery_v2")
GATEWAY = np.array([50.0, 50.0])
RADIO_RANGE = 55.0
# Abstract normalized radio cost per packet: electronics + amplifier*d^2.
E_ELEC = 1.0
E_AMP = 0.0002
FAULT_COST = 1.5
STRATEGIES = ("no_recovery", "generic_rerouting", "diagnosis_aware")


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def route_options(source, candidates, gateway=GATEWAY, radio_range=RADIO_RANGE):
    """Enumerate direct and one-relay routes; topology shared by all policies."""
    options = []
    direct_d = distance((source.x, source.y), gateway)
    if direct_d <= radio_range:
        options.append({"relay": None, "hops": [(None, direct_d)], "distance": direct_d})
    for relay in candidates:
        if relay.node_id == source.node_id:
            continue
        d1 = distance((source.x, source.y), (relay.x, relay.y))
        d2 = distance((relay.x, relay.y), gateway)
        if max(d1, d2) <= radio_range:
            options.append({"relay": relay, "hops": [(source.node_id, d1), (relay.node_id, d2)], "distance": d1 + d2})
    return options


def select_route(source, options, row_by_node, strategy):
    eligible = []
    for route in options:
        relay = route["relay"]
        label = "normal" if relay is None else row_by_node[relay.node_id]["predicted_label"]
        if strategy == "generic_rerouting" and label != "normal":
            continue
        if strategy == "diagnosis_aware" and label == "blackhole":
            continue
        penalty = FAULT_COST if strategy == "diagnosis_aware" and label == "fault" else 1.0
        # Compare all strategies on one candidate set and one topology.
        eligible.append((route["distance"] * penalty, route))
    fallback = not eligible
    if fallback:
        eligible = [(r["distance"], r) for r in options]
    return (min(eligible, key=lambda item: (item[0], -1 if item[1]["relay"] is None else item[1]["relay"].node_id))[1], fallback)


def packet_outcome(source, route, row_by_node):
    """Use telemetry PDR as per-node forwarding reliability proxy."""
    relay = route["relay"]
    p_source = float(row_by_node[source.node_id]["packet_delivery_ratio"])
    p_relay = 1.0 if relay is None else float(row_by_node[relay.node_id]["packet_delivery_ratio"])
    pdr = p_source * p_relay
    delay = float(row_by_node[source.node_id]["delay_ms"])
    if relay is not None:
        delay += float(row_by_node[relay.node_id]["delay_ms"])
    energy = 0.0
    for transmitter, hop_distance in route["hops"]:
        energy += E_ELEC + E_AMP * hop_distance**2
    if relay is not None:
        energy += E_ELEC  # receiver electronics, abstract units
    return pdr, delay, energy


def run_seed(seed):
    faults, blackholes = assign_behaviors(seed)
    net = WSN(num_nodes=NUM_NODES, seed=seed, fault_nodes=faults,
              blackhole_nodes=blackholes, challenging_mode=True)
    frame = extract_features(net.simulate(rounds=ROUNDS))
    predicted = train_and_predict(frame, seed)
    predicted = predicted[predicted.node_id.isin(TEST_NODES)]
    rows = []
    for rnd in range(1, ROUNDS + 1):
        data = predicted[predicted["round"] == rnd].set_index("node_id")
        row_by_node = data.to_dict(orient="index")
        candidates = [net.nodes[n] for n in TEST_NODES]
        for source_id in TEST_NODES:
            source = net.nodes[source_id]
            options = route_options(source, candidates)
            for policy in STRATEGIES:
                if options:
                    route, fallback = select_route(source, options, row_by_node, policy)
                    pdr, delay, energy = packet_outcome(source, route, row_by_node)
                    relay = route["relay"]
                    actual_bad = relay is not None and row_by_node[relay.node_id]["label"] != "normal"
                    blackhole = relay is not None and row_by_node[relay.node_id]["label"] == "blackhole"
                    rows.append(dict(seed=seed, round=rnd, source=source_id, strategy=policy,
                        delivered_probability=pdr, delay_ms=delay, energy_units=energy,
                        route_found=True, hops=len(route["hops"]), relay_node=-1 if relay is None else relay.node_id,
                        relay_predicted="direct" if relay is None else row_by_node[relay.node_id]["predicted_label"],
                        relay_actual="direct" if relay is None else row_by_node[relay.node_id]["label"],
                        selected_unhealthy=actual_bad, selected_blackhole=blackhole,
                        fallback_used=fallback, route_distance=route["distance"]))
                else:
                    rows.append(dict(seed=seed, round=rnd, source=source_id, strategy=policy,
                        delivered_probability=0.0, delay_ms=np.nan, energy_units=0.0,
                        route_found=False, hops=0, relay_node=-1, relay_predicted="none",
                        relay_actual="none", selected_unhealthy=False, selected_blackhole=False,
                        fallback_used=False, route_distance=np.nan))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [r for seed in range(1, NUM_SEEDS + 1) for r in run_seed(seed)]
    detail = pd.DataFrame(records)
    detail.to_csv(OUT / "round_source_results.csv", index=False)
    per_seed = detail.groupby(["seed", "strategy"]).agg(
        packet_delivery_ratio=("delivered_probability", "mean"),
        route_availability=("route_found", "mean"),
        delay_ms=("delay_ms", "mean"), energy_units=("energy_units", "sum"),
        mean_hops=("hops", "mean"), unhealthy_relay_fraction=("selected_unhealthy", "mean"),
        blackhole_relay_fraction=("selected_blackhole", "mean"),
        fallback_routes=("fallback_used", "sum"),
    ).reset_index()
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary = per_seed.groupby("strategy").agg(["mean", "std"])
    summary.to_csv(OUT / "mean_std_metrics.csv")
    print(summary.round(4))
    print(f"\nWrote new run to {OUT}")


if __name__ == "__main__":
    main()
