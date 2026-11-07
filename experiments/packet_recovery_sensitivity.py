"""One-factor-at-a-time sensitivity study for packet-level recovery.

The node classifier uses stratified group cross-validation: every node routed in
the evaluated topology receives out-of-fold predictions from a model that did
not train on that node. See docs/recovery_sensitivity_assumptions.md.
"""
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from experiments.recovery_evaluation import ROUNDS, extract_features
from src.simulator import WSN

OUT = Path("results/recovery_sensitivity_v1")
SEEDS = tuple(range(101, 131))
NETWORK_SIZES = (20, 30, 50)
BASE_RANGE = 48.0
BASE_BLACKHOLE_P = 0.08
BASE_FAULT_P = 0.62
PACKETS_PER_SOURCE_ROUND = 10
MAX_ATTEMPTS = 3
GATEWAY = (50.0, 50.0)
GATEWAY_ID = 50  # fixed across network sizes so paired link draws stay aligned
STRATEGIES = ("no_recovery", "generic_rerouting", "diagnosis_aware")
FAULT_ROUTE_PENALTY = 1.5
E_TX_BASE = 1.0
E_TX_DISTANCE = 0.0002
E_RX = 0.5


def scenarios():
    """Ten unique runs: one shared baseline plus nine one-factor alternatives."""
    result = [dict(factor="baseline", value="baseline", num_nodes=20,
                   radio_range=BASE_RANGE, blackhole_p=BASE_BLACKHOLE_P,
                   fault_p=BASE_FAULT_P)]
    for value in (35.0, 60.0, 75.0):
        result.append(dict(factor="radio_range", value=value, num_nodes=20,
                           radio_range=value, blackhole_p=BASE_BLACKHOLE_P,
                           fault_p=BASE_FAULT_P))
    for value in (30, 50):
        result.append(dict(factor="network_size", value=value, num_nodes=value,
                           radio_range=BASE_RANGE, blackhole_p=BASE_BLACKHOLE_P,
                           fault_p=BASE_FAULT_P))
    for value in (0.05, 0.20):
        result.append(dict(factor="blackhole_forwarding_p", value=value, num_nodes=20,
                           radio_range=BASE_RANGE, blackhole_p=value,
                           fault_p=BASE_FAULT_P))
    for value in (0.40, 0.80):
        result.append(dict(factor="fault_forwarding_p", value=value, num_nodes=20,
                           radio_range=BASE_RANGE, blackhole_p=BASE_BLACKHOLE_P,
                           fault_p=value))
    for i, row in enumerate(result, start=1):
        row["scenario_id"] = f"S{i:02d}"
    return result


def assign_labels(num_nodes, seed):
    """Use a shared 20-node core and extend labels reproducibly to larger networks."""
    rng = np.random.default_rng(seed + 7_000_033)
    core_order = rng.permutation(20)
    labels = {i: "normal" for i in range(50)}
    for node in core_order[:4]:
        labels[int(node)] = "fault"
    for node in core_order[4:8]:
        labels[int(node)] = "blackhole"
    for start in range(20, 50, 5):
        block = ["fault", "blackhole", "normal", "normal", "normal"]
        rng.shuffle(block)
        for offset, label in enumerate(block):
            labels[start + offset] = label
    return {node: labels[node] for node in range(num_nodes)}


def out_of_fold_predictions(data, seed):
    """Predict each node only with a model trained on other nodes."""
    excluded = {"node_id", "x", "y", "round", "label"}
    feature_cols = [c for c in data.columns if c not in excluded]
    nodes = data[["node_id", "label"]].drop_duplicates().sort_values("node_id")
    y = nodes.label.to_numpy()
    min_class = int(nodes.label.value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError("Need at least two nodes per behavior for grouped CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = {}
    node_ids = nodes.node_id.to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(node_ids, y)):
        train_nodes = set(node_ids[train_idx])
        test_nodes = set(node_ids[test_idx])
        train = data[data.node_id.isin(train_nodes)]
        test = data[data.node_id.isin(test_nodes)]
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=seed + fold)
        model.fit(train[feature_cols], train.label)
        pred = model.predict(test[feature_cols])
        for node, rnd, label in zip(test.node_id, test["round"], pred):
            predictions[(int(node), int(rnd))] = str(label)
    if len(predictions) != len(data):
        raise RuntimeError("Grouped out-of-fold predictions are incomplete")
    return predictions


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def make_graph(network, radio_range):
    positions = {n.node_id: (float(n.x), float(n.y)) for n in network.nodes}
    positions[GATEWAY_ID] = GATEWAY
    graph = defaultdict(list)
    nodes = list(positions)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            d = distance(positions[u], positions[v])
            if d <= radio_range:
                graph[u].append((v, d))
                graph[v].append((u, d))
    return positions, graph


def edge_cost(v, d, predictions, strategy):
    if v == GATEWAY_ID:
        return d
    label = predictions.get(v, "normal")
    if strategy == "generic_rerouting" and label != "normal":
        return np.inf
    if strategy == "diagnosis_aware":
        if label == "blackhole":
            return np.inf
        if label == "fault":
            return d * FAULT_ROUTE_PENALTY
    return d


def shortest_path(source, graph, predictions, strategy, blocked=(), visited=()):
    blocked = {tuple(edge) for edge in blocked}
    visited = set(visited)
    dist = {source: 0.0}
    prev = {}
    queue = [(0.0, source)]
    while queue:
        cost, u = heappop(queue)
        if cost != dist.get(u):
            continue
        if u == GATEWAY_ID:
            break
        for v, d in graph.get(u, ()):
            if (u, v) in blocked or (v in visited and v != GATEWAY_ID):
                continue
            candidate = cost + edge_cost(v, d, predictions, strategy)
            if np.isfinite(candidate) and candidate < dist.get(v, np.inf):
                dist[v] = candidate
                prev[v] = u
                heappush(queue, (candidate, v))
    if GATEWAY_ID not in dist:
        return None
    path = [GATEWAY_ID]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return list(reversed(path))


def link_uniforms(seed, rnd, source, packet, u, v):
    # Tuple-keyed draws remain the same across scenarios and strategy traversal order.
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, rnd, source, packet, u, v, 381_721]))
    return rng.random(MAX_ATTEMPTS)


def hop_success_p(d, receiver, labels, config):
    radio = max(0.55, 0.99 - 0.35 * (d / config["radio_range"]) ** 2)
    if receiver == GATEWAY_ID:
        forward = 1.0
    elif labels[receiver] == "blackhole":
        forward = config["blackhole_p"]
    elif labels[receiver] == "fault":
        forward = config["fault_p"]
    else:
        forward = 0.98
    return radio * forward


def simulate_packet(source, strategy, initial, graph, positions, predictions,
                    labels, config, seed, rnd, packet, trial_cache):
    if initial is None:
        return dict(delivered=False, delay_ms=0.0, energy_units=0.0, hops=0,
                    retransmissions=0, route_changes=0, fallback=False)
    current = source
    visited = {source}
    path = initial[:]
    idx = 0
    blocked = set()
    delay = energy = 0.0
    hops = retransmissions = changes = 0
    fallback = False
    while current != GATEWAY_ID:
        nxt = path[idx + 1]
        d = distance(positions[current], positions[nxt])
        p = hop_success_p(d, nxt, labels, config)
        edge = (current, nxt)
        if edge not in trial_cache:
            trial_cache[edge] = link_uniforms(seed, rnd, source, packet, current, nxt)
        uniforms = trial_cache[edge]
        ok = False
        for value in uniforms:
            energy += E_TX_BASE + E_TX_DISTANCE * d * d + E_RX
            delay += 2.0 + d / config["radio_range"] * 8.0
            if value < p:
                ok = True
                break
            retransmissions += 1
        hops += 1
        if ok:
            current = nxt
            visited.add(current)
            idx += 1
            continue
        blocked.add((current, nxt))
        if strategy == "no_recovery":
            break
        alt = shortest_path(current, graph, predictions, strategy,
                            blocked, visited - {current})
        if alt is None:
            alt = shortest_path(current, graph,
                {n: "normal" for n in predictions}, "no_recovery",
                blocked, visited - {current})
            fallback = alt is not None
        if alt is None:
            break
        changes += 1
        path = alt
        idx = 0
    return dict(delivered=current == GATEWAY_ID, delay_ms=delay,
                energy_units=energy, hops=hops, retransmissions=retransmissions,
                route_changes=changes, fallback=fallback)


def prepare_state(num_nodes, seed):
    """Generate telemetry and out-of-fold classifier decisions once per size/seed."""
    labels = assign_labels(num_nodes, seed)
    fault_nodes = [n for n, label in labels.items() if label == "fault"]
    blackhole_nodes = [n for n, label in labels.items() if label == "blackhole"]
    network = WSN(num_nodes=num_nodes, seed=seed, fault_nodes=fault_nodes,
                  blackhole_nodes=blackhole_nodes, challenging_mode=True)
    data = extract_features(network.simulate(rounds=ROUNDS))
    predictions = out_of_fold_predictions(data, seed)
    positions = {n.node_id: (float(n.x), float(n.y)) for n in network.nodes}
    positions[GATEWAY_ID] = GATEWAY
    return labels, network, predictions, positions


def run_scenario_seed(config, seed, state=None):
    if state is None:
        state = prepare_state(config["num_nodes"], seed)
    labels, network, predictions, positions = state
    _, graph = make_graph(network, config["radio_range"])
    all_nodes = list(range(config["num_nodes"]))
    rows = []
    normal_predictions = {n: "normal" for n in all_nodes}
    for rnd in range(1, ROUNDS + 1):
        round_pred = {n: predictions[(n, rnd)] for n in all_nodes}
        for source in all_nodes:
            physical = shortest_path(source, graph, normal_predictions, "no_recovery") is not None
            routes = {}
            initial_fallback = {}
            for strategy in STRATEGIES:
                routes[strategy] = shortest_path(source, graph, round_pred, strategy)
                initial_fallback[strategy] = False
                if routes[strategy] is None:
                    routes[strategy] = shortest_path(source, graph, normal_predictions, "no_recovery")
                    initial_fallback[strategy] = routes[strategy] is not None
            for packet in range(PACKETS_PER_SOURCE_ROUND):
                trial_cache = {}
                for strategy in STRATEGIES:
                    outcome = simulate_packet(source, strategy, routes[strategy],
                        graph, positions, round_pred, labels, config,
                        seed, rnd, packet, trial_cache)
                    outcome["fallback"] = outcome["fallback"] or initial_fallback[strategy]
                    rows.append(dict(seed=seed, strategy=strategy,
                        physical_route_available=physical, **outcome))
    detail = pd.DataFrame(rows)
    summary = detail.groupby("strategy").agg(
        packet_delivery_ratio=("delivered", "mean"),
        physical_route_availability=("physical_route_available", "mean"),
        delay_ms_per_offered_packet=("delay_ms", "mean"),
        energy_units_per_offered_packet=("energy_units", "mean"),
        hops_per_offered_packet=("hops", "mean"),
        retransmissions_per_offered_packet=("retransmissions", "mean"),
        route_changes_per_offered_packet=("route_changes", "mean"),
        emergency_fallback_fraction=("fallback", "mean"),
    ).reset_index()
    for key, value in config.items():
        summary[key] = value
    summary["seed"] = seed
    summary["scenario_id"] = config["scenario_id"]
    return summary


def summarize(per_seed):
    metrics = ["packet_delivery_ratio", "physical_route_availability",
               "delay_ms_per_offered_packet", "energy_units_per_offered_packet",
               "hops_per_offered_packet", "retransmissions_per_offered_packet",
               "route_changes_per_offered_packet", "emergency_fallback_fraction"]
    keys = ["scenario_id", "factor", "value", "num_nodes", "radio_range", "blackhole_p", "fault_p"]
    summary = per_seed.groupby(keys + ["strategy"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col
                       for col in summary.columns]
    paired = []
    for scenario_key, group in per_seed.groupby(keys):
        by_seed = group.pivot(index="seed", columns="strategy", values="packet_delivery_ratio")
        for method, reference in (("generic_rerouting", "no_recovery"),
                                  ("diagnosis_aware", "no_recovery"),
                                  ("diagnosis_aware", "generic_rerouting")):
            delta = (by_seed[method] - by_seed[reference]).to_numpy()
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1))
            se = sd / np.sqrt(len(delta))
            crit = float(student_t.ppf(0.975, df=len(delta) - 1))
            row = dict(zip(keys, scenario_key))
            row.update(comparison=f"{method} minus {reference}",
                       paired_pdr_delta=mean, paired_delta_sd=sd,
                       ci95_low=mean - crit * se, ci95_high=mean + crit * se,
                       seeds=len(delta))
            paired.append(row)
    baseline = per_seed[per_seed.scenario_id == "S01"].set_index(["seed", "strategy"])
    base_differences = []
    for scenario_id, group in per_seed[per_seed.scenario_id != "S01"].groupby("scenario_id"):
        for strategy in STRATEGIES:
            current = group[group.strategy == strategy].set_index("seed")
            reference = baseline.xs(strategy, level="strategy")
            for metric in ("packet_delivery_ratio", "physical_route_availability",
                           "delay_ms_per_offered_packet", "energy_units_per_offered_packet"):
                delta = (current[metric] - reference[metric]).dropna().to_numpy()
                mean = float(delta.mean())
                sd = float(delta.std(ddof=1))
                crit = float(student_t.ppf(0.975, df=len(delta) - 1))
                se = sd / np.sqrt(len(delta))
                row = group[group.strategy == strategy].iloc[0]
                base_differences.append(dict(
                    scenario_id=scenario_id, factor=row.factor, value=row.value,
                    strategy=strategy, metric=metric, paired_delta_mean=mean,
                    paired_delta_sd=sd, ci95_low=mean - crit * se,
                    ci95_high=mean + crit * se, seeds=len(delta)))
    return summary, pd.DataFrame(paired), pd.DataFrame(base_differences)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config_rows = scenarios()
    metrics_path = OUT / "per_seed_metrics.csv"
    if metrics_path.exists():
        per_seed = pd.read_csv(metrics_path)
    else:
        per_seed = pd.DataFrame()
    state_cache = {}
    for i, config in enumerate(config_rows, start=1):
        print(f"Scenario {i}/{len(config_rows)}: {config['factor']}={config['value']}", flush=True)
        for seed in SEEDS:
            done = (not per_seed.empty and
                    ((per_seed.scenario_id == config["scenario_id"]) &
                     (per_seed.seed == seed)).sum() == len(STRATEGIES))
            if done:
                continue
            print(f"  seed {seed}", flush=True)
            state_key = (config["num_nodes"], seed)
            if state_key not in state_cache:
                state_cache[state_key] = prepare_state(*state_key)
            result = run_scenario_seed(config, seed, state_cache[state_key])
            per_seed = pd.concat([per_seed, result], ignore_index=True)
            per_seed.to_csv(metrics_path, index=False)
    summary, paired, baseline_differences = summarize(per_seed)
    summary.to_csv(OUT / "sensitivity_summary.csv", index=False)
    paired.to_csv(OUT / "paired_pdr_differences.csv", index=False)
    baseline_differences.to_csv(OUT / "scenario_vs_baseline.csv", index=False)
    print("\nMean PDR by scenario and strategy:")
    print(summary[["factor", "value", "strategy", "packet_delivery_ratio_mean",
                   "packet_delivery_ratio_std"]].round(4).to_string(index=False))
    print(f"\nSaved {len(per_seed)} seed-policy rows to {OUT}")


if __name__ == "__main__":
    main()
