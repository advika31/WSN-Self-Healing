"""Battery-calibrated paired recovery study with transient faults and fallback counts.

This separate version preserves v2 results. It uses a healthy-network control to
set an abstract battery budget that does not itself break the no-fault service.
"""
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from experiments.recovery_evaluation import extract_features
from src.simulator import WSN

OUT = Path("results/recovery_policy_v3")
NODE_COUNT = 20
GATEWAY = NODE_COUNT
SOURCES = tuple(range(10))
RELAYS = tuple(range(10, 20))
BRIDGE = 16
SEEDS = tuple(range(201, 231))
ROUNDS = 60
TELEMETRY_ROUNDS = 60
PACKETS_PER_SOURCE_ROUND = 2
INITIAL_BATTERY = 2000.0
MAX_ATTEMPTS = 3
QUARANTINE_ROUNDS = 2
NORMAL_READMISSION_ROUNDS = 2
BLACKHOLE_FORWARD_P = 0.08
FAULT_FORWARD_P = 0.30
NORMAL_FORWARD_P = 0.98
RADIO_SUCCESS_P = 0.96
SERVICE_PDR_THRESHOLD = 0.50
SERVICE_LOSS_CONFIRM_ROUNDS = 3
STRATEGIES = ("blind_reroute", "generic_permanent_exclusion", "diagnosis_aware")
FAULT_LOADS = (0.20, 0.40)
TOPOLOGIES = ("sparse", "bottleneck")


class FaultState:
    """State for one relay's temporary quarantine, readmission, and power boost."""
    def __init__(self):
        self.remaining = 0
        self.normal_streak = 0
        self.quarantined = False
        self.power_boost = False
        self.readmission_round = None
        self.fault_detections = 0
        self.maintenance_flags = 0
        self.last_fault_round = None
        self.readmission_delay = None
        self.readmission_delays = []
        self.blackhole_isolated = False

    def update(self, round_number, prediction):
        if self.blackhole_isolated:
            return "isolated", False
        if prediction == "blackhole":
            self.blackhole_isolated = True
            return "isolated", False
        if prediction == "fault":
            self.fault_detections += 1
            self.last_fault_round = round_number
            if not self.quarantined and self.readmission_round is not None:
                self.maintenance_flags += 1
            self.quarantined = True
            self.remaining = QUARANTINE_ROUNDS
            self.normal_streak = 0
            self.power_boost = False
            return "quarantined", False
        if self.quarantined:
            self.remaining = max(0, self.remaining - 1)
            self.normal_streak += 1
            if self.remaining == 0 and self.normal_streak >= NORMAL_READMISSION_ROUNDS:
                self.quarantined = False
                self.power_boost = True
                self.readmission_round = round_number
                self.readmission_delay = round_number - self.last_fault_round
                self.readmission_delays.append(self.readmission_delay)
                return "active_boosted", True
            return "quarantined", False
        return ("active_boosted" if self.power_boost else "active"), False


def make_topology(kind):
    """Explicit weighted graph. The bottleneck graph has a single bridge relay."""
    edges = []
    def add(u, v, d=1.0):
        edges.append((u, v, float(d)))
    source_pairs = [(10, 11), (10, 11), (11, 12), (11, 12), (12, 13),
                    (12, 13), (13, 14), (13, 14), (14, 15), (14, 15)]
    for source, pair in zip(SOURCES, source_pairs):
        for relay in pair:
            add(source, relay, 1.0)
    if kind == "bottleneck":
        for relay in range(10, 16):
            add(relay, BRIDGE, 1.1)
        add(BRIDGE, 17, 1.0)
        add(BRIDGE, 18, 1.1)
        add(17, 19, 1.0)
        add(18, 19, 1.0)
        add(19, GATEWAY, 1.0)
    elif kind == "sparse":
        # Several low-degree paths provide a detour around bridge 16.
        for u, v, d in [(10, 16, 1.0), (11, 16, 1.0), (12, 16, 1.0),
                        (13, 17, 1.0), (14, 18, 1.0), (15, 18, 1.0),
                        (16, 19, 1.0), (17, 19, 1.1), (18, 19, 1.2),
                        (19, GATEWAY, 1.0)]:
            add(u, v, d)
    else:
        raise ValueError(f"Unknown topology: {kind}")
    graph = defaultdict(list)
    distance = {}
    for u, v, d in edges:
        graph[u].append((v, d))
        graph[v].append((u, d))
        distance[(u, v)] = distance[(v, u)] = d
    return dict(graph), distance


def assign_labels(seed, fault_fraction):
    """Keep the bridge faulty and vary the number of additional faulty relays."""
    rng = np.random.default_rng(seed + int(fault_fraction * 100_000) + 91)
    # Two branch-end blackholes keep a live alternative branch available in
    # the sparse design; relay 16 remains the deliberate bottleneck fault.
    blackholes = {10, 15}
    pool = [n for n in RELAYS if n not in blackholes and n != BRIDGE]
    rng.shuffle(pool)
    total_faults = round(len(RELAYS) * fault_fraction)
    extras = [n for n in pool[2:] if n not in blackholes][:total_faults - 1]
    faults = {BRIDGE, *extras}
    labels = {n: "normal" for n in range(NODE_COUNT)}
    for node in faults:
        labels[node] = "fault"
    for node in blackholes:
        labels[node] = "blackhole"
    return labels


def out_of_fold_predictions(data, seed):
    """Generate node-grouped out-of-fold predictions for every node and round."""
    excluded = {"node_id", "x", "y", "round", "label"}
    columns = [c for c in data.columns if c not in excluded]
    nodes = data[["node_id", "label"]].drop_duplicates().sort_values("node_id")
    y = nodes.label.to_numpy()
    minimum = int(nodes.label.value_counts().min())
    folds = min(5, minimum)
    if folds < 2:
        raise ValueError("At least two nodes per class are needed for grouped CV")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ids = nodes.node_id.to_numpy()
    predictions = {}
    for fold, (train_idx, test_idx) in enumerate(splitter.split(ids, y)):
        train_nodes, test_nodes = set(ids[train_idx]), set(ids[test_idx])
        train = data[data.node_id.isin(train_nodes)]
        test = data[data.node_id.isin(test_nodes)]
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=seed + fold, n_jobs=-1)
        model.fit(train[columns], train.label)
        for node, rnd, predicted in zip(test.node_id, test["round"],
                                        model.predict(test[columns])):
            predictions[int(node), int(rnd)] = str(predicted)
    if len(predictions) != len(data):
        raise RuntimeError("Incomplete out-of-fold prediction coverage")
    return predictions


def make_diagnoses(seed, labels):
    faults = [n for n, label in labels.items() if label == "fault"]
    blackholes = [n for n, label in labels.items() if label == "blackhole"]
    net = WSN(num_nodes=NODE_COUNT, seed=seed, fault_nodes=faults,
              blackhole_nodes=blackholes, challenging_mode=True)
    telemetry = extract_features(net.simulate(rounds=ROUNDS))
    return net, telemetry, out_of_fold_predictions(telemetry, seed)


def make_healthy_control(seed):
    """No-anomaly control used to check that battery budget alone preserves service."""
    net = WSN(num_nodes=NODE_COUNT, seed=seed, fault_nodes=set(),
              blackhole_nodes=set(), challenging_mode=True)
    telemetry = extract_features(net.simulate(rounds=ROUNDS))
    predictions = {(node, rnd): "normal" for node in range(NODE_COUNT)
                   for rnd in range(1, ROUNDS + 1)}
    return net, telemetry, predictions


def shortest_path(source, graph, excluded=(), dead=(), fault_penalty=1.0,
                  predictions=None, blocked_edges=()):
    excluded = set(excluded)
    dead = set(dead)
    predictions = predictions or {}
    costs, parent = {source: 0.0}, {}
    queue = [(0.0, source)]
    while queue:
        cost, node = heappop(queue)
        if cost != costs.get(node):
            continue
        if node == GATEWAY:
            break
        for neighbor, distance in graph.get(node, ()):
            if (node, neighbor) in blocked_edges or (neighbor, node) in blocked_edges:
                continue
            if neighbor in excluded or (neighbor in dead and neighbor != GATEWAY):
                continue
            penalty = fault_penalty if predictions.get(neighbor) == "fault" else 1.0
            candidate = cost + distance * penalty
            if candidate < costs.get(neighbor, float("inf")):
                costs[neighbor] = candidate
                parent[neighbor] = node
                heappush(queue, (candidate, neighbor))
    if GATEWAY not in costs:
        return None
    path = [GATEWAY]
    while path[-1] != source:
        path.append(parent[path[-1]])
    return list(reversed(path))


def _policy_state(policy, node, prediction, round_number, state, generic_blocked,
                  alerts):
    if policy == "blind_reroute":
        return "active", False
    if policy == "generic_permanent_exclusion":
        if prediction != "normal":
            generic_blocked.add(node)
        if prediction == "blackhole" and node not in alerts:
            alerts.add(node)
        return ("isolated" if node in generic_blocked else "active"), False
    current, readmitted = state[node].update(round_number, prediction)
    if prediction == "blackhole" and node not in alerts:
        alerts.add(node)
    return current, readmitted


def _transmit(path, source, policy, predictions, labels, active_faults,
              power_nodes, graph_distances,
              uniforms, energy, counters):
    if path is None or energy[source] <= 0:
        return False, 0.0, 0.0, 0, "unreachable", 0
    delay = packet_energy = 0.0
    retries = 0
    fallback_routes = 0
    trace = [source]
    current = source
    route = list(path)
    index = 0
    blocked_edges = set()
    while current != GATEWAY:
        if index + 1 >= len(route) or route[index] != current:
            route = shortest_path(current, counters["graph"],
                excluded=counters["excluded"], dead={n for n, e in energy.items() if e <= 0},
                fault_penalty=1.35 if policy == "diagnosis_aware" else 1.0,
                predictions=predictions, blocked_edges=blocked_edges)
            index = 0
            if route is None:
                break
        nxt = route[index + 1]
        d = graph_distances[(current, nxt)]
        success_p = RADIO_SUCCESS_P
        behavior = labels[current]
        if behavior == "blackhole":
            success_p *= BLACKHOLE_FORWARD_P
        elif behavior == "fault" and current in active_faults:
            success_p *= FAULT_FORWARD_P
        else:
            success_p *= NORMAL_FORWARD_P
        if current in power_nodes:
            success_p = min(0.995, success_p + 0.08)
        hop_ok = False
        for attempt in range(MAX_ATTEMPTS):
            if energy[current] <= 0 or (nxt != GATEWAY and energy[nxt] <= 0):
                break
            power_factor = 1.25 if current in power_nodes else 1.0
            tx = (1.0 + 0.002 * d * d) * power_factor
            rx = 0.5
            energy[current] = max(0.0, energy[current] - tx)
            if nxt != GATEWAY:
                energy[nxt] = max(0.0, energy[nxt] - rx)
            packet_energy += tx + (rx if nxt != GATEWAY else 0.0)
            delay += 2.0 + 0.1 * d
            trial = uniforms[current, nxt, attempt]
            if trial < success_p:
                hop_ok = True
                break
            retries += 1
        if not hop_ok:
            blocked_edges.add((current, nxt))
            blocked_edges.add((nxt, current))
            alt = shortest_path(current, counters["graph"],
                excluded=counters["excluded"],
                dead={n for n, e in energy.items() if e <= 0},
                fault_penalty=1.35 if policy == "diagnosis_aware" else 1.0,
                predictions=predictions, blocked_edges=blocked_edges)
            if alt is None:
                break
            fallback_routes += 1
            route, index = alt, 0
            continue
        current = nxt
        trace.append(current)
        index += 1
    return (current == GATEWAY, delay, packet_energy, retries,
            "-".join(map(str, trace)), fallback_routes)


def simulate_condition(seed, fault_fraction, topology_name, network, telemetry,
                       predictions_by_node_round):
    graph, distances = make_topology(topology_name)
    labels = network.behavior_by_node
    fault_nodes = {n for n, label in labels.items() if label == "fault"}
    telemetry_pdr = {(int(row.node_id), int(row.round)): float(row.packet_delivery_ratio)
                     for row in telemetry.itertuples()}
    policy_states = {n: {relay: FaultState() for relay in RELAYS}
                     for n in STRATEGIES}
    per_policy = {}
    for policy in STRATEGIES:
        energy = {n: INITIAL_BATTERY for n in range(NODE_COUNT)}
        generic_blocked, alerts = set(), set()
        power_nodes = set()
        first_node_death = None
        first_service_loss = None
        low_service_streak = 0
        round_records = []
        for rnd in range(1, ROUNDS + 1):
            predictions = {n: predictions_by_node_round[n, rnd]
                           for n in range(NODE_COUNT)}
            # In challenging mode the fault telemetry generator uses a low-PDR
            # branch (uniform .45-.80) with probability 0.55. Reuse that same
            # observed branch as the packet-level transient failure state.
            active_faults = {n for n in fault_nodes
                             if telemetry_pdr[n, rnd] <= 0.80}
            excluded = set()
            readmissions = 0
            for relay in RELAYS:
                status, readmitted = _policy_state(policy, relay, predictions[relay],
                    rnd, policy_states[policy], generic_blocked, alerts)
                if policy == "diagnosis_aware":
                    if status in {"quarantined", "isolated"}:
                        excluded.add(relay)
                        power_nodes.discard(relay)
                    elif status == "active_boosted":
                        power_nodes.add(relay)
                    if readmitted:
                        readmissions += 1
                elif policy == "generic_permanent_exclusion":
                    if status == "isolated":
                        excluded.add(relay)
                # Alert is attached to each newly diagnosed blackhole in all policies.
            total, delivered, delay_sum, energy_sum, retry_sum = 0, 0, 0.0, 0.0, 0
            fallback_routes = fallback_packets = available_count = 0
            source_uniforms = np.random.default_rng(seed * 1_000_003 + rnd * 10_007)
            for source in SOURCES:
                for packet in range(PACKETS_PER_SOURCE_ROUND):
                    uniforms = source_uniforms.random((NODE_COUNT + 1,
                                                       NODE_COUNT + 1,
                                                       MAX_ATTEMPTS))
                    path = shortest_path(source, graph, excluded=excluded,
                        dead={n for n, e in energy.items() if e <= 0},
                        fault_penalty=1.35 if policy == "diagnosis_aware" else 1.0,
                        predictions=predictions)
                    if path is not None:
                        available_count += 1
                    counters = {"graph": graph, "excluded": excluded}
                    ok, delay, used_energy, retries, trace, packet_fallbacks = _transmit(
                        path, source, policy, predictions, labels, active_faults,
                        power_nodes, distances, uniforms, energy, counters)
                    total += 1
                    delivered += int(ok)
                    delay_sum += delay
                    energy_sum += used_energy
                    retry_sum += retries
                    fallback_routes += packet_fallbacks
                    fallback_packets += int(packet_fallbacks > 0)
            alive = [n for n, residual in energy.items() if residual > 0]
            if first_node_death is None and len(alive) < NODE_COUNT:
                first_node_death = rnd
            pdr = delivered / total if total else 0.0
            low_service_streak = low_service_streak + 1 if pdr < SERVICE_PDR_THRESHOLD else 0
            if first_service_loss is None and low_service_streak >= SERVICE_LOSS_CONFIRM_ROUNDS:
                first_service_loss = rnd - SERVICE_LOSS_CONFIRM_ROUNDS + 1
            round_records.append(dict(round=rnd, pdr=pdr, delay=delay_sum / total,
                energy=energy_sum / total, retries=retry_sum / total,
                route_availability=available_count / total,
                fallback_packet_rate=fallback_packets / total,
                fallback_routes_per_packet=fallback_routes / total,
                residual_energy=sum(energy.values()), dead_nodes=NODE_COUNT - len(alive),
                alert_count=len(alerts), readmissions=readmissions,
                maintenance_flags=sum(s.maintenance_flags for s in
                                      policy_states[policy].values())))
        per_policy[policy] = dict(
            packet_delivery_ratio=float(np.mean([r["pdr"] for r in round_records])),
            delay_ms_per_packet=float(np.mean([r["delay"] for r in round_records])),
            abstract_energy_per_packet=float(np.mean([r["energy"] for r in round_records])),
            retries_per_packet=float(np.mean([r["retries"] for r in round_records])),
            route_availability=float(np.mean([r["route_availability"] for r in round_records])),
            fallback_packet_rate=float(np.mean([r["fallback_packet_rate"] for r in round_records])),
            fallback_routes_per_packet=float(np.mean([r["fallback_routes_per_packet"] for r in round_records])),
            first_node_death_round=first_node_death if first_node_death is not None else ROUNDS + 1,
            service_lifetime_50pct_round=first_service_loss if first_service_loss is not None else ROUNDS + 1,
            final_residual_energy=round_records[-1]["residual_energy"],
            unique_blackhole_alerts=round_records[-1]["alert_count"],
            total_readmissions=sum(r["readmissions"] for r in round_records),
            maintenance_flags=round_records[-1]["maintenance_flags"],
            recovery_rounds_to_readmission=float(np.mean([
                delay for s in policy_states[policy].values()
                for delay in s.readmission_delays
            ])) if policy == "diagnosis_aware" and any(
                s.readmission_delays for s in policy_states[policy].values()
            ) else np.nan,
        )
    return per_policy


def confidence_interval(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1)
                 / np.sqrt(len(values)))
    return mean - half, mean + half


def classification_metrics(labels, predictions):
    truth = [labels[n] for n in range(NODE_COUNT) for rnd in range(1, ROUNDS + 1)]
    guesses = [predictions[n, rnd] for n in range(NODE_COUNT)
               for rnd in range(1, ROUNDS + 1)]
    result = {}
    for label in ("normal", "fault", "blackhole"):
        tp = sum(a == label and b == label for a, b in zip(truth, guesses))
        fp = sum(a != label and b == label for a, b in zip(truth, guesses))
        fn = sum(a == label and b != label for a, b in zip(truth, guesses))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[f"{label}_precision"] = precision
        result[f"{label}_recall"] = recall
        result[f"{label}_f1"] = (2 * precision * recall / (precision + recall)
                                  if precision + recall else 0.0)
    result["accuracy"] = sum(a == b for a, b in zip(truth, guesses)) / len(truth)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, class_records = [], []
    healthy_records = []
    for seed in SEEDS:
        healthy_net, healthy_telemetry, healthy_predictions = make_healthy_control(seed)
        for topology_name in TOPOLOGIES:
            healthy_metrics = simulate_condition(seed, 0.0, topology_name,
                healthy_net, healthy_telemetry, healthy_predictions)
            for strategy, values in healthy_metrics.items():
                healthy_records.append(dict(seed=seed, topology=topology_name,
                                            strategy=strategy, **values))
        for fault_fraction in FAULT_LOADS:
            labels = assign_labels(seed, fault_fraction)
            network, telemetry, predictions = make_diagnoses(seed, labels)
            class_records.append({"seed": seed, "fault_fraction": fault_fraction,
                                  **classification_metrics(labels, predictions)})
            for topology_name in TOPOLOGIES:
                metrics = simulate_condition(seed, fault_fraction, topology_name,
                                             network, telemetry, predictions)
                for policy, values in metrics.items():
                    records.append(dict(seed=seed, fault_fraction=fault_fraction,
                                        topology=topology_name, strategy=policy, **values))
    per_seed = pd.DataFrame(records)
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    healthy = pd.DataFrame(healthy_records)
    healthy.to_csv(OUT / "healthy_control_per_seed.csv", index=False)
    healthy_metrics = ["packet_delivery_ratio", "route_availability",
                       "first_node_death_round", "service_lifetime_50pct_round",
                       "abstract_energy_per_packet", "fallback_packet_rate",
                       "fallback_routes_per_packet"]
    healthy.groupby(["topology", "strategy"])[healthy_metrics].agg(["mean", "std"]).reset_index().to_csv(
        OUT / "healthy_control_summary.csv", index=False)
    pd.DataFrame(class_records).to_csv(OUT / "classification_per_seed.csv", index=False)
    metrics = ["packet_delivery_ratio", "delay_ms_per_packet", "abstract_energy_per_packet",
               "retries_per_packet", "route_availability", "fallback_packet_rate",
               "fallback_routes_per_packet", "first_node_death_round",
               "service_lifetime_50pct_round", "final_residual_energy",
               "unique_blackhole_alerts", "total_readmissions", "maintenance_flags",
               "recovery_rounds_to_readmission"]
    summary = per_seed.groupby(["topology", "fault_fraction", "strategy"])[metrics].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    paired_rows = []
    for (topology_name, fault_fraction), group in per_seed.groupby(["topology", "fault_fraction"]):
        wide = group.pivot(index="seed", columns="strategy")
        for metric in metrics:
            for left, right in (("diagnosis_aware", "blind_reroute"),
                                ("diagnosis_aware", "generic_permanent_exclusion")):
                diffs = wide[metric][left] - wide[metric][right]
                lower, upper = confidence_interval(diffs)
                paired_rows.append(dict(topology=topology_name, fault_fraction=fault_fraction,
                    metric=metric, contrast=f"{left} minus {right}",
                    mean_difference=diffs.mean(), ci95_low=lower, ci95_high=upper,
                    n_seeds=diffs.count()))
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_comparisons.csv", index=False)
    class_df = pd.DataFrame(class_records)
    ci_rows = []
    for fault_fraction, group in class_df.groupby("fault_fraction"):
        for metric in [c for c in class_df if c.endswith(("_precision", "_recall", "_f1"))]:
            lo, hi = confidence_interval(group[metric])
            ci_rows.append(dict(fault_fraction=fault_fraction, metric=metric,
                                mean=group[metric].mean(), ci95_low=lo, ci95_high=hi,
                                n_seeds=group[metric].count()))
    pd.DataFrame(ci_rows).to_csv(OUT / "classification_summary_ci.csv", index=False)
    print("Primary endpoint: paired PDR difference, diagnosis-aware minus blind rerouting.")
    print("Healthy no-fault control:")
    print(healthy.groupby(["topology", "strategy"])[healthy_metrics[:4]].mean().round(4).to_string())
    print(per_seed.groupby(["topology", "fault_fraction", "strategy"])
          [metrics[:5]].mean().round(4).to_string())
    print(f"\nSaved new recovery-policy results to {OUT}")


if __name__ == "__main__":
    main()
