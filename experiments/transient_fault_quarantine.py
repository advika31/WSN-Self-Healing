"""Day 1 controlled experiment for transient-fault quarantine and re-admission.

This small event-level model deliberately does not modify the v3/v1 experiments.
See docs/transient_fault_quarantine_day1.md for its assumptions and definitions.
"""
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/transient_fault_quarantine_day1")
GATEWAY = 9
ROUNDS = 30
PACKETS_PER_ROUND = 100
MAX_ATTEMPTS = 3
QUARANTINE_ROUNDS = 2
READMISSION_NORMAL_ROUNDS = 2
STRATEGIES = ("no_recovery", "generic_permanent", "diagnosis_quarantine")
BLACKHOLES = {8}
FAULT_RELAY = 1


def topology(scenario):
    """Return an explicit, reproducible graph and positions for each test case."""
    if scenario == "sparse":
        # A two-route sparse mesh. The longer route is the fault-tolerant bypass.
        edges = [(0, 1, 1.0), (1, 9, 1.0), (0, 2, 1.2), (2, 3, 1.2),
                 (3, 4, 1.2), (4, 9, 1.2), (0, 8, 1.0), (8, 9, 1.0)]
    elif scenario == "bottleneck":
        # Relay 1 is the preferred bridge; the second, longer path is independent.
        edges = [(0, 1, 1.0), (1, 9, 1.0), (0, 2, 1.5), (2, 3, 1.5),
                 (3, 9, 1.5), (0, 8, 1.0), (8, 9, 1.0)]
    else:
        raise ValueError(f"Unknown topology: {scenario}")
    adjacency = defaultdict(list)
    distances = {}
    for u, v, distance in edges:
        adjacency[u].append((v, distance))
        adjacency[v].append((u, distance))
        distances[(u, v)] = distances[(v, u)] = distance
    return dict(adjacency), distances


def shortest_path(source, adjacency, excluded=()):
    """Shortest-distance path from source to gateway, respecting exclusions."""
    excluded = set(excluded) | BLACKHOLES
    costs = {source: 0.0}
    parent = {}
    queue = [(0.0, source)]
    while queue:
        cost, node = heappop(queue)
        if cost != costs.get(node):
            continue
        if node == GATEWAY:
            break
        for neighbor, distance in adjacency.get(node, ()):
            if neighbor in excluded and neighbor != GATEWAY:
                continue
            candidate = cost + distance
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


class QuarantinePolicy:
    """Finite fault quarantine; two normal diagnoses allow re-admission."""
    def __init__(self, quarantine_rounds=QUARANTINE_ROUNDS,
                 normal_rounds=READMISSION_NORMAL_ROUNDS):
        self.quarantine_rounds = quarantine_rounds
        self.normal_rounds = normal_rounds
        self.remaining = 0
        self.normal_streak = 0
        self.ever_quarantined = False
        self.readmitted_round = None

    def update(self, round_number, diagnosis):
        if diagnosis == "blackhole":
            # Blackholes are hard-isolated separately by every routing policy.
            return "isolated"
        if diagnosis == "fault":
            self.remaining = self.quarantine_rounds
            self.normal_streak = 0
            self.ever_quarantined = True
            return "quarantined"
        if self.remaining:
            self.remaining -= 1
            self.normal_streak += 1
            if self.remaining == 0 and self.normal_streak >= self.normal_rounds:
                self.readmitted_round = round_number
                return "active"
            return "quarantined"
        if self.ever_quarantined and self.normal_streak < self.normal_rounds:
            self.normal_streak += 1
            if self.normal_streak >= self.normal_rounds:
                self.readmitted_round = round_number
        return "active"


def diagnosis_for(round_number, fault_start=6, fault_end=11):
    """A declared synthetic detector trace for the transient bridge fault."""
    if fault_start <= round_number <= fault_end:
        return "fault"
    return "normal"


def simulate_seed(seed, scenario):
    adjacency, distances = topology(scenario)
    scenario_code = 1 if scenario == "sparse" else 2
    rng = np.random.default_rng(seed * 1_000_003 + scenario_code * 100_003)
    shared_trials = rng.random((ROUNDS, PACKETS_PER_ROUND, 10, 10, MAX_ATTEMPTS))
    quarantine = QuarantinePolicy()
    generic_blacklisted = set(BLACKHOLES)
    records = []
    for rnd in range(1, ROUNDS + 1):
        diagnosis = diagnosis_for(rnd)
        status = quarantine.update(rnd, diagnosis)
        if diagnosis == "fault":
            generic_blacklisted.add(FAULT_RELAY)
        policy_exclusions = {
            "no_recovery": set(BLACKHOLES),
            "generic_permanent": generic_blacklisted,
            "diagnosis_quarantine": set(BLACKHOLES) | (
                {FAULT_RELAY} if status == "quarantined" else set()
            ),
        }
        for policy in STRATEGIES:
            route = shortest_path(0, adjacency, policy_exclusions[policy])
            fallback = False
            if route is None and policy != "no_recovery":
                # Emergency route may use a fault relay, but never the blackhole.
                route = shortest_path(0, adjacency, BLACKHOLES)
                fallback = route is not None
            route_available = route is not None
            for packet in range(PACKETS_PER_ROUND):
                delivered = route_available
                delay = energy = 0.0
                if route:
                    for u, v in zip(route, route[1:]):
                        d = distances[(u, v)]
                        hop_ok = False
                        hop_delay = 0.0
                        for attempt in range(MAX_ATTEMPTS):
                            energy += 1.0 + 0.0002 * d**2 + 0.5
                            delay += 2.0 + d * 0.08
                            hop_delay += 2.0 + d * 0.08
                            # Synthetic radio delivery and transient forwarding.
                            radio_p = 0.97
                            forward_p = 0.98
                            if v == FAULT_RELAY:
                                forward_p = 0.15 if diagnosis == "fault" else 0.98
                            success_p = radio_p * forward_p
                            if shared_trials[rnd - 1, packet, u, v, attempt] < success_p:
                                hop_ok = True
                                break
                        if not hop_ok:
                            delivered = False
                            break
                records.append(dict(seed=seed, scenario=scenario, round=rnd,
                    packet=packet, strategy=policy, diagnosis=diagnosis,
                    fault_relay_status=status, route_available=route_available,
                    fallback=fallback, delivered=delivered, delay_ms=delay,
                    energy_units=energy,
                    route="-".join(map(str, route)) if route else "unreachable",
                    recovery_rounds_to_readmission=(
                        quarantine.readmitted_round - 11
                        if policy == "diagnosis_quarantine"
                        and quarantine.readmitted_round is not None else np.nan
                    )))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = []
    for scenario in ("sparse", "bottleneck"):
        for seed in range(101, 131):
            all_records.extend(simulate_seed(seed, scenario))
    results = pd.DataFrame(all_records)
    results.to_csv(OUT / "packet_results.csv", index=False)
    per_seed = results.groupby(["scenario", "seed", "strategy"]).agg(
        packet_delivery_ratio=("delivered", "mean"),
        delay_ms_per_offered_packet=("delay_ms", "mean"),
        abstract_energy_per_offered_packet=("energy_units", "mean"),
        route_availability=("route_available", "mean"),
        fallback_fraction=("fallback", "mean"),
        recovery_rounds_to_readmission=("recovery_rounds_to_readmission", "first"),
    ).reset_index()
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary = per_seed.groupby(["scenario", "strategy"]).agg(
        packet_delivery_ratio_mean=("packet_delivery_ratio", "mean"),
        packet_delivery_ratio_sd=("packet_delivery_ratio", "std"),
        delay_ms_mean=("delay_ms_per_offered_packet", "mean"),
        abstract_energy_mean=("abstract_energy_per_offered_packet", "mean"),
        route_availability_mean=("route_availability", "mean"),
        fallback_fraction_mean=("fallback_fraction", "mean"),
        readmission_rounds_mean=("recovery_rounds_to_readmission", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved Day 1 results to {OUT}")


if __name__ == "__main__":
    main()
