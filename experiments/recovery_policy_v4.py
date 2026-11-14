"""Consolidated, fair recovery study (v4).

Replaces the interpretation of Day-1, v2 and v3. Design rules:

* Every policy except ``no_recovery`` shares the same reactive rerouting and the
  same last-resort fallback (unconstrained route). Diagnosis therefore cannot lose
  packets merely by refusing to use a suspected node when no clean route exists.
* Blackholes can drop silently (they acknowledge the hop, then discard the
  packet), which is the usual definition and is invisible to hop-level retries.
  A ``detectable`` mode (drop looks like a failed hop) is kept for comparison.
* Fault nodes can be persistent or transient (degraded only in some rounds,
  using the simulator's own latent state).
* Random-geometric topologies come from the simulator; one deliberately
  constructed bottleneck topology is included and labelled as a stress test.
* An oracle policy (true labels / true degraded state) bounds what any diagnosis
  could achieve, so a null result can be separated from a classifier problem.

All probabilities and energy units are assumptions, not measurements.
"""
from __future__ import annotations

from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from experiments.recovery_evaluation import extract_features
from src.simulator import WSN

OUT = Path("results/recovery_policy_v4")
NODES = 20
GATEWAY_ID = 99
GATEWAY = (50.0, 50.0)
ROUNDS = 40
PACKETS = 4                      # packets per source per round
MAX_ATTEMPTS = 3
CRASH_ROUND = 15
SEEDS = tuple(range(301, 331))
STRATEGIES = ("no_recovery", "blind_reroute", "generic_exclusion",
              "generic_quarantine", "diagnosis_aware", "oracle_diagnosis")
CLASSES = ("normal", "fault", "blackhole")
FAULT_PENALTY = 1.5
SUSPECT_PENALTY = 2.0
E_TX, E_TX_D2, E_RX = 1.0, 0.0002, 0.5
FWD_NORMAL, FWD_BLACKHOLE = 0.98, 0.08
FWD_FAULT_PERSISTENT = 0.62
FWD_FAULT_BAD, FWD_FAULT_GOOD = 0.30, 0.95

SCENARIOS = [
    dict(name="S1_dense_detectable", radio=48.0, topology="random",
         drop="detectable", fault="persistent", crash=0.0),
    dict(name="S2_dense_silent", radio=48.0, topology="random",
         drop="silent", fault="persistent", crash=0.0),
    dict(name="S3_sparse_silent", radio=36.0, topology="random",
         drop="silent", fault="persistent", crash=0.0),
    dict(name="S4_sparse_silent_transient", radio=36.0, topology="random",
         drop="silent", fault="transient", crash=0.0),
    dict(name="S5_bottleneck_silent_transient", radio=40.0, topology="bottleneck",
         drop="silent", fault="transient", crash=0.0),
    dict(name="S6_sparse_silent_transient_crash", radio=36.0, topology="random",
         drop="silent", fault="transient", crash=0.15),
]


# ----------------------------------------------------------------- labels/state
def random_labels(seed):
    rng = np.random.default_rng(seed + 55_001)
    order = rng.permutation(NODES)
    labels = {n: "normal" for n in range(NODES)}
    for n in order[:4]:
        labels[int(n)] = "fault"
    for n in order[4:8]:
        labels[int(n)] = "blackhole"
    return labels


def bottleneck_layout(seed):
    """Constructed stress topology: 4 inner relays (1 blackhole, 2 fault, 1 normal)
    carry all traffic of 16 outer nodes. Positions and labels are by design."""
    rng = np.random.default_rng(seed + 77_003)
    pos = {}
    for n in range(4):
        pos[n] = (float(rng.uniform(60, 70)), float(rng.uniform(42, 58)))
    for n in range(4, NODES):
        pos[n] = (float(rng.uniform(86, 98)), float(rng.uniform(20, 80)))
    labels = {n: "normal" for n in range(NODES)}
    inner = list(rng.permutation(4))
    labels[int(inner[0])] = "blackhole"
    labels[int(inner[1])] = "fault"
    labels[int(inner[2])] = "fault"
    outer = list(rng.permutation(np.arange(4, NODES)))
    for n in outer[:3]:
        labels[int(n)] = "blackhole"
    for n in outer[3:5]:
        labels[int(n)] = "fault"
    return pos, labels


def oof_predictions(data, seed):
    """Node-grouped out-of-fold predictions: no node is predicted by a model that
    trained on it."""
    excluded = {"node_id", "x", "y", "round", "label", "latent_degraded"}
    cols = [c for c in data.columns if c not in excluded]
    nodes = data[["node_id", "label"]].drop_duplicates().sort_values("node_id")
    folds = min(4, int(nodes.label.value_counts().min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ids, y = nodes.node_id.to_numpy(), nodes.label.to_numpy()
    pred = {}
    for k, (tr, te) in enumerate(splitter.split(ids, y)):
        train = data[data.node_id.isin(set(ids[tr]))]
        test = data[data.node_id.isin(set(ids[te]))]
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=seed + k)
        model.fit(train[cols], train.label)
        for n, r, p in zip(test.node_id, test["round"], model.predict(test[cols])):
            pred[int(n), int(r)] = str(p)
    if len(pred) != len(data):
        raise RuntimeError("Incomplete out-of-fold coverage")
    return pred


def build_state(kind, seed):
    if kind == "bottleneck":
        positions, labels = bottleneck_layout(seed)
    else:
        labels = random_labels(seed)
        positions = None
    net = WSN(num_nodes=NODES, seed=seed,
              fault_nodes=[n for n, l in labels.items() if l == "fault"],
              blackhole_nodes=[n for n, l in labels.items() if l == "blackhole"],
              challenging_mode=True)
    data = net.simulate(rounds=ROUNDS, record_latent=True)
    latent = {(int(r.node_id), int(r.round)): bool(r.latent_degraded)
              for r in data.itertuples()}
    feats = extract_features(data.drop(columns=["latent_degraded"]))
    if positions is None:
        positions = {n.node_id: (float(n.x), float(n.y)) for n in net.nodes}
    positions = dict(positions)
    positions[GATEWAY_ID] = GATEWAY
    return dict(labels=labels, positions=positions, latent=latent,
                pred=oof_predictions(feats, seed), feats=feats)


# ------------------------------------------------------------------- routing
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def build_graph(positions, radio):
    ids = list(positions)
    graph = {i: [] for i in ids}
    for i, u in enumerate(ids):
        for v in ids[i + 1:]:
            d = dist(positions[u], positions[v])
            if d <= radio:
                graph[u].append((v, d))
                graph[v].append((u, d))
    return graph


def dijkstra(src, graph, mult, blocked=(), visited=()):
    """Least-cost path; mult maps node -> cost multiplier (inf = excluded)."""
    best, prev, queue = {src: 0.0}, {}, [(0.0, src)]
    while queue:
        c, u = heappop(queue)
        if c != best.get(u):
            continue
        if u == GATEWAY_ID:
            break
        for v, d in graph[u]:
            if (u, v) in blocked or (v in visited and v != GATEWAY_ID):
                continue
            m = 1.0 if v == GATEWAY_ID else mult.get(v, 1.0)
            if m == np.inf:
                continue
            nc = c + d * m
            if nc < best.get(v, np.inf):
                best[v], prev[v] = nc, u
                heappush(queue, (nc, v))
    if GATEWAY_ID not in best:
        return None
    path = [GATEWAY_ID]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def find_route(src, graph, tiers, blocked=(), visited=()):
    """Try each constraint tier, then an unconstrained last-resort fallback."""
    if not tiers:
        return dijkstra(src, graph, {}, blocked, visited), False
    for tier in tiers:
        path = dijkstra(src, graph, tier, blocked, visited)
        if path is not None:
            return path, False
    return dijkstra(src, graph, {}, blocked, visited), True


class Memory:
    """Causal per-node diagnosis memory for the diagnosis-aware policy."""

    def __init__(self):
        self.hist = {n: [] for n in range(NODES)}
        self.isolated = set()

    def update(self, preds):
        for n in range(NODES):
            self.hist[n].append(preds[n])
            if sum(p == "blackhole" for p in self.hist[n][-3:]) >= 2:
                self.isolated.add(n)       # permanent isolation + alert

    def quarantined(self):
        return {n for n in range(NODES) if n not in self.isolated
                and "fault" in self.hist[n][-2:]}


def policy_tiers(strategy, preds, labels, degraded, memory):
    """Return the list of cost-multiplier tiers (strict first) for a strategy."""
    if strategy in ("no_recovery", "blind_reroute"):
        return []
    if strategy == "generic_exclusion":
        return [{n: np.inf for n, p in preds.items() if p != "normal"}]
    if strategy == "generic_quarantine":
        # Ablation control: the SAME hysteresis window as diagnosis_aware, but it
        # cannot tell fault from attack, so every flagged node gets temporary
        # quarantine and later readmission (no permanent isolation, no alert).
        flagged = {n for n in range(NODES)
                   if any(p != "normal" for p in memory.hist[n][-2:])}
        strict = {n: np.inf for n in flagged}
        relaxed = {n: FAULT_PENALTY for n in flagged}
        return [strict, relaxed]
    if strategy == "diagnosis_aware":
        iso = set(memory.isolated)
        quar = memory.quarantined()
        suspect = {n for n, p in preds.items() if p == "blackhole"} - iso
        strict = {n: np.inf for n in iso | quar | suspect}
        relaxed = {n: np.inf for n in iso}
        relaxed.update({n: FAULT_PENALTY for n in quar})
        relaxed.update({n: SUSPECT_PENALTY for n in suspect})
        return [strict, relaxed]
    if strategy == "oracle_diagnosis":
        bad = {n for n, l in labels.items() if l == "blackhole"}
        deg = {n for n, l in labels.items() if l == "fault" and degraded[n]}
        strict = {n: np.inf for n in bad | deg}
        relaxed = {n: np.inf for n in bad}
        relaxed.update({n: FAULT_PENALTY for n in deg})
        return [strict, relaxed]
    raise ValueError(strategy)


# ---------------------------------------------------------------- simulation
def fwd_probability(node, labels, scenario, degraded):
    label = labels[node]
    if label == "blackhole":
        return FWD_BLACKHOLE
    if label == "fault":
        if scenario["fault"] == "transient":
            return FWD_FAULT_BAD if degraded[node] else FWD_FAULT_GOOD
        return FWD_FAULT_PERSISTENT
    return FWD_NORMAL


def send_packet(src, strategy, route, fallback0, graph, positions, labels,
                scenario, degraded, crashed, tiers, uniforms, drop_u, node_energy):
    """Forward one packet. Returns outcome dict and mutates node_energy."""
    out = dict(delivered=False, delay=0.0, energy=0.0, retx=0, changes=0,
               fallback=fallback0, silent_loss=False, hops=0)
    if route is None:
        return out
    cur, path, idx = src, route, 0
    blocked, visited = set(), {src}
    while cur != GATEWAY_ID:
        nxt = path[idx + 1]
        d = dist(positions[cur], positions[nxt])
        radio = max(0.55, 0.99 - 0.35 * (d / scenario["radio"]) ** 2)
        silent_drop_here = False
        if nxt == GATEWAY_ID:
            p = radio
        elif nxt in crashed:
            p = 0.0
        else:
            f = fwd_probability(nxt, labels, scenario, degraded)
            if labels[nxt] == "blackhole" and scenario["drop"] == "silent":
                p = radio                              # ACKed, then dropped
                silent_drop_here = drop_u[nxt] >= f
            else:
                p = radio * f
        ok = False
        for a in range(MAX_ATTEMPTS):
            cost = E_TX + E_TX_D2 * d * d
            node_energy[cur] += cost
            node_energy[nxt] += E_RX
            out["energy"] += cost + E_RX
            out["delay"] += 2.0 + d / scenario["radio"] * 8.0
            if uniforms[cur, nxt, a] < p:
                ok = True
                break
            out["retx"] += 1
        out["hops"] += 1
        if ok:
            if silent_drop_here:
                out["silent_loss"] = True
                return out
            cur = nxt
            visited.add(cur)
            idx += 1
            continue
        blocked.add((cur, nxt))
        blocked.add((nxt, cur))
        if strategy == "no_recovery":
            return out
        alt, fb = find_route(cur, graph, tiers, blocked, visited - {cur})
        if alt is None:
            return out
        out["fallback"] = out["fallback"] or fb
        out["changes"] += 1
        path, idx = alt, 0
    out["delivered"] = True
    return out


def sustained_recovery_rounds(per_round_pdr, pre_crash_pdr, crash_round,
                              horizon=ROUNDS, consecutive=3, threshold=0.95):
    """Rounds after crash until PDR stays above a fraction of baseline.

    Returns the observed delay, or the number of post-crash rounds if recovery
    is not observed before the experiment ends. ``crash_round`` is 1-indexed.
    """
    if pre_crash_pdr <= 0:
        return float(horizon - crash_round + 1)
    target = threshold * pre_crash_pdr
    last_start = horizon - consecutive + 1
    for round_number in range(crash_round, last_start + 1):
        start = round_number - 1
        if np.all(np.asarray(per_round_pdr)[start:start + consecutive] >= target):
            return float(round_number - crash_round)
    return float(horizon - crash_round + 1)


def run_scenario(scenario, seed, state):
    labels, positions = state["labels"], state["positions"]
    graph = build_graph(positions, scenario["radio"])
    rng = np.random.default_rng(seed + 424_243)
    normals = [n for n, l in labels.items() if l == "normal"]
    n_crash = int(round(scenario["crash"] * NODES))
    crash_set = set(int(x) for x in rng.choice(normals, size=n_crash, replace=False)) \
        if n_crash else set()
    memory = Memory()
    acc = {s: dict(off=0, ok=0, energy=0.0, delay_ok=0.0, retx=0, changes=0,
                   fb=0, silent=0, reach=0, node_energy=np.zeros(NODES + 100),
                   per_round_ok=np.zeros(ROUNDS), per_round_off=np.zeros(ROUNDS))
           for s in STRATEGIES}
    for rnd in range(1, ROUNDS + 1):
        preds = {n: state["pred"][n, rnd] for n in range(NODES)}
        memory.update(preds)
        degraded = {n: (state["latent"][n, rnd] if scenario["fault"] == "transient"
                        else True) for n in range(NODES)}
        crashed = crash_set if rnd >= CRASH_ROUND else set()
        tiers = {s: policy_tiers(s, preds, labels, degraded, memory)
                 for s in STRATEGIES}
        for src in range(NODES):
            # Keep application demand fixed after crashes. A crashed source still
            # offers its packet but cannot transmit; counting it as a miss avoids
            # inflating PDR by silently removing failed sources from the denominator.
            routes = ({s: (None, False) for s in STRATEGIES} if src in crashed else
                      {s: find_route(src, graph, tiers[s]) for s in STRATEGIES})
            for pk in range(PACKETS):
                prng = np.random.default_rng(
                    np.random.SeedSequence([seed, rnd, src, pk, 9001]))
                table = prng.random((NODES + 1, NODES + 1, MAX_ATTEMPTS))
                table_full = _LinkTable(table)
                drop_u = prng.random(NODES + 1)
                for s in STRATEGIES:
                    route, fb0 = routes[s]
                    o = (dict(delivered=False, delay=0.0, energy=0.0, retx=0,
                              changes=0, fallback=False, silent_loss=False, hops=0)
                         if src in crashed else
                         send_packet(src, s, route, fb0 and route is not None, graph,
                                     positions, labels, scenario, degraded, crashed,
                                     tiers[s], table_full, drop_u,
                                     acc[s]["node_energy"]))
                    a = acc[s]
                    a["off"] += 1
                    a["ok"] += o["delivered"]
                    a["energy"] += o["energy"]
                    a["delay_ok"] += o["delay"] if o["delivered"] else 0.0
                    a["retx"] += o["retx"]
                    a["changes"] += o["changes"]
                    a["fb"] += int(o["fallback"])
                    a["silent"] += int(o["silent_loss"])
                    a["reach"] += int(route is not None)
                    a["per_round_off"][rnd - 1] += 1
                    a["per_round_ok"][rnd - 1] += o["delivered"]
    rows = []
    flagged = {n for n in range(NODES)
               if np.mean([state["pred"][n, r] != "normal"
                           for r in range(ROUNDS - 9, ROUNDS + 1)]) >= 0.5}
    true_bh = {n for n, l in labels.items() if l == "blackhole"}
    for s in STRATEGIES:
        a = acc[s]
        drain = a["node_energy"][:NODES] / ROUNDS
        pr = a["per_round_ok"] / np.maximum(a["per_round_off"], 1)
        rec = np.nan
        if scenario["crash"] > 0:
            pre = pr[CRASH_ROUND - 8:CRASH_ROUND - 1].mean()
            rec = sustained_recovery_rounds(pr, pre, CRASH_ROUND)
        rows.append(dict(
            scenario=scenario["name"], seed=seed, strategy=s,
            pdr=a["ok"] / a["off"],
            route_available=a["reach"] / a["off"],
            silent_loss_fraction=a["silent"] / a["off"],
            energy_per_offered=a["energy"] / a["off"],
            energy_per_delivered=a["energy"] / max(a["ok"], 1),
            delay_per_delivered_ms=a["delay_ok"] / max(a["ok"], 1),
            retx_per_offered=a["retx"] / a["off"],
            route_changes_per_offered=a["changes"] / a["off"],
            fallback_fraction=a["fb"] / a["off"],
            max_node_energy_per_round=float(drain.max()),
            recovery_rounds_after_crash=rec))
    # node-level diagnosis outputs (do not depend on routing strategy)
    iso = memory.isolated
    tp_diag = len(iso & true_bh)
    tp_gen = len(flagged & true_bh)
    rows_diag = dict(
        scenario=scenario["name"], seed=seed,
        diag_attack_alert_precision=tp_diag / len(iso) if iso else np.nan,
        diag_attack_alert_recall=tp_diag / len(true_bh) if true_bh else np.nan,
        generic_alert_precision=tp_gen / len(flagged) if flagged else np.nan,
        generic_alert_recall=tp_gen / len(true_bh) if true_bh else np.nan,
        normal_nodes_isolated=len([n for n in iso if labels[n] == "normal"]))
    return rows, rows_diag


class _LinkTable:
    """Adapter so uniforms[cur, nxt, attempt] indexes the pre-drawn table;
    the gateway id maps to the last row/column."""

    def __init__(self, table):
        self.t = table

    def __getitem__(self, key):
        u, v, a = key
        u = NODES if u == GATEWAY_ID else u
        v = NODES if v == GATEWAY_ID else v
        return self.t[u, v, a]


# ------------------------------------------------------------------ analysis
def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_table(per_seed, metric, contrasts):
    """contrasts: (a, b, family). Holm correction is applied within each family
    (primary = pre-specified diagnosis effect; secondary = context contrasts)."""
    rows = []
    for scen, g in per_seed.groupby("scenario"):
        wide = g.pivot(index="seed", columns="strategy", values=metric)
        for a, b, family in contrasts:
            d = (wide[a] - wide[b]).dropna().to_numpy()
            n = len(d)
            mean, sd = d.mean(), d.std(ddof=1)
            crit = student_t.ppf(0.975, n - 1)
            se = sd / np.sqrt(n)
            p = ttest_1samp(d, 0.0).pvalue if sd > 0 else (1.0 if mean == 0 else 0.0)
            rows.append(dict(scenario=scen, metric=metric, family=family,
                             contrast=f"{a} minus {b}",
                             mean_diff=mean, ci95_low=mean - crit * se,
                             ci95_high=mean + crit * se, p_raw=p, n_seeds=n))
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, idx in table.groupby("family").groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_raw"].to_numpy())
    return table


def classification_summary(states_by_seed):
    rows = []
    for seed, st in states_by_seed.items():
        y = [st["labels"][n] for n, r in sorted(st["pred"])]
        p = [st["pred"][k] for k in sorted(st["pred"])]
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, p, labels=list(CLASSES), zero_division=0)
        for c, a, b, f in zip(CLASSES, pr, rc, f1):
            rows.append(dict(seed=seed, cls=c, precision=a, recall=b, f1=f))
    df = pd.DataFrame(rows)
    out = []
    for c, g in df.groupby("cls"):
        for m in ("precision", "recall", "f1"):
            v = g[m].to_numpy()
            se = v.std(ddof=1) / np.sqrt(len(v))
            crit = student_t.ppf(0.975, len(v) - 1)
            out.append(dict(cls=c, metric=m, mean=v.mean(),
                            ci95_low=v.mean() - crit * se,
                            ci95_high=v.mean() + crit * se, n_seeds=len(v)))
    return pd.DataFrame(out)


def main(seeds=SEEDS, scenarios=SCENARIOS, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    per_rows, diag_rows, cls_states = [], [], {}
    for seed in seeds:
        states = {}
        for sc in scenarios:
            kind = sc["topology"]
            if kind not in states:
                states[kind] = build_state(kind, seed)
            r, d = run_scenario(sc, seed, states[kind])
            per_rows += r
            diag_rows.append(d)
        cls_states[seed] = states["random"] if "random" in states else next(iter(states.values()))
        print(f"seed {seed} done", flush=True)
        pd.DataFrame(per_rows).to_csv(out / "per_seed_metrics.csv", index=False)
    per_seed = pd.DataFrame(per_rows)
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out / "diagnosis_alert_metrics.csv", index=False)
    metrics = [c for c in per_seed.columns if c not in ("scenario", "seed", "strategy")]
    summary = per_seed.groupby(["scenario", "strategy"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)
    contrasts = [("diagnosis_aware", "generic_quarantine", "primary"),
                 ("diagnosis_aware", "generic_exclusion", "secondary"),
                 ("diagnosis_aware", "blind_reroute", "secondary"),
                 ("generic_quarantine", "blind_reroute", "secondary"),
                 ("oracle_diagnosis", "diagnosis_aware", "secondary")]
    pd.concat([paired_table(per_seed, "pdr", contrasts),
               paired_table(per_seed, "energy_per_delivered", contrasts)]).to_csv(
        out / "paired_comparisons.csv", index=False)
    classification_summary(cls_states).to_csv(out / "classification_summary.csv",
                                              index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
