# Recovery evaluation v2: design and assumptions

## What this experiment measures

`experiments.recovery_network_evaluation` compares no recovery, generic rerouting, and diagnosis-aware routing over identical simulated node placements, per-round readings, source demands, and radio links. Each of the six held-out nodes generates one packet per round. A packet can travel directly to the gateway or through one other held-out node. A hop exists when Euclidean distance is at most 55 coordinate units. All policies enumerate the same feasible routes. No recovery minimizes route distance without consulting diagnosis; generic rerouting excludes predicted fault and blackhole relays; diagnosis-aware routing excludes predicted blackholes and applies a 1.5 distance multiplier to predicted faults. If policy exclusions remove all routes, the policy falls back to all feasible routes and logs this.

## Reported metrics

- Packet delivery ratio: mean end-to-end delivery probability, modeled as the product of source and relay telemetry PDR (gateway link reliability is assumed perfect).
- Route availability: fraction of source-round demands with at least one feasible route.
- Delay: sum of the source and relay telemetry delays for a selected route; direct routes use source delay only.
- Energy: abstract per-packet radio cost. Every transmission costs `1 + 0.0002*d^2` units; a relay reception costs one additional unit. These are relative model units, not joules or measured battery consumption.
- Mean hops, fraction routed through ground-truth unhealthy/blackhole relays, and fallback route count.

This is still an analytical proxy, not packet-level simulation or a standards-compliant protocol. It uses deterministic disk connectivity, one possible relay hop, fixed node behavior labels, independently observed per-node PDR values as a proxy for link reliability, no interference/retransmissions/queueing/control overhead, and no battery depletion caused by these evaluated packets. The energy metric excludes sensing, classifier computation, and management traffic. Therefore, it cannot support claims about real network lifetime or deployed performance. A subsequent validation should use an established simulator or a testbed and model directed, stochastic links and forwarding.

The held-out nodes form both the source set and relay pool so their predicted labels are out-of-node predictions. The training-node predictions are not used as relay decisions. A route without any feasible candidate is counted as unavailable for every strategy; those rounds are not silently discarded from PDR.

## Reproduction

From the project root in PowerShell:

```powershell
python -m experiments.recovery_network_evaluation
```

New files go to `results/recovery_v2/`; legacy `results/recovery_*.csv` files are not touched.
