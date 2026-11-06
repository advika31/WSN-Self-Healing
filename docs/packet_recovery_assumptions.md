# Packet-level recovery experiment (v3)

## Design

Run `python -m experiments.packet_recovery_evaluation` from the project root. Results are written to `results/recovery_v3_packet/`; prior output folders are not overwritten. Ten seeds use the existing 20-node placement and challenging telemetry generator. The six held-out nodes are the packet sources and relay candidates, so the RF predictions used for policy decisions are out-of-node. Each held-out source offers 20 packets in each of 30 rounds.

The held-out nodes and gateway form a graph with a deterministic disk range of 48 coordinate units. Links within range have a distance-dependent radio success probability `max(0.55, 0.99 - 0.35*(d/range)^2)`. A successful reception at an intermediate node is then subject to a behavior-dependent forwarding probability: normal 0.98, fault 0.62, blackhole 0.08. The gateway always accepts a successfully transmitted packet. These values make packet loss, link failures, retries, and forwarding attacks explicit; they are selected synthetic parameters, not calibrated radio or attack measurements.

Every hop gets up to three transmission attempts. A failed edge is treated as unavailable for the remainder of that packet. No recovery keeps the initially selected shortest path and drops the packet if a hop exhausts its retries. Generic rerouting excludes nodes predicted as fault or blackhole and, after a failed edge, recomputes a path. Diagnosis-aware routing excludes predicted blackholes, keeps predicted faults with a 1.5 distance-cost multiplier, and also recomputes after a failed edge. If policy constraints make a route impossible, the simulator logs and uses an emergency unconstrained route. All strategies share the same placements, behavior labels, generated telemetry, and pre-drawn uniform link-trial values for each source/round/packet/directed edge/attempt. The experiment is paired by seed and offered packet.

## Metrics

- Packet delivery ratio: delivered packets divided by offered packets, including unreachable demands as failures.
- Delay: accumulated per-attempt delay proxy (2 ms base plus a distance term); an exhausted or unreachable packet retains the delay incurred before drop.
- Energy: abstract units per offered packet; each attempt costs `1 + 0.0002*d^2` transmit units and 0.5 receive units. Failed attempts also consume transmit energy.
- Hops and retransmissions per offered packet.
- Route changes per offered packet and emergency fallback fraction.

This is now packet-by-packet forwarding with stochastic directed transmission outcomes, finite retransmissions, failed-edge avoidance, and alternate path selection. It remains a small custom synthetic simulator rather than an implementation of RPL or a validated radio model. It does not model MAC contention, collisions, queues, control-plane message overhead, asynchronous traffic, duty cycling, physical battery units, or calibration against hardware. In particular, packet forwarding probabilities are assigned from fixed class behavior rather than learned from measured links. Consequently, report results as controlled synthetic-model outcomes only.

## Ten-seed result

The run produced 108,000 packet-policy rows (10 seeds × 30 rounds × 6 sources × 20 packets × 3 strategies). Means and sample standard deviations across the ten per-seed values were:

| Policy | PDR | Physical route availability | Delay (ms/packet) | Energy units/packet | Retransmissions/packet | Route changes/packet | Fallback fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diagnosis-aware | 0.8802 ± 0.2467 | 0.9167 ± 0.2115 | 10.4186 ± 3.4957 | 2.4397 ± 0.7028 | 0.3863 ± 0.2031 | 0.0149 ± 0.0175 | 0.0343 ± 0.0698 |
| Generic rerouting | 0.8802 ± 0.2467 | 0.9167 ± 0.2115 | 10.4367 ± 3.4946 | 2.4438 ± 0.7029 | 0.3886 ± 0.2021 | 0.0156 ± 0.0174 | 0.0753 ± 0.0964 |
| No recovery | 0.8705 ± 0.2439 | 0.9167 ± 0.2115 | 10.1122 ± 3.2676 | 2.3647 ± 0.6444 | 0.3645 ± 0.1926 | 0 | 0 |

Across the ten paired seeds, each recovery policy improved mean PDR over no recovery by 0.00964 (sample SD 0.00979; descriptive paired t interval 0.00263 to 0.01665). Diagnosis-aware and generic rerouting had identical PDR on every seed. Thus, this run suggests a modest benefit from rerouting under the chosen synthetic model, but provides no evidence that the diagnosis-specific policy improves delivery over generic rerouting. Recovery incurred slightly higher modeled delay and energy per offered packet. Between-seed variation is large: one topology has much lower physical route availability, so these averages are sensitive to placement. Treat this interval as exploratory, not confirmatory evidence.

## Reproducibility and tests

```powershell
python -m unittest discover -s tests -v
python -m experiments.packet_recovery_evaluation
```
