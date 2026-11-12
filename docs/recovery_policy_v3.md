# Recovery policy v3: calibrated control and route-critical stress tests

## Why this revision exists

The v2 experiment used an initial abstract battery budget of 1,000 units. An all-normal diagnostic run delivered only about 0.52 of offered packets and depleted its first node around rounds 31–32 of a 60-round run. This showed that battery exhaustion dominated the endpoint, even without anomalous nodes. V2 and its results are preserved as an exploratory run; do not pool its results with v3.

V3 raises the budget to 2,000 and adds a matched all-normal, no-fault control for all 30 seeds and both topologies. The control delivered mean PDR 0.9998, had route availability 1.0, and did not reach either lifetime event by round 60. The latter is recorded as round 61 and means right-censored (>60), not a measured lifetime. This control supports interpreting the v3 fault runs as stress cases while retaining the limitation that the battery scale is abstract and not physically calibrated.

## Frozen study

Run `python -m experiments.recovery_policy_v3`. It uses seeds 201–230, 60 rounds, 10 relay candidates, 10 sources, and two packets per source per round. The two hand-built graphs are a sparse topology with detours and a bottleneck topology in which relay 16 is the only bridge. Relay 16 is a transient fault in every seed; the total fault fraction is 20% or 40%. Two branch-end relays are blackholes. The Random Forest predictions are grouped out-of-fold by node, with the held-out node's readings never used to train its predictor. Forwarding behavior and labels remain synthetic.

## Policies

- `blind_reroute` uses no diagnosis; after three failed transmissions on a hop, it excludes that edge for the current packet and searches for an alternate route.
- `generic_permanent_exclusion` permanently excludes a relay after a predicted fault or blackhole.
- `diagnosis_aware` permanently isolates predicted blackholes and counts one alert per isolated node. It quarantines a relay diagnosed as faulty for at least two rounds, then requires two consecutive normal predictions before readmission. Re-admission applies a modeled transmit-power adjustment. Repeated fault diagnosis after re-admission increments a maintenance flag.

The dataset classifier evaluations are separate: neither WSN-DS nor Intel Lab observations drive these packet simulations.

## Endpoints and limitations

- **Primary:** paired, seed-level diagnosis-aware minus blind-reroute PDR, with a two-sided 95% paired t interval, stratified by topology and fault fraction. Seed is the replicate (n=30).
- **Other packet outcomes:** delay per offered packet (model milliseconds), abstract energy per offered packet, retries per offered packet, route availability before transmission, fallback packet rate, and fallback routes per offered packet.
- **Fallback definition:** a packet is a fallback packet if a hop exhausts its retries and a different route is computed. The route count records each alternate-route computation. It is a link-failure detour measure, not a diagnosis fallback, quarantine re-entry, or guaranteed successful delivery.
- **Lifetime proxies:** first node reaching zero abstract energy; and the first of three consecutive rounds with PDR below 0.50. Event-free runs are recorded as round 61 (>60, right-censored). These are not battery life measurements. Energy excludes idle listening, sensing, computation, collision costs, and control packets.
- **Recovery time:** rounds from fault diagnosis until re-admission, as defined by the implemented policy. It is policy timing, not physical fault repair time.

The v3 healthy-control summary and seed-level control are `results/recovery_policy_v3/healthy_control_summary.csv` and `healthy_control_per_seed.csv`. Main outputs are `per_seed_metrics.csv`, `summary.csv`, and `paired_comparisons.csv`. They are independent of and do not overwrite v2 outputs.

## Results interpretation

Across 30 paired seeds, diagnosis-aware minus blind-reroute PDR was:

| Topology | Fault fraction | Difference | 95% paired t interval |
| --- | ---: | ---: | ---: |
| Bottleneck | 20% | -0.3471 | [-0.3735, -0.3208] |
| Bottleneck | 40% | -0.3710 | [-0.3925, -0.3495] |
| Sparse | 20% | -0.2861 | [-0.3655, -0.2067] |
| Sparse | 40% | -0.5757 | [-0.6366, -0.5149] |

This study does not show a diagnosis-aware delivery advantage. In the fixed bottleneck graph, the diagnosis-driven exclusions often remove the only usable bridge; in sparse conditions, exclusions also reduce service. Lower delay and energy under exclusion can result from dropping work and must not be presented as efficiency gains without joint consideration of delivered packets. The observed outcomes are evidence about these specific synthetic action rules and graphs only. No protocol benchmark, field deployment, or public-data packet-recovery validation was run.
