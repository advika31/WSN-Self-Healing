# Packet recovery sensitivity study v1

## Research question and scope

Does the packet-level rerouting result persist as radio range, network size, and the synthetic forwarding reliability of fault/blackhole relays change? Does diagnosis-aware routing outperform generic rerouting in any of these one-factor conditions? This is a pre-specified one-factor-at-a-time study, not a search for favorable parameter combinations. It does not estimate interactions among the factors.

## Conditions

There are ten unique conditions: one shared baseline and nine alternatives. The baseline is 20 nodes, 48 coordinate-unit radio range, blackhole forwarding probability 0.08, and fault forwarding probability 0.62. Alternatives are:

- Radio range: 35, 60, and 75 (baseline 48).
- Network size: 30 and 50 nodes (baseline 20).
- Blackhole forwarding probability: 0.05 and 0.20 (baseline 0.08).
- Fault forwarding probability: 0.40 and 0.80 (baseline 0.62).

Each condition is run for 30 seeds (101–130). Each node offers ten packets per round over 30 rounds. Up to three transmission attempts are allowed per hop. The remaining packet, routing, link-quality, delay, energy, fallback, and diagnosis policy parameters match the packet experiment v3.

## Diagnosis and network-size comparability

The behavior prevalence remains 20% fault, 20% blackhole, and 60% normal at all three network sizes. The 20-node core has exactly four nodes in each anomalous class; additional nodes are added in balanced five-node blocks. For each network size and seed, stratified group cross-validation holds out whole nodes (four folds at 20 nodes, five folds at 30 and 50 nodes). Every relay decision uses an out-of-fold prediction from a Random Forest that did not train on that node. The training procedure and class prevalence are identical across conditions of a fixed size.

Within each fixed-size seed, the generated telemetry and out-of-fold predictions are reused across radio-range and forwarding-probability alternatives. Directed link-trial uniforms are keyed by seed, round, source, packet, transmitter, receiver, and attempt; common edges therefore receive the same trial values across parameter settings and policies. The gateway ID stays fixed across network sizes. Node placements and behavior labels for the first 20 nodes are shared across size conditions. The simulator uses a sequential global random stream for telemetry, so changing network size can alter later-round telemetry draws; size comparisons are matched by seed but are not exact common-random-number comparisons of telemetry.

## Outcomes and analysis

Primary outcome is delivered packets / offered packets. Secondary outcomes are physical route availability, delay per offered packet, abstract energy units per offered packet, hops, retransmissions, route changes, and emergency fallback fraction. Summaries report means and sample standard deviations across the 30 seed-level experimental units. Paired t intervals are exploratory and calculated over seed-level differences, not over individual packets. Outputs include:

- `per_seed_metrics.csv`: all 30 seed summaries for each condition and strategy.
- `sensitivity_summary.csv`: mean and sample standard deviation by condition and strategy.
- `paired_pdr_differences.csv`: within-condition paired strategy contrasts.
- `scenario_vs_baseline.csv`: paired within-seed parameter changes versus the shared baseline, by strategy and outcome.

PDR is the primary endpoint. Energy and delay trade-offs should be interpreted alongside PDR. Do not treat the many packet events within a seed as independent replicates.

## Reproduction

From the project root in PowerShell:

```powershell
python -m unittest discover -s tests -v
python -m experiments.packet_recovery_sensitivity
```

The experiment checkpoints seed summaries after each seed to `results/recovery_sensitivity_v1/per_seed_metrics.csv`, so an interrupted run can resume. Existing `results/recovery_v3_packet/` and prior outputs are not overwritten.

## Limitations

The simulation remains a small custom model with a deterministic disk graph, selected distance-dependent link probabilities, fixed class-based forwarding probabilities, no MAC/queue/collision/control-plane model, and uncalibrated abstract energy. Radio range and forwarding probability scenarios are assumptions rather than measured device behavior. The study tests sensitivity within this model and cannot establish field performance or publication-grade external validity by itself.

## Observed results

Rerouting improved PDR over no recovery in all ten conditions. At the baseline (20 nodes, 48-unit range), mean PDR was 0.9397 for no recovery, 0.9980 for generic rerouting, and 0.9981 for diagnosis-aware routing. The paired diagnosis-aware minus no-recovery difference was +0.05834 (95% exploratory t interval +0.04304 to +0.07365). Across the conditions, that difference ranged from +0.00331 at range 75 to +0.12618 at range 35; the paired intervals were above zero in all ten conditions.

The delivery benefit shrank as connectivity saturated: at radio range 35, mean PDR was 0.8090 with no recovery and 0.9352 with diagnosis-aware routing; at range 75, it was 0.9967 and 1.0000. At 30 and 50 nodes under the 48-unit range, recovery PDR was approximately 1.000 while no-recovery PDR was 0.9300 and 0.9348. Physical route availability was 0.9633 at range 35 and 1.000 in all other conditions. There is no monotonic network-size effect in the no-recovery baseline, and all recovery results at higher density approach the ceiling.

Diagnosis-aware routing was numerically higher than generic rerouting by at most 0.00174 (0.174 percentage points), at range 35. All paired 95% intervals for diagnosis-aware minus generic rerouting included zero; most differences were smaller than 0.0002. This study therefore still provides no clear diagnosis-specific delivery benefit. Energy and delay costs vary by condition. At baseline, diagnosis-aware recovery added about 0.338 ms and 0.0519 abstract energy units per offered packet versus no recovery. At range 35, the costs rose to about 2.125 ms and 0.4373 units. At range 75, the differences were about 0.040 ms and 0.0133 units. Values at high density can be equal or slightly lower because fewer attempts/routes are needed; these are model-specific proxy units.

The new study is not numerically interchangeable with the earlier v3 experiment: it uses all network nodes as sources and relays, cross-fitted predictions for every node, a balanced 20%/20% fault/blackhole prevalence, and ten packets per node-round. Keep its estimates separate from the earlier six-node held-out packet run. The main pattern is that generic rerouting helps over a static no-recovery path under the tested conditions, while diagnosis-specific policy selection changes PDR little or not at all.
