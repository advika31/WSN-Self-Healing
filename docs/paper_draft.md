# Diagnosis-Aware Fault Classification and Recovery in a Simulated Wireless Sensor Network

**Draft status:** working draft for internal review. Evidence comprises synthetic packet simulation and separate public-data classifier checks; no measured network recovery or hardware result is available. Recovery-policy v2 is superseded for interpretation by v3 because its no-fault control exposed battery-budget service failure. Do not submit until full-text novelty review, bibliography verification, supervisor review, and venue-specific formatting are complete.

## Abstract

Wireless sensor networks must maintain data delivery despite unreliable nodes and routing attacks, but node-level diagnosis is useful only if it informs effective recovery. This work combines synthetic node telemetry classification with packet-level recovery experiments and two public-data classifier checks. A five-fold node-grouped Random Forest evaluation on WSN-DS produced accuracy 0.99545 and macro-F1 0.97086 (95% node-cluster bootstrap interval 0.96448–0.97504); this dataset is NS-2/LEACH simulated attack traffic. On Intel Lab environmental readings with explicitly injected temperature faults, five-fold mote-grouped evaluation produced accuracy 0.92727 and macro-F1 0.59798 (95% mote-cluster bootstrap interval 0.58109–0.61590). The Intel fault labels are synthetic, not naturally observed. In the new 30-seed packet-recovery study, diagnosis-aware quarantine did not improve PDR over blind rerouting; mean PDR was low in all conditions. No public-data recovery or physical-device result is available. The abstract battery and lifetime measures remain uncalibrated proxies, and ESP32 measurement is future work.

**Keywords:** wireless sensor networks; fault diagnosis; blackhole attack; Random Forest; routing recovery; simulation

## 1. Introduction

Wireless sensor networks operate with constrained devices and lossy communication, so routing decisions must account for reliability and resource use. RPL is an IETF routing protocol designed for low-power and lossy networks [1]. Routing attacks such as blackhole behavior can disrupt forwarding, and the literature has surveyed a range of routing-attack detection approaches and their limitations [2]. Energy-aware routing is also a longstanding concern; classic work on wireless microsensor networks demonstrates that routing and energy dissipation are coupled [3].

This work studies a compact pipeline: generate node telemetry, classify each node's observed behavior, translate the predicted class into a node status, and evaluate routing choices. The goal is not to claim a new routing protocol or a field-ready intrusion detector. Instead, the current contribution is a reproducible prototype and a transparent investigation of whether diagnosis-dependent eligibility rules change outcomes in the project's synthetic model.

The experiments highlight an important distinction: classification performance does not establish that a recovery policy improves delivery. Earlier recovery prototypes and the v2 stress run did not establish such an advantage. A no-fault control revealed that v2's battery budget was too small to sustain even a healthy network, so v3 raised the abstract budget and retained a matched control. In v3, the healthy control had mean PDR 0.9998, while diagnosis-aware recovery delivered fewer packets than blind rerouting in all four fault/topology conditions. This remains a result about the coded synthetic model, not a deployed WSN.

## 2. Research questions

1. How accurately does a Random Forest distinguish normal, fault, and blackhole telemetry when class distributions overlap?
2. How does this classifier compare with a decision tree and a fixed rule-based baseline?
3. Does packet rerouting improve delivery relative to no recovery across selected radio-range, network-size, and forwarding-probability conditions, and does diagnosis-aware routing improve on generic rerouting?

## 3. Methods

### 3.1 Synthetic network and telemetry

The simulator places 20 nodes in a 100 × 100 coordinate region and runs for 30 rounds. Each node retains one behavior label across its rounds. In challenging mode, normal, fault, and blackhole telemetry distributions overlap; fault behavior is intermittent and blackhole behavior selectively drops packets. The reported features include RSSI, PDR, packet loss, delay, residual energy, and energy consumption. Rolling means and standard deviations are calculated from each node's current and preceding two readings.

The simulator's labels and telemetry are synthetic by construction. They do not derive from a physical testbed, an emulated radio stack, or a public deployment trace. These labels represent the simulator's chosen behavior classes, not independently verified field faults or attacker behaviors.

### 3.2 Classification and split

The classifier is a 100-tree Random Forest with balanced class weights. Evaluation holds out entire nodes rather than randomly splitting individual readings, which reduces leakage from the same node appearing in both training and evaluation. The challenging classifier results currently available are accuracy 0.8239 and macro-F1 0.8160. Mean per-class recall is 0.9717 for normal, 0.5650 for fault, and 0.9350 for blackhole. The fault recall deficit is material: a diagnosis-aware policy cannot route around faults the classifier does not identify.

The baseline comparison reports challenging-data accuracy/macro-F1 of 0.5233/0.5005 for fixed rules, 0.7956/0.7918 for a decision tree, and 0.8239/0.8160 for the Random Forest. The decision tree has higher fault recall (0.6400 vs. 0.5650), while the Random Forest has higher blackhole recall (0.9350 vs. 0.8567). Model selection therefore depends on the operational cost of missed faults versus missed attacks; accuracy alone does not resolve this tradeoff.

### 3.3 Diagnosis and recovery policy

Predictions map to active (normal), unreliable (fault), or isolated (blackhole) statuses. The original recovery experiment is a simplified single-relay gateway proxy and is reported only as preliminary context. Its ten-seed mean PDRs were 0.8224 for diagnosis-aware, 0.8503 for generic rerouting, and 0.6655 for no recovery. These estimates are not directly comparable to the new topology proxy because the measurement procedures differ; the old experiment selected one relay per round and derived outcomes from that node's telemetry.

The new proxy evaluates the six held-out nodes as packet sources and as the only candidate relay pool. Each source generates one packet in each of 30 rounds. A route is either a direct gateway hop or one relay hop followed by a gateway hop. Connectivity is a deterministic disk model with a 55-unit transmission radius. All policies use the same placement, per-round telemetry, source demands, and feasible-route set. No recovery minimizes route distance without consulting diagnosis; generic rerouting excludes predicted fault and blackhole relays; diagnosis-aware routing excludes predicted blackhole relays and multiplies predicted-fault route distance by 1.5. If exclusions leave no feasible route, the implementation falls back to all feasible routes and records the fallback.

End-to-end delivery probability is modeled as the product of the source and relay telemetry PDRs; gateway-link success is assumed perfect. Delay sums source and relay telemetry delay. Abstract energy cost for each transmission is `1 + 0.0002 × distance²`; relay reception adds one energy unit. The constants are illustrative and are not calibrated to a radio or expressed in joules. Route availability is the fraction of source-round demands with a feasible route. The ten seeds are summarized by the mean and sample standard deviation of per-seed metrics.

### 3.4 Packet-forwarding recovery experiment

The follow-on experiment retains the six held-out nodes as sources and relays and uses a 48-unit disk communication graph. Each source offers 20 packets per round. Link delivery is stochastic and directed, with distance-dependent radio success; intermediate forwarding success depends on the simulator's ground-truth behavior label. Each hop gets up to three attempts. A failed edge is blocked for the current packet, after which the two recovery policies recompute a route. No recovery keeps its initial shortest route and drops the packet after retries fail. Policies receive the same topology and pre-drawn transmission trial values for each offered packet, directed edge, and attempt. Route availability, packet delivery, delay, retransmissions, energy, and route changes are recorded. Parameters and limitations are documented in `docs/packet_recovery_assumptions.md`.

### 3.5 Sensitivity and scalability design

The one-factor-at-a-time study uses one shared baseline (20 nodes, radio range 48, blackhole forwarding probability 0.08, fault forwarding probability 0.62) and nine alternatives: radio ranges 35, 60, and 75; network sizes 30 and 50; blackhole forwarding probabilities 0.05 and 0.20; and fault forwarding probabilities 0.40 and 0.80. Each of the ten conditions is run on 30 seeds. Each node offers ten packets in each of 30 rounds. Behavior prevalence is 20% fault, 20% blackhole, and 60% normal. Four-fold stratified group cross-validation is used at 20 nodes and five-fold at 30 and 50 nodes, so each routed node has an out-of-fold prediction. Telemetry/predictions are reused across parameter conditions of a fixed size and seed, with keyed link draws for paired policy comparisons. The network-size change uses matched seeds and a shared 20-node placement/label core; simulator telemetry's sequential random stream means size conditions are not exact common-random-number replications. Outcomes are summarized by seed, with seed as the experimental unit.

### 3.6 Reproducibility

Run the classification and recovery modules from the repository root in the documented Python environment. The packet experiment writes to `results/recovery_v3_packet/` and preserves legacy recovery outputs:

```powershell
python -m unittest discover -s tests -v
python -m experiments.packet_recovery_evaluation
python -m experiments.packet_recovery_sensitivity
python -m experiments.plot_recovery_sensitivity
```

## 4. Results

### 4.1 Classification

On the challenging in-domain node-held-out evaluation, Random Forest accuracy was 0.8239 and macro-F1 was 0.8160. Class recall was uneven, with fault recall at 0.5650. In comparison, the decision tree had lower aggregate scores but greater fault recall; the rule-based baseline was weaker on aggregate metrics. This indicates that error costs and class-specific recall should guide future model selection.

The previously recorded cross-domain result (train on clean data, evaluate on challenging data) was accuracy 0.5433 and macro-F1 0.4788. This substantial reduction shows sensitivity to the change in telemetry distributions and cautions against using the in-domain score as a generalization claim.

### 4.2 Recovery proxy

Across ten seeds, the new topology proxy produced these means (sample standard deviations are in the generated `mean_std_metrics.csv`):

| Policy | End-to-end PDR | Route availability | Delay (ms) | Abstract energy units |
|---|---:|---:|---:|---:|
| Diagnosis-aware | 0.7169 | 0.9667 | 50.7008 | 230.8910 |
| Generic rerouting | 0.7170 | 0.9667 | 50.6709 | 230.8856 |
| No recovery | 0.7170 | 0.9667 | 50.6709 | 230.8856 |

Differences are negligible and do not favor diagnosis-aware recovery. Route availability is identical because physical reachability, rather than diagnosis, determines whether a route exists. The policies mostly choose the same route, and the fault penalty can slightly increase distance and delay. This is a negative result for the present heuristic and proxy, not evidence that diagnosis-aware recovery cannot help under other topologies or protocols.

### 4.3 Packet-forwarding recovery

The 10-seed packet experiment comprised 108,000 packet-policy records. Physical route availability was 0.9167 ± 0.2115 for all policies. PDR was 0.8802 ± 0.2467 for both recovery policies and 0.8705 ± 0.2439 for no recovery. The paired mean PDR difference versus no recovery was 0.00964 (sample SD 0.00979; exploratory paired t interval 0.00263–0.01665). Diagnosis-aware and generic rerouting had identical PDR for every seed. Diagnosis-aware routing used 2.4397 ± 0.7028 abstract energy units per offered packet and 10.4186 ± 3.4957 ms modeled delay, compared with 2.4438 ± 0.7029 units and 10.4367 ± 3.4946 ms for generic rerouting, and 2.3647 ± 0.6444 units and 10.1122 ± 3.2676 ms for no recovery. Rerouting modestly improves delivery over retaining a failed path in this simulator, with increased modeled delay and energy; diagnosis-specific routing does not improve delivery over generic rerouting. High between-seed variation reflects sensitivity to the six-node topology, so these estimates are exploratory.

### 4.4 Sensitivity and scalability

The 30-seed sensitivity study evaluated ten unique conditions: one shared baseline and nine one-factor alternatives. At the baseline (20 nodes, range 48, blackhole forwarding probability 0.08, fault forwarding probability 0.62), mean PDR was 0.9397 for no recovery, 0.9980 for generic rerouting, and 0.9981 for diagnosis-aware routing. The paired diagnosis-aware minus no-recovery difference was +0.05834 (95% exploratory t interval +0.04304 to +0.07365). Across all conditions, the recovery-minus-no-recovery paired intervals remained above zero; the diagnosis-aware PDR gain ranged from +0.00331 at radio range 75 to +0.12618 at range 35.

The benefit narrowed as radio connectivity increased: at range 35, PDR was 0.8090 for no recovery and 0.9352 for diagnosis-aware routing; at range 75, it was 0.9967 and 1.0000. At network sizes 30 and 50, recovery PDR approached 1.000 while no-recovery PDR was 0.9300 and 0.9348. Changing blackhole forwarding probability from 0.05 to 0.20 moved no-recovery PDR from 0.9362 to 0.9523, while diagnosis-aware PDR stayed between 0.9979 and 0.9988. Across all ten conditions, diagnosis-aware versus generic rerouting differences were at most 0.00174; every paired 95% interval included zero. This study therefore supports a rerouting benefit within the tested model, but still does not show an incremental diagnosis-specific benefit. At baseline, recovery added about 0.338 ms and 0.052 abstract energy units per offered packet versus no recovery; at range 35, these costs increased to about 2.125 ms and 0.437 units.

The mean sensitivity results and seed-level paired contrasts are saved in `results/recovery_sensitivity_v1/`. Figure 1 plots mean PDR and ±1 sample standard deviation across the four parameter families.

![Packet delivery ratio across sensitivity conditions](../results/recovery_sensitivity_v1/figures/packet_delivery_sensitivity.png)

### 4.5 Transient quarantine and topology stress experiment

**Superseded exploratory result.** The following v2 run used a 1,000-unit initial battery. Its later healthy-control audit showed service failure from battery depletion even without anomalous nodes. Preserve the numbers as a record of the design iteration only; do not use them as the primary finding. The v3 study in Section 4.7 recalibrates this abstract budget against a matched no-fault control and is the current packet-recovery result.

The separate recovery-policy v2 study used 30 seeds (201–230), two topologies (a sparse graph with detours and a graph whose only source-to-gateway bridge is relay 16), and 20% or 40% fault loads among ten relays. Relay 16 was always a transient fault; two other relays were blackholes. All policies received the same topology, offered packets, fault-state trace, and keyed link draws. The classifier used node-grouped out-of-fold predictions. Diagnosis-aware recovery permanently isolated predicted blackholes, quarantined predicted faults, and required two normal diagnoses after a two-round minimum before readmission. Readmitted relays received a modeled power boost. Generic permanent exclusion was included as a conservative secondary comparator; blind rerouting was the primary comparison.

The diagnosis-aware minus blind-rerouting PDR differences were −0.1174 (95% paired t interval −0.1445 to −0.0904) for bottleneck/20% faults, −0.1659 (−0.1805 to −0.1512) for bottleneck/40%, −0.0316 (−0.0791 to 0.0159) for sparse/20%, and −0.2243 (−0.2708 to −0.1779) for sparse/40%. Mean PDRs for blind rerouting were 0.2186, 0.2005, 0.3758, and 0.3764, respectively; diagnosis-aware PDRs were 0.1012, 0.0346, 0.3442, and 0.1521. Diagnosis-aware recovery beat permanent generic exclusion in all four paired contrasts, but that is not evidence it beats routing without diagnosis. The very low PDRs mean none of these conditions provides a viable-service demonstration.

At 20% fault load, fault-class precision/recall/F1 were 0.8510/0.5614/0.6750 (95% seed-level intervals: 0.8350–0.8669, 0.5354–0.5873, 0.6522–0.6977). At 40% they were 0.8603/0.7574/0.8048 (0.8505–0.8702, 0.7393–0.7755, 0.7924–0.8173). These are means and t intervals over 30 seeds, with each seed as the replicate; node-round rows were not treated as independent. Full seed-level results and all class intervals are in `results/recovery_policy_v2/`.

This result is an exploratory negative finding for the current action policy and simulator, not a general statement that diagnosis-aware recovery cannot help. In particular, fault states are derived from the simulator's observed low-PDR telemetry branch, candidate routes are sparse and hand-built, blackhole prediction errors affect exclusion, and low mean PDR suggests the offered load/topology/reliability combination is poorly calibrated. The battery model and lifetime proxy should not be interpreted as physical energy or field network life.

### 4.6 Public-data classifier checks

The downloaded WSN-DS copy contained 374,661 rows, including 8,873 exact duplicate rows. After removing exact duplicates, five-fold node-grouped out-of-fold evaluation covered 365,788 rows across 100 inferred node IDs, using 15 numeric predictors while excluding node ID, time, cluster-head ID and target. Accuracy was 0.99545 and macro-F1 was 0.97086 (95% node-cluster bootstrap interval 0.96448–0.97504). Full per-class precision, recall and F1 with cluster-bootstrap intervals are in `results/public_dataset_validation/wsn_ds_class_metrics.csv`. The dataset's documented origin is NS-2/LEACH simulated traffic, so these values do not represent field attack detection or transient sensor fault classification.

For Intel Lab, the code aggregated 2.3-million-scale raw readings into five-minute temperature means and injected bias, drift, stuck-at and spike faults in a separate derived copy. It used 53 motes, 407,147 observed bins, and five-fold mote-grouped validation. Accuracy was 0.92727, while macro-F1 was 0.59798 (95% mote-cluster bootstrap interval 0.58109–0.61590). The normal class dominates the data; stuck-fault precision was especially poor, about 0.0063, despite 0.1834 recall. Therefore accuracy is misleading, and the result mainly shows how difficult the selected injection design is for this model. The fault labels are created by this study and cannot be described as naturally observed Intel Lab faults. Injection definitions, exclusions, class metrics, and checksums are in `docs/public_dataset_validation.md` and `results/public_dataset_validation/`.

### 4.7 Recovery policy v3 with healthy control

V3 retained the paired 30-seed design (201–230), 60 rounds, sparse and route-critical bottleneck graphs, and 20%/40% fault loads. It added a matched all-normal control and used 2,000 abstract energy units per node after v2's all-normal run depleted its first node around rounds 31–32 with a 1,000-unit budget. Under the v3 control, mean PDR was 0.9998, route availability was 1.0, and the first-node-death and 50%-service-loss events were both right-censored beyond the 60-round horizon (stored as round 61). This supports the selected simulated horizon; it does not calibrate energy to joules or real battery life.

The primary outcome was the paired seed-level difference in PDR for diagnosis-aware minus blind rerouting. Across 30 seeds, the differences were −0.3471 (95% paired t interval −0.3735 to −0.3208) for bottleneck/20% faults, −0.3710 (−0.3925 to −0.3495) for bottleneck/40%, −0.2861 (−0.3655 to −0.2067) for sparse/20%, and −0.5757 (−0.6366 to −0.5149) for sparse/40%. Every interval excludes zero in the direction of lower delivery under diagnosis-aware eligibility. In the bottleneck graph, excluding the sole bridge can disconnect sources; in the sparse graph, exclusion also reduces the available detours. The finding is therefore an adverse outcome for these specific coded rules and topologies, not a general test of all diagnosis-aware recovery policies.

The v3 metrics also include delay, abstract energy, retries, route availability, fallback use, alerts, readmissions, maintenance flags and two lifetime proxies. Fallback means an alternate-route computation after a link exhausts retries; it does not mean diagnosis failure. Reduced delay or energy alongside lower PDR may reflect dropped packets rather than efficiency. All energy and lifetime values are model-based and uncalibrated. The paired analysis treats seeds as replicates; the four topology/load contrasts are exploratory and no multiplicity adjustment was applied. Full outputs are in `results/recovery_policy_v3/`; definitions and limitations are in `docs/recovery_policy_v3.md`.

## 5. Discussion

The experiments suggest two separate research challenges. First, the classifier has difficulty identifying intermittent faults under overlapping distributions. Second, even accurate diagnoses do not automatically improve routing if the action policy is poorly matched to topology or if alternatives are not meaningfully different. The sensitivity study indicates a consistent rerouting benefit over retaining the initial path across the tested conditions, with a smaller benefit at high radio connectivity as PDR approaches its ceiling. Diagnosis-aware and generic rerouting remain nearly indistinguishable in PDR, and their paired intervals overlap zero in every condition. These findings support the project's differentiated action framework as an implementation concept, but do not validate that fault-versus-blackhole differentiation improves network outcomes. The model remains synthetic and uncalibrated.

Several assumptions limit interpretation. The packet experiments use a disk connectivity model and selected distance/behavior probabilities rather than calibrated radio traces. They omit MAC contention, collisions, queues, control-plane overhead, mobility, and route discovery messages. The abstract energy model is not physically calibrated and excludes sensing, computation, and idle listening. The first packet run used six held-out nodes; the sensitivity study uses all nodes with grouped out-of-fold predictions. Forwarding success is assigned directly from synthetic ground-truth labels, so these are controlled mechanism experiments rather than independent networking validation. The network-size conditions use matched seeds and a shared core, although the simulator's sequential random stream means telemetry is not exactly paired across different sizes.

The study should therefore be positioned as an early simulation prototype and methodological audit. Comparison with RPL or another explicit protocol implementation is future work, not a result of the present code. A follow-on should scale the packet model, calibrate it against an established simulator or testbed, include routing/control overhead and battery depletion, add oracle and random/geometric baselines, and assess paired per-seed differences under pre-specified outcomes. Hardware validation on an ESP32 testbed is also required before any deployment claim.

## 6. Conclusion

This prototype combines synthetic telemetry classification with diagnosis-driven route selection and two separate public-data classifier checks. WSN-DS classification concerns simulated NS-2/LEACH attack traffic. Intel Lab uses real environmental measurements with synthetic injected temperature faults; it does not provide natural fault labels. In the calibrated-control packet experiment, diagnosis-aware recovery delivered fewer packets than blind rerouting in each of four synthetic stress conditions. This project therefore has no demonstrated diagnosis-specific recovery benefit, no naturally labeled Intel fault-detection result, and no field performance evidence. ESP32 inference/RAM/energy measurement remains future work because the hardware and power meter are unavailable. Supervisor review is required before selecting a venue or submission.

## References

1. T. Winter, P. Thubert, A. Brandt, J. Hui, R. Kelsey, P. Levis, K. Pister, R. Struik, J. P. Vasseur, and R. Alexander, “RPL: IPv6 Routing Protocol for Low-Power and Lossy Networks,” RFC 6550, Internet Engineering Task Force, 2012. https://www.rfc-editor.org/rfc/rfc6550
2. Z. Alansari, N. B. Anuar, A. Kamsin, and M. R. Belgaum, “A systematic review of routing attacks detection in wireless sensor networks,” *PeerJ Computer Science*, vol. 8, e1135, 2022, doi:10.7717/peerj-cs.1135.
3. W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, “Energy-efficient communication protocol for wireless microsensor networks,” in *Proceedings of the 33rd Annual Hawaii International Conference on System Sciences*, 2000, doi:10.1109/HICSS.2000.926982.

The expanded 18-source candidate set and source-audit status are recorded in `docs/literature_review.md`. It is not yet a fully reviewed bibliography: several entries have only metadata/abstract checks, and all items used for novelty claims still require full-text comparison. Particularly relevant prior work already studies reuse of faulty WSN nodes [Chanak & Banerjee, 2016, doi:10.1016/j.eswa.2015.09.040], while a 2026 controlled HIL study reports a 50-node self-healing system and measured power-related outcomes [Alghawli et al., doi:10.3389/fcomp.2026.1914381]. Claims that faulty-node reuse or HIL energy measurement are novel are therefore unsupported.

## Items to resolve before submission

- Re-run all paper-reported experiments from a clean environment and archive exact software versions and seeds.
- Re-run all paper-reported experiments from a clean environment and archive exact software versions and seeds.
- Treat the paired t intervals as exploratory; pre-specify contrasts and address multiple comparisons before making confirmatory claims.
- Complete a full-text comparison of the 18 candidate sources, verify every final reference, and add only sources directly used in the argument.
- Add all complete results tables, including standard deviations and per-class precision/recall/F1, from the committed experiment artifacts.
- Strengthen the networking model and baseline suite before framing a recovery contribution.
- Obtain supervisor review, complete an authorized plagiarism check, and select a venue only after scope and contribution are agreed. No submission or preprint has been made.
