# Diagnosis-Aware Fault Classification and Recovery in a Simulated Wireless Sensor Network

**Draft status:** working draft for internal review. The current evidence is simulation-only. Do not submit until the open methodological items below are addressed and bibliographic details are checked against the target venue.

## Abstract

Wireless sensor networks must maintain data delivery despite unreliable nodes and routing attacks, but node-level diagnosis is useful only if it can inform an effective recovery decision. This study develops a simulation prototype that classifies node telemetry as normal, fault, or blackhole behavior and maps diagnoses to routing eligibility. A Random Forest model uses rolling summaries of received signal strength, packet delivery ratio (PDR), packet loss, delay, residual energy, and energy consumption. In a challenging synthetic setting with overlapping telemetry distributions, the previously recorded node-held-out evaluation reported accuracy 0.8239 and macro-F1 0.8160; fault recall (0.5650) was lower than normal (0.9717) and blackhole (0.9350) recall. A ten-seed packet-forwarding experiment models multi-hop paths, directed stochastic link outcomes, retransmissions, failed-edge avoidance, and route recomputation. Mean PDR was 0.8802 under both generic and diagnosis-aware rerouting and 0.8705 with no recovery. The paired-seed difference versus no recovery was 0.00964 (exploratory 95% t interval 0.00263–0.01665), while generic and diagnosis-aware routing had identical PDR on every seed. Thus, this simulation suggests rerouting may improve delivery under the selected assumptions, but does not show an incremental diagnosis-specific benefit. These are synthetic-model results, not evidence of deployed WSN performance.

**Keywords:** wireless sensor networks; fault diagnosis; blackhole attack; Random Forest; routing recovery; simulation

## 1. Introduction

Wireless sensor networks operate with constrained devices and lossy communication, so routing decisions must account for reliability and resource use. RPL is an IETF routing protocol designed for low-power and lossy networks [1]. Routing attacks such as blackhole behavior can disrupt forwarding, and the literature has surveyed a range of routing-attack detection approaches and their limitations [2]. Energy-aware routing is also a longstanding concern; classic work on wireless microsensor networks demonstrates that routing and energy dissipation are coupled [3].

This work studies a compact pipeline: generate node telemetry, classify each node's observed behavior, translate the predicted class into a node status, and evaluate routing choices. The goal is not to claim a new routing protocol or a field-ready intrusion detector. Instead, the current contribution is a reproducible prototype and a transparent investigation of whether diagnosis-dependent eligibility rules change outcomes in the project's synthetic model.

The experiments highlight an important distinction. Classification performance does not by itself show that a recovery policy improves network delivery. An initial single-relay proxy did not establish an advantage for diagnosis-aware recovery. A follow-on one-relay topology proxy still showed virtually identical outcomes. We then implemented a packet-forwarding experiment with retries and failed-edge route recomputation. It suggests a small delivery improvement for rerouting versus no recovery, but diagnosis-aware and generic rerouting tie on PDR.

## 2. Research questions

1. How accurately does a Random Forest distinguish normal, fault, and blackhole telemetry when class distributions overlap?
2. How does this classifier compare with a decision tree and a fixed rule-based baseline?
3. Do diagnosis-aware route eligibility and fault penalties improve delivery, availability, delay, or modeled energy relative to generic rerouting and no recovery in a paired synthetic packet-forwarding experiment?

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

### 3.5 Reproducibility

Run the classification and recovery modules from the repository root in the documented Python environment. The packet experiment writes to `results/recovery_v3_packet/` and preserves legacy recovery outputs:

```powershell
python -m unittest discover -s tests -v
python -m experiments.packet_recovery_evaluation
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

## 5. Discussion

The experiments suggest two separate research challenges. First, the classifier has difficulty identifying intermittent faults under overlapping distributions. Second, even accurate diagnoses do not automatically improve routing if the action policy is poorly matched to topology or if alternatives are not meaningfully different. The packet experiment provides initial evidence that route recomputation after link failure can improve delivery over retaining the failed route, but diagnosis-aware and generic rerouting tie on PDR. It therefore does not establish an incremental diagnosis-specific gain. The six-node synthetic graph is too small and abstract to support performance claims for deployed networks.

Several assumptions limit interpretation. The packet experiment uses a disk connectivity model and selected distance/behavior probabilities rather than calibrated radio traces. It omits MAC contention, collisions, queues, control-plane overhead, mobility, and route discovery messages. The abstract energy model is not physically calibrated and excludes sensing, computation, and idle listening. Only six held-out nodes participate in the routing pool. Forwarding success is assigned directly from synthetic ground-truth labels, so this is a controlled mechanism experiment rather than independent networking validation.

The study should therefore be positioned as an early simulation prototype and methodological audit. Comparison with RPL or another explicit protocol implementation is future work, not a result of the present code. A follow-on should scale the packet model, calibrate it against an established simulator or testbed, include routing/control overhead and battery depletion, add oracle and random/geometric baselines, and assess paired per-seed differences under pre-specified outcomes. Hardware validation on an ESP32 testbed is also required before any deployment claim.

## 6. Conclusion

This prototype combines synthetic telemetry classification with diagnosis-driven route selection. The challenging classification results show useful separation for normal and blackhole classes but weaker fault recall and a large cross-domain performance decline. The packet-forwarding experiment suggests that rerouting can improve delivery over keeping a failed route under the selected synthetic assumptions. Diagnosis-aware routing ties generic rerouting on PDR, so the current results do not establish a diagnosis-specific gain. Larger packet-level validation with calibrated radio, routing, energy, and attack models, followed by independent sensor data and testbed evaluation, is needed before making deployment claims.

## References

1. T. Winter, P. Thubert, A. Brandt, J. Hui, R. Kelsey, P. Levis, K. Pister, R. Struik, J. P. Vasseur, and R. Alexander, “RPL: IPv6 Routing Protocol for Low-Power and Lossy Networks,” RFC 6550, Internet Engineering Task Force, 2012. https://www.rfc-editor.org/rfc/rfc6550
2. Z. Alansari, N. B. Anuar, A. Kamsin, and M. R. Belgaum, “A systematic review of routing attacks detection in wireless sensor networks,” *PeerJ Computer Science*, vol. 8, e1135, 2022, doi:10.7717/peerj-cs.1135.
3. W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, “Energy-efficient communication protocol for wireless microsensor networks,” in *Proceedings of the 33rd Annual Hawaii International Conference on System Sciences*, 2000, doi:10.1109/HICSS.2000.926982.

## Items to resolve before submission

- Re-run all paper-reported experiments from a clean environment and archive exact software versions and seeds.
- Reconcile the older and new recovery estimates in the narrative; they use different evaluation procedures and must not be pooled.
- Add uncertainty intervals and paired tests only after selecting a pre-specified primary outcome and a defensible independent experimental unit.
- Verify every reference and add current primary studies directly relevant to diagnosis-aware recovery and routing under blackhole behavior.
- Add all complete results tables, including standard deviations and per-class precision/recall/F1, from the committed experiment artifacts.
- Strengthen the networking model and baseline suite before framing a recovery contribution.
