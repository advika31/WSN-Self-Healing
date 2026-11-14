# When Can Node Diagnosis Change Packet Recovery in a Wireless Sensor Network? A Paired Simulation Study

**Manuscript status:** revised research draft for supervisor and technical review; not submission-ready. It replaces the older narrative in `paper_draft.md` for current findings while preserving that file as an audit trail. The project has no physical-network recovery result. The principal evidence is a custom synthetic simulator; the public datasets support classifier checks only.

## Abstract

Node diagnosis can affect packet delivery only when its routing actions change a route decision that matters. We evaluate this condition in a pre-specified, paired simulation study with six dense, sparse, transient-fault, bottleneck and node-crash scenarios. For each of 30 seeds, every relay prediction is generated out of fold by a Random Forest that has not trained on that node. The primary outcome is packet delivery ratio (PDR); the primary contrast is diagnosis-aware recovery versus a diagnosis-blind quarantine policy with matched two-round memory. The diagnosis-aware policy permanently isolates nodes repeatedly predicted as blackholes and temporarily quarantines predicted faults, allowing readmission after clean predictions. Across the six scenario-specific paired contrasts, mean PDR differences ranged from approximately 0.00001 to 0.02083. After Holm correction, only the sparse transient-fault scenario with 15% node crashes had a statistically significant difference: +0.01745 (95% unadjusted paired t interval 0.00692–0.02798; Holm-adjusted p=0.0122). This is evidence for a conditional effect of the **combined policy package** in this simulator, not a causal estimate of diagnosis alone. All scenario dynamics, radio outcomes and energy units are synthetic. Crash recovery to 95% of pre-crash PDR for three consecutive rounds was not observed for the diagnosis-aware policy within the 26-round post-crash observation window. Separate classifier-only checks yielded WSN-DS accuracy 0.99545 and macro-F1 0.97086 on node-held-out records from an NS-2/LEACH dataset, and Intel Lab injected-fault accuracy 0.92727 and macro-F1 0.59798 on mote-held-out records. Neither dataset tests packet recovery; Intel fault labels are injected. No calibrated energy, network-lifetime or hardware result is available.

**Keywords:** wireless sensor networks; fault diagnosis; blackhole; packet delivery; recovery; node-held-out validation; simulation

## 1. Introduction

Wireless sensor networks must move readings through links and relays that may fail or behave maliciously. A detector's classification score alone does not show whether acting on its predictions preserves service. A diagnosis can help only when the recovery policy changes relay selection, and false alarms can disconnect routes or concentrate traffic on fewer nodes.

This work examines that connection using a small, reproducible packet-level model. It asks whether diagnosis-conditioned handling of suspected blackholes and transient faults changes packet delivery under specified connectivity and failure assumptions. The study includes an explicit diagnosis-blind quarantine comparator, sparse and constructed bottleneck topologies, and a post-crash scenario. The resulting evidence is exploratory and simulator-specific. It is not a new routing standard, field evaluation, or proof of deployment readiness.

Prior work already covers learned WSN fault detection, fault-node management and reuse, self-healing optimization, and recovery-time or energy outcomes. In particular, previous fault-management work studies reuse of faulty nodes; recent systems combine learned detection with recovery optimization; and a 2026 controlled hardware-in-the-loop (HIL) study reports physical-node validation and measured power. Therefore, this study does not claim that these broad topics are new. Its narrow question is how a pre-specified diagnosis-aware policy package compares with a memory-matched diagnosis-blind quarantine policy in the project's paired simulator. The closest supplied studies include fault/recovery optimization [3], HIL self-healing [4], and simulated reinforcement-learning self-healing [5].

## 2. Research questions and hypotheses

1. **Primary:** Does diagnosis-aware recovery change PDR relative to diagnosis-blind quarantine with a matched two-round history, separately in each of six pre-specified scenarios?
2. **Secondary:** How do the policies compare with blind rerouting and generic exclusion in PDR, route availability, fallback, modeled delay and abstract energy?
3. **Classifier context:** How does node-held-out classification perform in the synthetic simulator and in two separate public-data classification checks?

No interaction effects or favorable parameter combinations are estimated. The six scenario-specific primary contrasts form one family and are corrected together using Holm's procedure.

## 3. Methods

### 3.1 Packet simulator and scenarios

The custom simulator has 20 nodes in a 100 by 100 coordinate region, 40 rounds and four offered packets per source per round. Each network has four fault, four blackhole and twelve normal nodes. A random-geometric link exists when the Euclidean distance is within the scenario's radio range. Routes terminate at a fixed gateway. Each hop permits up to three attempts; distance-dependent link outcomes are drawn from shared pre-generated uniforms so policies within a seed face paired random conditions.

Forwarding success is assigned directly from the ground-truth class: normal nodes use probability 0.98; blackhole nodes use 0.08; persistent faults use 0.62; and transient faults use 0.30 in degraded rounds and 0.95 otherwise. A blackhole can be detectable (its drop resembles a failed hop) or silent (it acknowledges its hop, then drops the packet). The silent-drop assumption matters because a hop retry cannot reveal an acknowledged downstream discard. These are model rules, not learned or measured node behaviors.

| Scenario | Topology and radio range | Blackhole behavior | Fault behavior | Additional condition |
|---|---|---|---|---|
| S1 | Random geometric, 48 units | Detectable | Persistent | Dense reference |
| S2 | Random geometric, 48 units | Silent | Persistent | Dense reference |
| S3 | Random geometric, 36 units | Silent | Persistent | Sparse connectivity |
| S4 | Random geometric, 36 units | Silent | Transient | Sparse connectivity |
| S5 | Constructed bottleneck, 40 units | Silent | Transient | Four inner relays carry outer-node traffic |
| S6 | Random geometric, 36 units | Silent | Transient | 15% of normal nodes crash at round 15 |

In the bottleneck condition, four inner relays (one blackhole, two faults and one normal) carry the traffic of 16 outer nodes. In the crash condition, crashed nodes continue to contribute offered source demand but cannot transmit. This fixed denominator prevents PDR from rising just because failed sources disappear from the workload. Physical route availability is computed before stochastic forwarding; it is identical across policies within a scenario and is not a measure of successful delivery.

### 3.2 Classifier and recovery policies

Telemetry is generated by the project's synthetic challenging-mode simulator. A 100-tree Random Forest with balanced class weights is trained using node-level stratified folds; all records from a test node are held out together. Feature extraction excludes node ID, coordinates, round, class label and latent transient-fault state. For each seed, the out-of-fold predictions are reused across policies. The labels and behavior ground truth are simulator-generated.

Six policies are compared:

- **No recovery:** retain the original route and stop after retry exhaustion.
- **Blind reroute:** recompute a route after a failed hop, with no diagnosis constraints.
- **Generic exclusion:** exclude all nodes predicted non-normal for that round; use an unconstrained fallback if no such route exists.
- **Generic quarantine:** apply the same two-round prediction history as diagnosis-aware recovery, but temporarily quarantine every non-normal prediction and relax the constraint when needed. It does not distinguish fault from attack, permanently isolate blackholes, or raise a diagnosis-specific attack alert.
- **Diagnosis-aware:** a node predicted blackhole in at least two of its last three rounds is permanently isolated and flagged; a predicted fault is quarantined while recently flagged and may be readmitted after clean predictions. A second route tier allows degraded relays with a 1.5 distance-cost multiplier and currently suspected blackholes with a 2.0 multiplier, while keeping confirmed blackholes isolated. The shared unconstrained last resort remains available.
- **Oracle:** use true labels and the latent transient state; this is an upper-bound comparator, not an implementable detector.

The primary contrast is diagnosis-aware minus generic quarantine. This does not isolate a pure causal effect of classification: the policies also differ in permanent blackhole isolation, suspicion penalties, and alerting. Conclusions are therefore about the full policy package. A future ablation must hold those actions fixed while varying only the diagnosis information if a causal diagnosis claim is sought.

### 3.3 Metrics and statistical analysis

PDR is delivered packets divided by all offered packets, including demands from crashed sources. Route availability is the fraction of offers with a graph route before stochastic forwarding. Delay is averaged over delivered packets only. Energy is a unitless accounting model: per-transmission base and distance costs plus receive costs. It is not joules. The maximum per-node energy per round is reported only as a traffic-concentration/load proxy; there is no finite battery or defensible network-lifetime estimate. Fallback fraction, retransmissions and route changes are also recorded.

For S6, recovery time is the number of rounds after round 15 until PDR reaches at least 95% of the pre-crash mean for three consecutive rounds. A run that does not meet this criterion before round 40 is right-censored at 26 post-crash rounds. A seed is the independent experimental unit. Primary estimates are paired mean PDR differences over 30 matched seeds, with two-sided 95% paired t intervals and paired t tests. The intervals are unadjusted; p-values are Holm-adjusted within the six-test primary family. Other policy contrasts and metrics are secondary/exploratory. No parameter sweep or tuning is performed in this study.

### 3.4 Separate public-data classification checks

These datasets are not fed into the packet simulator and cannot validate its routing results.

**WSN-DS.** The downloaded copy contains 374,661 records and 100 encoded node groups; exact duplicates are removed, leaving 365,788 records. The source paper describes NS-2 simulation using LEACH, 23 features and normal, blackhole, grayhole, flooding and scheduling classes [1]. The analysis excludes ID/time/cluster-head and label fields, uses 15 numeric predictors and node-grouped five-fold out-of-fold evaluation. Confidence intervals resample nodes rather than records. Accuracy is 0.99545; macro-F1 is 0.97086 (95% node-cluster bootstrap interval 0.96448–0.97504). The result concerns held-out node identifiers in the same simulated dataset; it is not physical attack detection, and WSN-DS does not provide this project's transient hardware-fault class.

**Intel Lab.** The downloaded environmental readings do not include verified fault labels; they are deployment measurements collected using the Intel Berkeley Research Lab sensor network [9]. Prior work has also used this dataset for sensor-fault detection [10]. Temperature readings are aggregated by mote and five-minute bin; a fixed schedule injects bias, drift, stuck-at and spike episodes. Five-fold mote-grouped evaluation excludes mote ID and time from predictors. Accuracy is 0.92727 and macro-F1 is 0.59798 (95% mote-cluster bootstrap interval 0.58109–0.61590). Stuck-fault precision is 0.0063, illustrating why accuracy alone is misleading. The label and fault magnitudes are synthetic. This is not naturally labeled fault recognition, communication-fault evaluation or recovery testing. Full preprocessing, exclusions, class metrics and source checksums are in `docs/public_dataset_validation.md` and `results/public_dataset_validation/`.

## 4. Results

### 4.1 Packet delivery

| Scenario | Diagnosis-aware PDR, mean (SD) | Generic-quarantine PDR, mean (SD) | Paired difference (95% CI) | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| S1 dense detectable | 0.99943 (0.00150) | 0.99942 (0.00150) | +0.00001 [−0.00001, +0.00003] | 0.326 | 0.352 |
| S2 dense silent | 0.99840 (0.00275) | 0.99522 (0.01425) | +0.00318 [−0.00151, +0.00786] | 0.176 | 0.352 |
| S3 sparse silent | 0.95851 (0.05211) | 0.93768 (0.07249) | +0.02083 [+0.00372, +0.03795] | 0.0188 | 0.0803 |
| S4 sparse transient | 0.95206 (0.05608) | 0.93367 (0.07495) | +0.01840 [+0.00368, +0.03311] | 0.0161 | 0.0803 |
| S5 bottleneck transient | 0.99873 (0.00282) | 0.99443 (0.01049) | +0.00430 [+0.00055, +0.00806] | 0.0262 | 0.0803 |
| S6 sparse transient with crashes | 0.84636 (0.06032) | 0.82892 (0.06702) | +0.01745 [+0.00692, +0.02798] | 0.00204 | 0.0122 |

Only S6 remains below 0.05 after Holm correction. S3–S5 have positive unadjusted intervals but do not survive correction. This supports a conditional signal in the crash stress scenario, not a general benefit across conditions. The diagnosis-aware policy's PDR in S6 is 0.846, so the result also does not imply satisfactory service reliability.

No-recovery mean PDR ranged from 0.696 in S6 to 0.931 in S1; blind rerouting ranged from 0.747 in S6 to 0.999 in S1. In S5, diagnosis-aware PDR was 0.9987 versus 0.8156 for blind rerouting, but the pre-specified primary contrast is against generic quarantine, and the S5 paired primary result did not survive Holm correction. S6 route availability was 0.8953 for every policy. This distinction matters: prediction-conditioned costs and fallback can affect delivery without changing physical graph reachability.

### 4.2 Crash recovery, energy and other outcomes

Using the three-consecutive-round threshold, mean recovery time in S6 was 26.0 rounds for diagnosis-aware and generic-quarantine policies: neither met the criterion within the observation window. Blind rerouting and no recovery averaged 25.13 rounds (SD 4.75), reflecting that these runs sometimes crossed the threshold before censoring; this is not evidence that their overall PDR was higher. The old single-round recovery values are superseded and must not be cited.

Diagnosis-aware S6 energy was 3.983 abstract units per offered packet versus 3.832 for generic quarantine and 3.325 for blind rerouting; energy per delivered packet was 4.731 versus 4.648 and 4.497, respectively. These model-dependent costs are not measured energy. In S3–S5, diagnosis-aware maximum per-node load exceeded the blind-reroute comparator, consistent with traffic concentration on fewer eligible relays. The complete mean, SD and paired metric tables are in `results/recovery_policy_v4/summary.csv` and `paired_comparisons.csv`.

### 4.3 Classifier behavior in the packet study

Across the 30 seeds, mean node-held-out fault recall was 0.7679 (95% t interval 0.7535–0.7824); fault precision was 0.8805. Blackhole precision and recall were 0.9612 and 0.9588. These are in-simulator class metrics over a small number of nodes, not external classifier-validation estimates. The detector still misses some faults, which constrains any policy depending on those predictions.

### 4.4 Public classifier-only checks

The WSN-DS and Intel Lab results are reported in Section 3.4 to keep their scope explicit. They should not be pooled with the packet-policy study: data sources, labels, experimental units and outcomes differ. High WSN-DS scores do not validate field behavior; the low Intel macro-F1 and near-zero stuck precision are material negative results.

## 5. Discussion

The only multiplicity-adjusted primary PDR difference appears in the sparse transient-fault scenario after node crashes. The observation is compatible with the intended mechanism: permanent treatment of confirmed blackholes and a less restrictive path for transient faults can preserve routes when network conditions change. However, the current comparator does not isolate diagnosis itself, and the model directly assigns forwarding probabilities from ground-truth labels. The result may instead reflect the whole policy bundle and the chosen acknowledgment/fallback assumptions. A diagnosis-only ablation and independent simulator reproduction are required before making a method-level claim.

The literature narrows the contribution further. Fault-node classification and reuse were studied before this project [2]; recent work pairs ML detection with a recovery optimizer [3]; and a recent controlled HIL system reports a 50-node testbed with packet, recovery and measured energy outcomes [4]. A recent large-scale RL self-healing framework remains simulation-based [5]. Studies using WSN-DS report very high classification performance on the same NS-2/LEACH benchmark [6], so this project's grouped split is a validation-protocol choice, not evidence that the dataset represents real network traffic. The supplied literature audit in `docs/literature_review.md` records all 16 unique user-supplied works and identifies sources whose full texts could not be obtained.

The most defensible current contribution is therefore a transparent, paired stress-test protocol and a conditional result in a custom model, alongside explicit evidence that public-data classifier scores do not establish end-to-end self-healing. Whether the result generalizes to a standard stack, realistic radio behavior, or field deployment is unknown.

## 6. Limitations

1. The packet simulator is custom and abstracts MAC, queues, contention, collisions, control traffic, link-layer acknowledgments, routing protocol timers and physical radio behavior.
2. Labels determine forwarding success directly. This makes classification and forwarding behavior dependent by construction and does not model an adversary or hardware failure process independently.
3. Silent blackhole drops are acknowledged. Other acknowledgment scopes and protocol stacks may make attacks detectable or cause different retry behavior.
4. The diagnosis-aware and generic-quarantine policies are not matched on every action; the primary difference is a combined policy-package contrast, not diagnosis alone.
5. The bottleneck topology is deliberately constructed and cannot support a population-level claim.
6. There are 30 seeds and only four nodes in each anomalous class at 20 nodes. The 95% paired t intervals are not multiplicity-adjusted; only p-values are Holm-adjusted. The one significant result is conditional on this frozen design.
7. Abstract energy has no calibrated radio model or finite battery. Maximum per-node energy is a load proxy, not network lifetime. No lifetime claim is made.
8. Public data validate classification only. WSN-DS is simulated, while Intel Lab has injected labels and weak current performance on several fault classes. Neither validates packet recovery.
9. No ESP32 or power meter is available. Model size, RAM, inference time and energy per inference remain unmeasured future work.
10. The literature search is not exhaustive. Several supplied publisher full texts were inaccessible; the review document distinguishes those from fully reviewed open papers. Novelty claims need a final full-text comparison and reference audit.

## 7. Conclusion

In this paired, custom simulation, diagnosis-aware recovery showed a positive Holm-adjusted PDR difference over diagnosis-blind quarantine in one of six scenarios: sparse connectivity with transient faults and 15% normal-node crashes. The other five primary contrasts did not survive correction. The observed S6 PDR remained 0.846, and recovery to the pre-crash threshold was censored for diagnosis-aware recovery. This is a conditional simulation finding, not proof of general diagnosis benefit. WSN-DS and Intel Lab support separate classifier-only evaluations; the latter includes synthetic injected labels and low macro-F1. There is no calibrated network lifetime, hardware measurement, or field recovery result. The work is suitable as a research prototype and a draft for supervisor review, but further ablations, independently reproducible networking validation, full-text literature review and final reference verification are needed before journal or conference submission.

## References cited in this draft

1. I. Almomani, B. Al-Kasasbeh and M. Al-Akhras, “WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks,” *Journal of Sensors*, 2016, Article 4731953. DOI: [10.1155/2016/4731953](https://doi.org/10.1155/2016/4731953).
2. S. Chanak and D. Banerjee, “Fuzzy rule-based faulty node classification and management scheme for large scale wireless sensor networks,” *Expert Systems with Applications*, 45, 307–321, 2016. DOI: [10.1016/j.eswa.2015.09.040](https://doi.org/10.1016/j.eswa.2015.09.040).
3. A. Alauthman and A. Al-Hyari, “Intelligent Fault Detection and Self-Healing Mechanisms in Wireless Sensor Networks Using Machine Learning and Flying Fox Optimization,” *Computers*, 14(6), 233, 2025. DOI: [10.3390/computers14060233](https://doi.org/10.3390/computers14060233).
4. H. Alghawli et al., “A Q-learning-assisted self-healing wireless sensor network for industrial carbon capture monitoring: a controlled HIL study,” *Frontiers in Computer Science*, 8, 1914381, 2026. DOI: [10.3389/fcomp.2026.1914381](https://doi.org/10.3389/fcomp.2026.1914381).
5. T. Timuçin, “Reinforcement Learning-Based Self-Healing Framework in IoT Sensor Networks,” *Gazi University Journal of Science Part A*, 13(2), 764–783, 2026. DOI: [10.54287/gujsa.1914900](https://doi.org/10.54287/gujsa.1914900).
6. M. A. Talukder et al., “MLSTL-WSN: machine learning-based intrusion detection using SMOTETomek in WSNs,” *International Journal of Information Security*, 23, 2139–2158, 2024. DOI: [10.1007/s10207-024-00833-z](https://doi.org/10.1007/s10207-024-00833-z).
7. P. S. Shakunt and S. K. Udgata, “Machine learning-based fault detection scheme for IoT-enabled WSNs,” *International Journal of Sensor Networks*, 46(2), 61–72, 2024. DOI: [10.1504/IJSNET.2024.141771](https://doi.org/10.1504/IJSNET.2024.141771).
8. N. Ijaz, S. U. Jan, M. N. Hasan and I. Koo, “A Hybrid LATAM and Few-Shot Learning Framework for Fault Diagnosis in Wireless Sensor Networks,” *IEEE Sensors Journal*, 25(23), 43102–43116, 2025. DOI: [10.1109/JSEN.2025.3620491](https://doi.org/10.1109/JSEN.2025.3620491).
9. A. Mainwaring, J. Polastre, R. Szewczyk, D. Culler and J. Anderson, “Wireless Sensor Networks for Habitat Monitoring,” in *Proceedings of the 1st ACM International Workshop on Wireless Sensor Networks and Applications*, 2002, pp. 88–97. DOI: [10.1145/570738.570751](https://doi.org/10.1145/570738.570751). Dataset information: [Intel Berkeley Research Lab data](https://db.csail.mit.edu/labdata/labdata.html).
10. S. Kang et al., “Spatial-Temporal Correlative Fault Detection in Wireless Sensor Networks,” *The Scientific World Journal*, 2014, Article 709390. DOI: [10.1155/2014/709390](https://doi.org/10.1155/2014/709390).

## Reproduction and artifacts

From the repository root with the project's `.venv`:

```powershell
.\.venv\Scripts\python.exe -m experiments.recovery_policy_v4
```

The run uses seeds 301–330 and writes per-seed results, means/standard deviations, paired contrasts and classifier summaries to `results/recovery_policy_v4/`. The exact interpreter package versions used for this rerun are recorded in `requirements.txt`. Public dataset checks are independently rerun with `python -m experiments.public_dataset_validation`. The data files remain local under `data/raw/`; the repository's ignore rules exclude raw data and generated results, so these results are not automatically included in a source archive. Before external review, prepare a reproducibility bundle containing the required inputs, checksums, environment lock and generated CSVs.
