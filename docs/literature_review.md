# Literature review and source audit (working document)

## Initial synthesis

The defensible gap is narrower than the draft previously implied. Fault diagnosis, routing recovery, fault-node reuse, WSN intrusion datasets, and hardware-in-the-loop (HIL) self-healing have all been studied. In particular, Chanak and Banerjee (2016) already propose classifying, managing, and reusing faulty WSN nodes, so quarantine/readmission by itself is not novel. Alghawli et al. (2026) report a controlled HIL self-healing network with fault/load/attenuation stress, recovery time, PDR, latency and power metrics. Accordingly, the project must not claim that measured self-healing cost or hardware-in-the-loop results are absent from prior work.

The more specific candidate question is whether node-grouped out-of-fold, multiclass diagnosis changes the trade-off between temporary degradation and permanent malicious forwarding when a route-critical but recoverable relay is quarantined and later readmitted. That candidate is not established as novel: the closest reuse work and HIL competitor must be compared in a full-text matrix, including their recovery decision rules, topology, labels, baseline policies, energy measurement and experimental unit. Our current v2 results are negative versus blind rerouting and far too low in PDR to support a contribution claim.

## Closest prior work

### Fault classification and reuse

Chanak and Banerjee's *Fuzzy rule-based faulty node classification and management scheme for large scale wireless sensor networks* (Expert Systems with Applications 45, 307–321, 2016; DOI [10.1016/j.eswa.2015.09.040](https://doi.org/10.1016/j.eswa.2015.09.040)) explicitly classifies nodes by fault status and reuses faulty nodes in routing to improve QoS and network lifetime. This directly overlaps with the re-admission idea. Our distinction, if supported after detailed comparison, would have to be the use of learned node-held-out multiclass predictions, separate permanent blackhole isolation, and a packet-level paired evaluation under a route-critical transient fault. These components are not yet sufficient to establish novelty.

Chanak, Banerjee and Sherratt's *Mobile sink based fault diagnosis scheme for wireless sensor networks* (Journal of Systems and Software 119, 45–57, 2016; DOI [10.1016/j.jss.2016.05.041](https://doi.org/10.1016/j.jss.2016.05.041)) evaluates mobile-sink diagnosis, with simulation and experimental work. It should be compared for diagnostics, recovery action and network burden.

### Self-healing and hardware evidence

Alghawli et al.'s *A Q-learning-assisted self-healing wireless sensor network for industrial carbon capture monitoring: a controlled HIL study* (Frontiers in Computer Science 8, 1914381, 2026; DOI [10.3389/fcomp.2026.1914381](https://doi.org/10.3389/fcomp.2026.1914381)) is a close recent competitor. The article describes a 50-node CC2650/ARM Cortex-M3 HIL setup, Raspberry Pi gateway, routing/relay emulation, an Otii Arc Pro energy instrument, and stress tests including node failure, traffic surge, and signal attenuation. It reports PDR, recovery time, latency and power-related outcomes. Our planned single ESP32 experiment would be a small independent reproduction/feasibility measurement, not the first hardware-cost or HIL demonstration. We need verify details and compare measured versus emulated parts before citing it as a direct apples-to-apples competitor.

### Datasets and what they validate

Almomani, Al-Kasasbeh and AL-Akhras (2016), *WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks* (Journal of Sensors, 2016, article 4731953; DOI [10.1155/2016/4731953](https://doi.org/10.1155/2016/4731953)), describe 23 features generated from NS-2 simulation using LEACH and normal/blackhole/grayhole/flooding/scheduling classes. It can support attack-classification experiments but is not real deployment traffic and does not label transient sensor hardware faults.

The [Intel Berkeley Lab page](https://db.csail.mit.edu/labdata/labdata.html) describes 54 Mica2Dot motes, topology and environmental readings, but no validated fault annotations. Kang et al. (2014), *Spatial-Temporal Correlative Fault Detection in Wireless Sensor Networks* (DOI [10.1155/2014/709390](https://doi.org/10.1155/2014/709390)), use Intel Lab data for fault detection and select a limited time window with preprocessing. de Bruijn et al. (2016), *Benchmark Datasets for Fault Detection and Classification in Sensor Data* (SENSORNETS, 185–195; DOI [10.5220/0005637901850195](https://doi.org/10.5220/0005637901850195)), describe derived benchmark subsets based on real sensor recordings with injected fault types. Thus, our Intel Lab evaluation must be called fault-injection evaluation, with the injection algorithm and labels disclosed; it cannot be presented as ground-truth naturally occurring fault detection.

## Candidate papers for the 15–20 source review

The items below form the tracked review set. “Primary pass” means the publisher/repository record or full text was opened and the title/metadata/abstract or relevant methods were checked during this initial pass. “Metadata/abstract” means identified as a relevant source but still requires full-text inspection and complete reference verification before relying on details in the manuscript.

1. Winter et al. (2012), *RPL: IPv6 Routing Protocol for Low-Power and Lossy Networks*, RFC 6550. [IETF RFC](https://www.rfc-editor.org/rfc/rfc6550). Primary pass; protocol background.
2. Heinzelman, Chandrakasan & Balakrishnan (2000), *Energy-efficient communication protocol for wireless microsensor networks*, HICSS. DOI [10.1109/HICSS.2000.926982](https://doi.org/10.1109/HICSS.2000.926982). Metadata pass; energy/routing background.
3. Almomani et al. (2016), *WSN-DS*. DOI [10.1155/2016/4731953](https://doi.org/10.1155/2016/4731953). Primary pass; synthetic attack dataset.
4. Mainwaring et al. (2002), *Wireless Sensor Networks for Habitat Monitoring*, WSNA. DOI [10.1145/570738.570751](https://doi.org/10.1145/570738.570751). Metadata pass; deployment-data provenance context.
5. de Bruijn et al. (2016), *Benchmark Datasets for Fault Detection and Classification in Sensor Data*. DOI [10.5220/0005637901850195](https://doi.org/10.5220/0005637901850195). Primary pass; real-base/injected-fault benchmark design.
6. Kang et al. (2014), *Spatial-Temporal Correlative Fault Detection in Wireless Sensor Networks*. DOI [10.1155/2014/709390](https://doi.org/10.1155/2014/709390). Primary pass; Intel data and fault detection.
7. Chanak & Banerjee (2016), *Fuzzy rule-based faulty node classification and management scheme for large scale wireless sensor networks*. DOI [10.1016/j.eswa.2015.09.040](https://doi.org/10.1016/j.eswa.2015.09.040). Primary pass; closest fault reuse overlap.
8. Chanak, Banerjee & Sherratt (2016), *Mobile sink based fault diagnosis scheme for wireless sensor networks*. DOI [10.1016/j.jss.2016.05.041](https://doi.org/10.1016/j.jss.2016.05.041). Primary pass; mobile-sink diagnosis.
9. Mehmood et al. (2018), *A Survey on Proactive, Active and Passive Fault Diagnosis Protocols for WSNs: Network Operation Perspective*, Sensors 18(6), 1787. DOI [10.3390/s18061787](https://doi.org/10.3390/s18061787). Primary pass; diagnosis taxonomy.
10. Xie et al. (2011), *Anomaly detection in wireless sensor networks: a survey*, Journal of Network and Computer Applications 34(4), 1302–1325. DOI [10.1016/j.jnca.2011.03.004](https://doi.org/10.1016/j.jnca.2011.03.004). Metadata pass; anomaly detection background.
11. Butun, Morgera & Sankar (2014), *A survey of intrusion detection systems in wireless sensor networks*, IEEE Communications Surveys & Tutorials 16(1), 266–282. DOI [10.1109/SURV.2013.050113.00191](https://doi.org/10.1109/SURV.2013.050113.00191). Metadata pass; IDS background.
12. Alansari et al. (2022), *A systematic review of routing attacks detection in wireless sensor networks*, PeerJ Computer Science 8, e1135. DOI [10.7717/peerj-cs.1135](https://doi.org/10.7717/peerj-cs.1135). Metadata pass; routing-attack detection context.
13. Karlof & Wagner (2003), *Secure routing in wireless sensor networks: attacks and countermeasures*, Ad Hoc Networks 1(2–3), 293–315. DOI [10.1016/S1570-8705(03)00008-8](https://doi.org/10.1016/S1570-8705(03)00008-8). Metadata pass; routing threats.
14. Tripathi, Gaur & Laxmi (2013), *Comparing the Impact of Black Hole and Gray Hole Attack on LEACH in WSN*, Procedia Computer Science 19, 1101–1107. DOI [10.1016/j.procs.2013.06.155](https://doi.org/10.1016/j.procs.2013.06.155). Metadata pass; attack behavior in LEACH.
15. Moreira Sá de Souza, Vogt & Beigl (2007), *A survey on fault tolerance in wireless sensor networks*, KIT internal report 2007/22. DOI [10.5445/IR/1000007250](https://doi.org/10.5445/IR/1000007250). Metadata/abstract pass; fault-tolerance survey.
16. Paradis & Han (2007), *A Survey of Fault Management in Wireless Sensor Networks*, Journal of Network and Systems Management 15(2), 171–190. DOI [10.1007/s10922-007-9062-0](https://doi.org/10.1007/s10922-007-9062-0). Metadata/abstract pass; detection-diagnosis-recovery framing. The earlier draft incorrectly attributed this DOI/title to Yu, Mokhtar & Merabti; that mismatched record has been corrected.
17. Alghawli et al. (2026), *A Q-learning-assisted self-healing wireless sensor network for industrial carbon capture monitoring: a controlled HIL study*, Frontiers in Computer Science 8, 1914381. DOI [10.3389/fcomp.2026.1914381](https://doi.org/10.3389/fcomp.2026.1914381). Primary pass; closest recent HIL/self-healing comparison.
18. Talayoglu, Vande Velde & da Silva (2025), *Lightweight AI for Sensor Fault Monitoring*, Electronics 14(22), 4532. DOI [10.3390/electronics14224532](https://doi.org/10.3390/electronics14224532). Primary abstract pass; model cost and injected sensor fault limitations.

The data experiments described in `docs/public_dataset_validation.md` are complete for the downloaded files. This is an 18-item candidate review set, not a claim that all 18 have been read in full. Before submission, read the full text of all items used for novelty claims, inspect Alghawli's methods and supplementary material, and verify every in-text citation against the final bibliography. Remove sources that do not directly support a statement.

## Manuscript changes required after review

- Replace broad novelty claims with a bounded comparison to the fault-reuse literature and Alghawli et al. (2026).
- State separately what WSN-DS and Intel Lab can and cannot validate.
- Do not call a single ESP32 test the first on-device measurement in the field.
- Treat the current result as no demonstrated diagnosis-specific PDR benefit; report the low service PDR as a design/calibration limitation.
- Distinguish this study's synthetic packet-level model from radio/HIL measurements and from the classifier dataset evaluation.
