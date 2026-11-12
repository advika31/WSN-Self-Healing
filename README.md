# WSN Self-Healing Research Prototype

A Python prototype for detecting normal, faulty, and blackhole sensor nodes in a simulated wireless sensor network (WSN), then applying diagnosis and recovery actions. The project includes a Random Forest classifier, a node-wise evaluation, robustness experiments with noisy measurements, and a fixed rule-based reference classifier.

## Setup

Run commands from the project root in Windows PowerShell. Create and activate the virtual environment if needed, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the interpreter directly as shown below, or use `Set-ExecutionPolicy -Scope Process RemoteSigned` for the current PowerShell process.

## Experiments

The experiments are Python modules; use `-m` from the project root so imports resolve correctly.

### Multiple-seed node-wise evaluation

```powershell
python -m experiments.multi_seed_nodewise_evaluation
```

Runs seeds 42–51. Each run simulates 20 nodes over 30 rounds, extracts three-reading rolling features, and holds out whole nodes. The held-out set contains four normal nodes, one fault node, and one blackhole node; all three labels are checked in both training and test partitions. Random Forest uses 100 trees, the simulation seed as `random_state`, and balanced class weights. Results are written to `results/multi_seed_nodewise/`.

### Noisy-measurement robustness evaluation

```powershell
python -m experiments.noisy_robustness_evaluation
```

Runs the same seeds and node splits at two additive, class-independent noise levels. The Gaussian standard deviations are:

| Level | RSSI (dBm) | Packet delivery ratio | Delay (ms) |
| --- | ---: | ---: | ---: |
| Moderate | 10 | 0.18 | 30 |
| Strong | 18 | 0.30 | 60 |

Packet delivery ratio is clipped to [0, 1], packet loss is recalculated, and delay is clipped to at least 1 ms. Features are extracted after noise is added. Results and the per-seed generated data are written to `results/noisy_robustness/`.

### Rule-based comparison

```powershell
python -m experiments.rule_based_baseline_comparison
```

This reuses the saved noisy feature sets, seeds, and node splits. Run the noisy-measurement experiment first if its generated files are missing. Fixed rules use rolling means: packet delivery below 0.20 predicts blackhole; otherwise RSSI below −70 dBm or delay above 65 ms predicts fault; all remaining records are predicted normal. The rules are reference thresholds based on the simulator ranges and are not tuned against the held-out nodes. Results are written to `results/baseline_comparison/`.

All experiment summaries include per-run accuracy, macro precision, macro recall, and macro F1; summary files report the mean and sample standard deviation across ten seeds. Macro metrics weight the three classes equally. The directories also contain `per_run_metrics.csv` and `mean_std_metrics.csv` files.

## Recorded results

These are the results from the current ten-seed runs in this project:

| Experiment | Method / noise | Accuracy mean ± std | Macro precision | Macro recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Node-wise, no added noise | Random Forest | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Moderate noise | Random Forest | 0.9933 ± 0.0057 | 0.9926 | 0.9875 | 0.9899 |
| Moderate noise | Rule-based | 0.9094 ± 0.0399 | 0.8926 | 0.9314 | 0.8999 |
| Strong noise | Random Forest | 0.9622 ± 0.0154 | 0.9540 | 0.9378 | 0.9440 |
| Strong noise | Rule-based | 0.7078 ± 0.0745 | 0.7549 | 0.7664 | 0.7074 |

The rule-based method has greater recall than precision under moderate noise, while strong noise lowers all its metrics substantially. The Random Forest performs better on these simulated runs. These results describe only this simulator and noise setup.

## End-to-end demo

The existing diagnosis and recovery demo predicts each node from the latest row in the existing processed feature file, applies recovery decisions, and selects a route:

```powershell
python main.py
```

It expects `data/processed/wsn_features.csv`, `results/anomaly_detector.pkl`, and `results/label_encoder.pkl` to exist. The node-wise experiments are separate and do not replace these files.

## Reproducibility and limitations

- The simulator and classifier receive explicit seeds. The node split is independently seeded and stratified by node label, so both partitions contain all three classes.
- Each node has one fixed behavior for all 30 rounds. The node-level assignment is held constant across seeds; seeds change simulated measurements and the selected held-out nodes.
- Each held-out set has 180 records: 120 normal, 30 fault, and 30 blackhole. Macro metrics are included because the held-out records are not class-balanced.
- Rolling features use the current reading plus up to the two preceding readings from the same node. The first reading's rolling standard deviation is set to zero.
- The simulated classes have deliberately distinct nominal behavior. Added Gaussian measurement noise explores only two selected noise levels; it does not model every source of variation in a deployed WSN.
- The perfect no-noise score and strong noisy scores are not evidence of real-world performance. Validate on independent, real sensor data before making deployment claims.
- Results and datasets under `results/` and `data/` are ignored by Git in this project. Re-run the modules above to regenerate the experiment artifacts.

## Suggested research reporting

### Topology-aware recovery proxy and manuscript draft

Run the new recovery experiment with:

```powershell
python -m experiments.recovery_network_evaluation
```

It writes separate outputs to `results/recovery_v2/` and preserves the original recovery CSVs. This is a one-relay topology proxy with documented assumptions, not a packet-level network simulator. See `docs/recovery_v2_assumptions.md` for metric definitions and limitations. A working, literature-cited manuscript draft is in `docs/paper_draft.md`.

### Packet-forwarding recovery evaluation

```powershell
python -m unittest discover -s tests -v
python -m experiments.packet_recovery_evaluation
```

The packet-level experiment writes its independent outputs to `results/recovery_v3_packet/`. Across ten seeds, mean packet delivery ratio was 0.8802 for both recovery policies and 0.8705 for no recovery. This suggests rerouting may help in the chosen synthetic model, while the identical delivery result for generic and diagnosis-aware rerouting does not establish a diagnosis-specific advantage. Results and assumptions are detailed in `docs/packet_recovery_assumptions.md`.

### Packet recovery sensitivity study

```powershell
python -m experiments.packet_recovery_sensitivity
python -m experiments.plot_recovery_sensitivity
```

This runs a shared baseline and nine one-factor alternatives across 30 seeds, changing radio range, network size, or fault/blackhole forwarding probability. Outputs checkpoint to `results/recovery_sensitivity_v1/`; see `docs/recovery_sensitivity_assumptions.md` for the frozen design, paired analysis, and findings. The paper-ready PDR figure is `results/recovery_sensitivity_v1/figures/packet_delivery_sensitivity.png`.

Describe the work as a simulation-based evaluation of node-wise generalization under the stated class assignments and noise distributions. Report all ten seeds, the macro metrics with mean and sample standard deviation, the rule thresholds, and the class counts in the held-out set. State that real-world WSN validation remains future work; do not present the simulated percentages as expected field performance.

### Day 1 transient-fault quarantine experiment

```powershell
python -m unittest discover -s tests -v
python -m experiments.transient_fault_quarantine
```

This isolated experiment uses two controlled sparse/bottleneck graphs to compare permanent generic exclusion with fault quarantine and re-admission after two consecutive normal diagnoses. Node 8 is hard-isolated as a perfect-blackhole-containment control in every strategy; this is a simplifying assumption, not a detector result. The fault and diagnosis traces are synthetic inputs, and the module does not use the learned classifier. Outputs go to `results/transient_fault_quarantine_day1/`. See `docs/transient_fault_quarantine_day1.md` for metric definitions and limitations. Network lifetime is omitted because the current model has no defensible battery-depletion model.

### Recovery policy v2

```powershell
python -m unittest discover -s tests -v
python -m experiments.recovery_policy_v2
```

This newer, separate study uses node-held-out predictions, two fault-load settings, sparse and single-bottleneck graphs, class-specific actions, an explicit abstract battery budget, and paired seed-level comparisons. Its primary contrast is diagnosis-aware recovery versus blind rerouting on PDR; permanent generic exclusion is a secondary comparator. Metrics and censoring rules are defined in `docs/recovery_policy_v2.md`; outputs go to `results/recovery_policy_v2/`. Treat the battery/lifetime outputs as uncalibrated proxies.

This is the preserved initial run. Its all-normal control revealed that the 1,000-unit abstract battery budget caused service failure even without anomalous nodes. Do not use v2 alone to make claims about fault diagnosis or pool it with v3. See `docs/recovery_policy_v2.md` for the original design.

### Recovery policy v3 (current calibrated-control run)

```powershell
python -m unittest discover -s tests -v
python -m experiments.recovery_policy_v3
```

This preserves v2 and writes independent results to `results/recovery_policy_v3/`. It doubles the abstract energy budget and adds a no-fault control, while retaining the 30 paired seeds, sparse and route-critical bottleneck topologies, 20%/40% fault loads, transient fault quarantine/readmission, permanent predicted-blackhole isolation and alerts. It also counts alternate-route fallbacks after link retry exhaustion. The primary paired PDR differences (diagnosis-aware minus blind rerouting) were −0.3471 (95% CI −0.3735 to −0.3208), −0.3710 (−0.3925 to −0.3495), −0.2861 (−0.3655 to −0.2067), and −0.5757 (−0.6366 to −0.5149) for bottleneck/20%, bottleneck/40%, sparse/20%, and sparse/40%, respectively. The healthy control achieved mean PDR 0.9998. These are synthetic stress-test results and show no diagnosis-aware delivery advantage. Energy and lifetime remain uncalibrated proxies; see `docs/recovery_policy_v3.md`.

### Dataset and hardware follow-up

The WSN-DS and Intel Lab files have been downloaded; the ESP32 board and power meter are unavailable. See `docs/datasets_and_hardware_plan.md` for data handling, dataset limitations, and the device-measurement plan. The source audit and 18-paper candidate review set are tracked in `docs/literature_review.md`; the manuscript still needs full-text review and final bibliography checks before submission.

The WSN-DS and Intel Lab files have since been placed under `data/raw/`. Run their grouped, classifier-only evaluation with:

```powershell
python -m experiments.public_dataset_validation
```

WSN-DS is treated as simulated attack traffic; Intel Lab is used only as real environmental readings with synthetic injected temperature faults. Neither provides a public-data packet-recovery result. Outputs and detailed data provenance/limitations are documented in `docs/public_dataset_validation.md` and saved to `results/public_dataset_validation/`. ESP32 measurements remain future work because the board and power meter are unavailable.
