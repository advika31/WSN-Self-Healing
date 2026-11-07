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
