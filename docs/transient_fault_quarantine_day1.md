# Day 1: Transient-fault quarantine and re-admission

## Question and scope

This separate experiment asks whether a diagnosis-aware recovery rule can help when a relay is temporarily faulty and later recovers. It compares `no_recovery`, `generic_permanent`, and `diagnosis_quarantine` on two explicitly constructed graphs (`sparse` and `bottleneck`). It leaves the existing packet-recovery and sensitivity experiments and their result directories untouched.

The transient fault is on relay 1 during rounds 6–11. A declared synthetic diagnosis trace reports `fault` in those rounds and `normal` otherwise. Diagnosis-aware recovery quarantines relay 1 while fault is reported, then requires two consecutive normal diagnoses before re-admitting it. Generic rerouting permanently excludes a node after its first fault diagnosis. The no-recovery strategy does not react to the transient diagnosis. All three strategies hard-exclude node 8, the blackhole, as an explicit perfect-containment control. That oracle-like control isolates the transient-fault question; it is not a claim that blackholes can be perfectly detected in practice.

Both topologies have a direct preferred route through relay 1 and a longer bypass. The sparse graph's bypass uses relays 2, 3, and 4; the bottleneck graph's bypass uses relays 2 and 3, leaving relay 1 as the dominant short route. Thus they differ in path redundancy and detour cost. Edges are fixed, bidirectional links with listed synthetic distances. This does not use the geometric radio model or the classifier. The diagnosis trace is idealized so the policy mechanism can be checked first; Day 1 results must not be presented as classifier performance.

## Packet and metric definitions

Each of 30 seeds (101–130), two topologies, and 30 rounds offers 100 packets from source 0. Each hop receives up to three attempts. Successful hop probability is 0.97 radio success times 0.98 normal forwarding, or 0.15 forwarding for relay 1 during its fault rounds. Shared seed-keyed random draws make the strategies comparable on matching packet, round, and directed edge attempts.

- **PDR:** delivered packets / all offered packets. An unreachable route counts as a failure.
- **Delay:** accumulated attempt delay proxy, `2 + 0.08 × edge distance` per attempt, in model-defined milliseconds.
- **Abstract energy:** `1 + 0.0002 × distance² + 0.5` units per attempt, including failed attempts. Not joules or battery measurements.
- **Route availability:** fraction of offered packets for which the policy has a route before transmission.
- **Fallback:** fraction using the emergency route when a policy's exclusions leave no route. Emergency routing can use relay 1 but still excludes node 8.
- **Recovery time:** rounds from the end of the declared fault interval (round 11) to first re-admission. This is a policy timing measure, not a physical network repair time.
- **Network lifetime:** omitted. The simulator has no validated finite-battery depletion or node-death model from which to define lifetime defensibly.

The seed is the experimental unit for summary standard deviations. Results are synthetic and exploratory; topology, fault schedule, detection quality, probabilities, energy, and delay are selected assumptions. There is no MAC, contention, queue, control traffic, radio calibration, classifier uncertainty, hardware validation, or dataset validation in this Day 1 model.

## Reproduction

From the repository root in PowerShell:

```powershell
python -m unittest discover -s tests -v
python -m experiments.transient_fault_quarantine
```

The new module writes only to `results/transient_fault_quarantine_day1/` (`packet_results.csv`, `per_seed_metrics.csv`, and `summary.csv`).
