# Recovery policy study v4 (consolidated)

**Status:** supersedes the interpretation of the Day-1 quarantine run, recovery-policy v2 and v3.
Those runs remain in `results/` for audit but should not be reported as findings:
v2/v3 give diagnosis-aware routing no last-resort fallback, so it drops packets that blind
rerouting delivers when its exclusions disconnect a hand-built single-bridge topology
(diagnosis-aware route availability 0.04-0.11 vs 0.55-0.9 for blind). That measures a missing
fallback, not the value of diagnosis. Day-1 uses scripted fault/diagnosis traces, not classifier output.

Reproduce: `python -m experiments.recovery_policy_v4` (about 2 minutes, 30 seeds 301-330),
`python -m unittest discover -s tests -v`.

## Design
* **Policies (6):** `no_recovery` (static route); `blind_reroute` (reactive reroute after a failed hop);
  `generic_exclusion` (exclude nodes flagged non-normal this round); `generic_quarantine`
  (ablation: same 2-round hysteresis/readmission as the diagnosis policy but cannot separate fault from
  attack); `diagnosis_aware` (blackhole confirmed by 2 of last 3 predictions -> permanent isolation +
  alert; fault -> quarantine, readmitted after 2 clean rounds, usable at 1.5x cost when no clean route);
  `oracle_diagnosis` (true labels / true degraded state; upper bound for any diagnosis).
* **Fairness:** all policies except `no_recovery` share reactive rerouting and the same unconstrained
  last-resort fallback. Pre-drawn link and drop uniforms are shared across policies per packet (paired).
* **Blackhole semantics:** `silent` = the blackhole acknowledges the hop and then drops the packet, so hop
  retries cannot see it; `detectable` = the drop looks like a failed hop (the original repo assumption).
* **Faults:** `persistent` (forwards with p=0.62) or `transient` (p=0.30 in degraded rounds, 0.95 otherwise,
  using the simulator's own latent state via `record_latent=True`).
* **Diagnosis:** node-grouped out-of-fold Random Forest predictions (no node predicted by a model trained
  on it), 20 nodes, 40 rounds, 4 packets per source-round, 30 seeds. Features exclude id, x, y, round, label.
* **Scenarios:** S1 dense/detectable/persistent; S2 dense/silent; S3 sparse(R=36)/silent; S4 sparse/silent/
  transient; S5 constructed bottleneck (4 inner relays: 1 blackhole, 2 fault, 1 normal carry 16 outer nodes)
  /silent/transient; S6 S4 + 15% of normal nodes crash at round 15.
* **Statistics:** seed is the experimental unit; paired t 95% CI; Holm correction within a family. Primary
  family (pre-specified): `diagnosis_aware - generic_quarantine` on PDR, one test per scenario.
  Everything else is secondary/exploratory.

## Results (means over 30 seeds)
The v4 code was rerun after fixing the S6 denominator so crashed sources remain offered demand and replacing the single-round recovery trigger with a three-consecutive-round criterion. The previous tables and interpretation are superseded. The prior output snapshot is retained locally in `results/recovery_policy_v4_before_load_fix/` for audit.

| Scenario | no recovery | blind reroute | generic exclusion | generic quarantine | diagnosis-aware | oracle |
|---|---:|---:|---:|---:|---:|---:|
| S1 dense, detectable | 0.9306 | 0.9994 | 0.9994 | 0.9994 | 0.9994 | 0.9994 |
| S2 dense, silent | 0.9241 | 0.9384 | 0.9946 | 0.9952 | 0.9984 | 0.9994 |
| S3 sparse, silent | 0.8235 | 0.8468 | 0.9412 | 0.9377 | 0.9585 | 0.9616 |
| S4 sparse, silent, transient | 0.8110 | 0.8415 | 0.9374 | 0.9337 | 0.9521 | 0.9554 |
| S5 bottleneck, silent, transient | 0.7454 | 0.8156 | 0.9937 | 0.9944 | 0.9987 | 1.0000 |
| S6 sparse, transient, 15% crashes | 0.6963 | 0.7466 | 0.8323 | 0.8289 | 0.8464 | 0.8494 |

Primary contrast, diagnosis-aware minus generic quarantine (PDR; paired 95% intervals are unadjusted):

| Scenario | Mean paired difference | 95% CI | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| S1 | +0.00001 | [−0.00001, +0.00003] | 0.326 | 0.352 |
| S2 | +0.00318 | [−0.00151, +0.00786] | 0.176 | 0.352 |
| S3 | +0.02083 | [+0.00372, +0.03795] | 0.0188 | 0.0803 |
| S4 | +0.01840 | [+0.00368, +0.03311] | 0.0161 | 0.0803 |
| S5 | +0.00430 | [+0.00055, +0.00806] | 0.0262 | 0.0803 |
| S6 | +0.01745 | [+0.00692, +0.02798] | 0.00204 | 0.0122 |

Only S6's primary p-value remains below 0.05 after Holm correction. The result is a conditional improvement over generic quarantine within this scenario and this simulator; it is not a general benefit. S3–S5 are positive before correction but do not meet the adjusted threshold. Every conclusion should include the corrected comparison and its limitations.

After the S6 crash, diagnosis-aware and generic-quarantine recovery times were both right-censored at 26 rounds: neither had three consecutive rounds at or above 95% of its pre-crash PDR mean. Blind rerouting and no recovery averaged 25.13 rounds (SD 4.75), partly because some runs met the threshold before censoring. This is not proof of higher service. Diagnosis-aware S6 PDR was 0.8464, generic-quarantine PDR 0.8289 and blind-reroute PDR 0.7466. Route availability was 0.8953 for every policy.

Diagnosis-aware S6 used 3.983 abstract energy units per offered packet (SD 1.076) versus 3.832 (1.038) for generic quarantine and 3.325 (0.686) for blind rerouting. These are simulator costs, not joules. Maximum per-node energy is a traffic-concentration proxy only; no battery model supports network-lifetime estimation. The diagnosis-aware packet-study classifier's mean fault recall was 0.7679 (95% t interval 0.7535–0.7824), not the earlier 0.84 estimate. The classifier summary and full seed-wise outcomes are in `results/recovery_policy_v4/`.

## Limitations
* Every behavior, link and cost is synthetic. Forwarding probability is assigned from ground-truth labels; the model has no MAC, queues, collisions, control traffic or calibrated radio stack.
* Silent blackholes acknowledge then discard packets. Whether this behavior is visible depends on the protocol and acknowledgment layer.
* The primary comparison bundles several policy differences; it does not isolate diagnosis causally.
* S5 is a constructed stress topology, not a sample of field network topologies.
* No finite battery or calibrated energy model exists. Do not call the hotspot/load measure network lifetime.
* The classifier has only four fault and four blackhole nodes in the 20-node scenarios. Fault recall is imperfect.
* The paired intervals are ordinary 95% t intervals; p-values, not intervals, are Holm-adjusted. The one adjusted-positive result is conditional on the six frozen scenarios.
* Policy penalties and confirmation/quarantine rules were pre-specified but not sensitivity-tested.
