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
| Scenario | no_rec | blind | generic_excl | generic_quar | diagnosis_aware | oracle |
|---|---|---|---|---|---|---|
| S1 dense, detectable | 0.931 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| S2 dense, silent | 0.924 | 0.938 | 0.995 | 0.995 | 0.999 | 0.999 |
| S3 sparse, silent | 0.824 | 0.847 | 0.940 | 0.935 | 0.957 | 0.962 |
| S4 sparse, silent, transient | 0.811 | 0.842 | 0.936 | 0.931 | 0.951 | 0.955 |
| S5 bottleneck, silent, transient | 0.745 | 0.816 | 0.994 | 0.991 | 0.9995 | 1.000 |
| S6 S4 + crashes | 0.772 | 0.827 | 0.920 | 0.914 | 0.936 | 0.941 |

Primary contrast, diagnosis_aware minus generic_quarantine (PDR): S1 +0.0000; S2 +0.0041 (Holm p=0.23);
S3 +0.0228 [0.0044, 0.0412] (Holm p=0.060); S4 +0.0201 [0.0042, 0.0360] (p=0.060);
S5 +0.0086 [0.0028, 0.0144] (p=0.027); S6 +0.0226 [0.0102, 0.0351] (p=0.005).

Reading: (1) when drops are detectable every reactive policy sits at the ceiling, which explains the earlier
null result; (2) when blackholes drop silently, proactive exclusion is worth 6-18 PDR points over blind
rerouting; (3) separating fault from attack adds about 1-2 PDR points beyond a memory-matched
diagnosis-blind control, confirmed after Holm correction in S5 and S6 and borderline in S3/S4;
(4) the oracle is within about 0.5 points of the learned diagnosis, so the classifier is not the bottleneck here.

Non-PDR outcomes: node-level attack alert precision 0.83 (recall 1.0) for diagnosis-aware versus 0.49 when
every flagged node is treated as a suspected attacker (a diagnosis-blind system cannot raise a meaningful
attack alert). Energy per delivered packet is lower than blind in S1-S5 but higher in S6. Hotspot energy
(max per-node energy per round) is 15-50% higher than blind in S3-S6 because proactive policies concentrate
traffic on surviving relays, so a lifetime proxy (battery / hotspot drain) would favour blind rerouting
even though it delivers far fewer packets. Abstract units only; no battery-depletion model.
After the S6 crash, PDR recovery took 2.4 rounds (diagnosis-aware) versus 5.1 (blind) and 9.4 (no recovery).
Average normal nodes falsely isolated per seed: 0.3.

## Limitations (state these in the paper)
* All dynamics are assumptions. The advantage exists only under silent-drop blackholes and constrained
  connectivity; the result is conditional on that modelling choice, not a field finding.
* S5 is a constructed stress topology. Do not generalise from it.
* Forwarding behaviour is assigned from ground-truth labels; no MAC, queues, collisions or control traffic.
* No calibrated energy or battery model; "lifetime" is only a hotspot proxy.
* The classifier here (node-grouped OOF, 4 faults/20 nodes, fault recall about 0.84) is a different protocol
  from the earlier 6-node held-out run (fault recall 0.565). Do not mix the numbers.
* Hyperparameters (1.5 and 2.0 penalties, 2-of-3 confirmation, 2-round quarantine) were fixed a priori
  but not tuned or sensitivity-tested; a sweep is future work.
