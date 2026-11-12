# Drop-in text for the paper (replaces sections 3.3-3.5 and 4.2-4.7 recovery material)

## Methods (recovery)
We compare six routing-recovery policies in a packet-level simulation of a 20-node network with a gateway.
Policies differ only in how predicted node states constrain routing: static routing (no recovery); reactive
rerouting after a failed hop (blind); exclusion of nodes currently predicted anomalous (generic exclusion);
the same temporary quarantine-with-readmission applied to every anomalous node (diagnosis-blind quarantine);
the proposed diagnosis-aware policy; and an oracle using true labels. The diagnosis-aware policy permanently
isolates a node after two blackhole predictions in the last three rounds and raises an alert, quarantines
predicted-fault nodes and readmits them after two clean rounds, and keeps quarantined nodes usable at a
1.5x cost when no clean route exists. Every policy except static routing shares the same reactive rerouting
and the same unconstrained last-resort route, so a policy cannot lose packets merely by refusing to use a
suspected node. Predictions are node-grouped out-of-fold Random Forest outputs. Blackholes are simulated in
two ways: detectable (the drop appears as a failed hop) and silent (the node acknowledges the hop, then drops
the packet). Faults are persistent or transient. Six scenarios cross connectivity, drop mode, fault mode and
node crashes; one is a constructed bottleneck stress topology. We use 30 seeds, paired comparisons with common
random numbers, and Holm-corrected paired t intervals; the pre-specified primary contrast is diagnosis-aware
versus diagnosis-blind quarantine on packet delivery ratio.

## Results (recovery)
When blackhole drops were detectable by hop-level retries, all reactive policies reached PDR 0.999 and
diagnosis made no difference, matching our earlier null result. When blackholes dropped silently, proactive
exclusion raised PDR by 6.1 to 18.4 percentage points over reactive rerouting. Relative to the
diagnosis-blind quarantine control, diagnosis-aware recovery added +0.9 to +2.3 points in sparse, bottleneck
and crash scenarios (Holm-adjusted p = 0.027 and 0.005 for the bottleneck and crash scenarios, 0.060 for
the two sparse scenarios) and no measurable gain in dense networks. The oracle policy was within about
0.5 points of the learned diagnosis. The diagnosis also gave attack-alert precision of 0.83 versus 0.49
for a diagnosis-blind system. Proactive policies concentrated traffic on surviving relays (hotspot energy
15 to 50% higher than reactive rerouting in the sparse and bottleneck scenarios).

## Claims you may make
* Diagnosis matters only when attack behaviour is not observable at hop level and connectivity is limited.
* Separating fault from attack gives a small, scenario-dependent delivery benefit and a clear alerting benefit.
* Earlier null results came from a ceiling effect (detectable drops) and, in v2/v3, a missing fallback.

## Claims you must not make
* Any field, hardware, or calibrated-energy claim. No network-lifetime improvement. No generality beyond the
  modelled assumptions. S5 is a stress test, not evidence about typical networks.
