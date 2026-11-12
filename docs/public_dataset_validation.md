# Public dataset classifier evaluation

## Scope

These experiments evaluate classification only. They do not run recovery, packet delivery, routing availability, network lifetime, or radio energy using either public dataset. WSN-DS is an NS-2/LEACH simulation dataset for attack classification; Intel Lab is environmental sensor data without verified fault labels. The WSN-DS paper states that its records cover normal, blackhole, grayhole, flooding and scheduling behavior and contain 374,661 records. The official Intel Lab page describes 54 Mica2Dot sensors and about 2.3 million readings, with temperature, humidity, light, voltage and topology.

Run both experiments from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.public_dataset_validation
```

The raw files are read-only inputs. Results, source checksums, out-of-fold predictions and the derived Intel injection data are written to `results/public_dataset_validation/`.

## WSN-DS method and result

The CSV contains 374,661 rows. This copy has 8,873 exact duplicate rows; the analysis removes those, leaving 365,788 rows and 100 node groups. The encoded node ID ends in the node number (the paper's example formats the identifier as round/stage/node); the code checks that the final three digits identify nodes 0–99. `id`, `Time`, `who CH`, and the label are excluded as predictors. The remaining 15 numeric predictors are used. `TDMA` is normalized to the paper's `Scheduling` label.

One five-fold `StratifiedGroupKFold` creates out-of-fold predictions with entire nodes held out. The Random Forest has 100 trees, balanced class weights, minimum leaf size 2, and fixed per-fold seeds. Results are pooled over held-out records. Uncertainty uses 2,000 bootstrap resamples of whole node groups; it does not resample individual rows. The imbalanced source class counts after duplicate removal are: Normal 332,040; Blackhole 10,049; Grayhole 13,909; Flooding 3,157; Scheduling 6,633.

The pooled out-of-fold accuracy is **0.99545**, macro-F1 **0.97086** (95% node-cluster bootstrap interval **0.96448–0.97504**). Per-class precision/recall/F1 (each with 95% cluster-bootstrap intervals) are: Normal 0.9985/0.9976/0.9980; Blackhole 0.9721/0.9939/0.9829; Grayhole 0.9751/0.9777/0.9764; Flooding 0.9049/0.9975/0.9489; Scheduling 0.9691/0.9279/0.9481. The five fold macro-F1 values range from 0.95746 to 0.97663. These high scores should be interpreted only as generalization to held-out node identifiers within this same simulated data generation; they are not performance on physical WSN attack traffic. Attack categories also are not the project's transient hardware-fault label.

## Intel Lab fault-injection method and result

The original readings file is whitespace-delimited. The experiment parses it in chunks and aggregates temperature by mote and five-minute time bin using the mean. It does not fill missing bins. The code excludes rows whose mote ID is outside 1–54 or invalid (10,392 rows in this copy), because the official deployment page describes 54 motes. Mote 5 has only 35 usable bins, too few for the frozen injection schedule, so it is excluded. The remaining 53 motes contribute 407,147 observed bins.

The derived copy injects three disjoint six-observation episodes per mote for each of bias (+3 °C), drift (+0.5 °C per observation), and stuck-at (held at the episode's initial temperature), plus two isolated +8 °C spikes per mote. Fault events are placed only within uninterrupted five-minute sequences. The random seed is 7301. No faults are inserted into the raw file. The classifier gets temperature, delta, rolling-three-observation mean, standard deviation, and range; mote ID and time are not predictors. Five-fold grouped out-of-fold validation holds out complete motes. The 95% intervals are based on 2,000 bootstrap resamples of whole motes.

Pooled out-of-fold accuracy is **0.92727**, macro-F1 **0.59798** (95% mote-cluster bootstrap interval **0.58109–0.61590**). Per-class precision/recall/F1 are: Normal 0.9950/0.9314/0.9622; Bias 0.8892/0.3784/0.5309; Drift 0.6807/0.4738/0.5587; Stuck 0.0063/0.1834/0.0122; Spike 0.9091/0.9434/0.9259. Each value has a 95% mote-cluster bootstrap interval in `intel_lab_injected_class_metrics.csv`. This gap illustrates why accuracy is misleading for the strongly imbalanced injected data. In particular, the current detector has very poor stuck-fault precision because many clean temperature sequences are similarly flat; this is a warning against treating injected-data accuracy as evidence of reliable fault detection.

This is a reproducible synthetic-fault benchmark built on real environmental measurements. Its fault labels were created by our injection code, not supplied by the Intel deployment. The selected fault types and magnitudes are assumptions, not measurements of actual sensor failures. The downloaded coordinate/connectivity files are checksummed but are not inputs to this classifier-only study. It does not validate communication faults, blackhole attacks, relay behavior, or diagnosis-aware routing.

## Artifact inventory and limitations

- `source_manifest.csv`: byte sizes and SHA-256 checksums for all downloaded raw dataset files.
- `wsn_ds_*`: data audit, class counts, fold metrics, class metrics/intervals, predictions and summary.
- `intel_lab_injected_*`: derived labels/readings, fold metrics, class metrics/intervals, predictions and summary.
- `dataset_summary.json`: compact summary of both evaluations.

The confidence intervals quantify variation across nodes/motes under cluster resampling; they do not account for alternative dataset generations, fault magnitudes, collection campaigns or model choices. No hyperparameter search was performed. A future revision should add pre-specified fault magnitude sensitivity, a second model/baseline, and temporal transfer evaluation without tuning on the held-out nodes. ESP32 inference/RAM/energy measurement is explicitly future work because hardware and a power meter are unavailable.
