# Dataset and hardware validation plan

Status as of 2026-11-10: WSN-DS and Intel Lab raw files are now present in the paths below and have been evaluated for classification. No ESP32 or power meter is available, so hardware measurements remain future work. No hardware result is claimed in this repository. See `docs/public_dataset_validation.md` for the completed dataset methods/results.

## WSN-DS download and use

1. Open the dataset page linked from the publisher's [WSN-DS paper](https://doi.org/10.1155/2016/4731953), or use the public [Kaggle copy](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds). Kaggle may require a free account and acceptance of its download terms.
2. Download the original archive/CSV without editing it. Save it under `data/raw/wsn-ds/` and preserve its original filename. Keep the source URL, download date, archive name, and SHA-256 checksum in a small manifest.
3. Do not describe these rows as field-collected sensor readings. The authors describe an NS-2 simulation using LEACH that was processed into 23 features. Its classes concern normal behavior and simulated blackhole, grayhole, flooding, and scheduling attacks. This is useful for an attack-classification comparison, but does not supply ground truth for transient hardware faults or physical fault recovery.
4. Before model training, inspect schema, class counts, duplicates, missingness, and available node/time identifiers. Use a node/episode split if identifiers allow it. If there is no independent deployment or time grouping key, report the limitation and do not treat random row splitting as independent-node generalization.

## Intel Berkeley Lab download and injected labels

1. Open the [official Intel Lab data page](https://db.csail.mit.edu/labdata/labdata.html).
2. Download the linked compressed readings file, mote-coordinate file, and (if useful for network reconstruction) topology/connectivity file. Save the untouched files under `data/raw/intel-lab/`, preserving original filenames and recording URL, date, and SHA-256 checksums.
3. The official page describes 54 Mica2Dot sensors with weather boards, timestamped topology, humidity, temperature, light, and voltage readings, sampled about every 31 seconds from 2004-02-28 to 2004-04-05. The raw dataset does not include verified fault labels. Published work may select a subset and preprocess it; that procedure must be documented.
4. Create a derived dataset in `data/processed/intel-lab-injected/`, never overwriting raw files. Keep one clean copy and separately inject explicitly synthetic faults (for example bias, drift, stuck-at, missingness, or burst corruption) with a fixed, recorded random seed and per-row injection provenance. The resulting evaluation tests detection against controlled injected labels; it does not prove that the detector recognizes naturally occurring faults.
5. For routing/security metrics, the sensor values alone are insufficient. Use the provided topology only as an observed-connectivity input if its semantics match the task; otherwise limit this dataset's role to sensing-value fault detection. Do not infer packet-forwarding blackhole labels from temperature or voltage values.

## ESP32 measurement plan (pending equipment)

No device-level measurement can be made until an ESP32-class target board and a suitable current/power measurement setup are available. Once available, freeze the model artifact and feature pipeline; record board model, clock, framework/compiler, supply voltage, power instrument and sampling rate. Measure model flash size, peak RAM, feature extraction memory, cold and steady-state inference latency (with warm-up and repeated trials), and energy per inference from the incremental current integral over a measured baseline. Report median and tail latency, repetitions, build settings, and uncertainty. Separate model inference from sensor/radio energy. Compare a simpler baseline if feasible. One-board measurements support that board/setup only; they do not establish broad WSN hardware performance.

## External data and authorship/submission dependencies

The data files have been downloaded and the loader/evaluations are now implemented. Supervisor review, plagiarism screening, venue choice, arXiv posting, and submission remain uncompleted; these require supervisor/institution details, access to the chosen checker/venue, and explicit author approval before any external posting or submission.
