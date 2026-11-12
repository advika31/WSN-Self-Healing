"""Leakage-aware classifier checks for WSN-DS and Intel Lab.

WSN-DS is an NS-2/LEACH attack dataset, not field telemetry. Intel Lab is
real environmental telemetry without fault ground truth; this module adds
explicit, reproducible synthetic temperature faults to a derived copy.
Neither dataset can validate packet-level recovery, energy, or lifetime.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
WSN_DS_PATH = ROOT / "data" / "raw" / "wsn-ds" / "WSN-DS.csv"
INTEL_DIR = ROOT / "data" / "raw" / "intel-lab"
OUT = ROOT / "results" / "public_dataset_validation"
SEED = 7301
CLASSES_WSN = ["Normal", "Blackhole", "Grayhole", "Flooding", "Scheduling"]
CLASSES_INTEL = ["normal", "bias", "drift", "stuck", "spike"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wsn_ds(path: Path = WSN_DS_PATH):
    """Return cleaned features, labels, node groups, and a source audit.

    WSN-DS encodes the node number in the last three digits of its unique
    node/round/stage ID. The CSV's decimal form is checked for the expected
    100 node identifiers. IDs, time and CH identity are excluded as inputs.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"id", "Time", "who CH", "Attack type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"WSN-DS is missing required columns: {sorted(missing)}")
    raw_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    frame["node_group"] = pd.to_numeric(frame["id"], errors="raise").astype("int64") % 1000
    groups_found = sorted(frame.node_group.unique().tolist())
    if groups_found != list(range(100)):
        raise ValueError("Expected encoded WSN-DS node IDs 0--99 in the last three digits")
    frame["label"] = frame["Attack type"].replace({"TDMA": "Scheduling"})
    if set(frame.label.unique()) != set(CLASSES_WSN):
        raise ValueError(f"Unexpected WSN-DS classes: {sorted(frame.label.unique())}")
    excluded = {"id", "Time", "who CH", "Attack type", "label", "node_group"}
    feature_names = [column for column in frame.columns if column not in excluded]
    features = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Unexpected nonnumeric or missing WSN-DS feature values")
    audit = {
        "source_rows": raw_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "analysis_rows": len(frame),
        "node_groups": len(groups_found),
        "features": feature_names,
        "class_counts": frame.label.value_counts().reindex(CLASSES_WSN, fill_value=0).to_dict(),
    }
    return features, frame.label.astype(str), frame.node_group.to_numpy(), audit


def grouped_oof_metrics(features, labels, groups, classes, seed=SEED, folds=5):
    """Generate one node-grouped out-of-fold prediction for each row."""
    y, groups = np.asarray(labels, dtype=str), np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = np.empty(len(y), dtype=object)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y, groups), 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Group leakage detected")
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
            random_state=seed + fold, n_jobs=-1, min_samples_leaf=2)
        model.fit(features.iloc[train_idx], y[train_idx])
        pred = model.predict(features.iloc[test_idx])
        predicted[test_idx] = pred
        fold_rows.append({"fold": fold, "n_test_rows": len(test_idx),
            "n_test_nodes": len(np.unique(groups[test_idx])),
            "accuracy": accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, labels=classes,
                                  average="macro", zero_division=0)})
    if pd.isna(predicted).any():
        raise RuntimeError("OOF predictions did not cover all records")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=classes, zero_division=0)
    class_rows = [{"class": label, "precision": p, "recall": r, "f1": score,
                   "support": int(n)}
                  for label, p, r, score, n in zip(classes, precision, recall, f1, support)]
    summary = {"n_rows": len(y), "n_groups": len(np.unique(groups)), "n_folds": folds,
        "accuracy_oof": accuracy_score(y, predicted),
        "macro_f1_oof": f1_score(y, predicted, labels=classes,
                                  average="macro", zero_division=0),
        "confusion_matrix_labels": list(classes),
        "confusion_matrix": confusion_matrix(y, predicted, labels=classes).tolist()}
    return summary, pd.DataFrame(class_rows), pd.DataFrame(fold_rows), predicted


def cluster_bootstrap_metrics(labels, predictions, groups, classes, seed=SEED,
                              replicates=2000):
    """Percentile intervals resampling whole nodes/motes, never individual rows."""
    y, pred, groups = np.asarray(labels, dtype=str), np.asarray(predictions, dtype=str), np.asarray(groups)
    group_ids = np.unique(groups)
    matrices = []
    for group in group_ids:
        mask = groups == group
        matrices.append(confusion_matrix(y[mask], pred[mask], labels=classes))
    matrices = np.asarray(matrices, dtype=np.int64)

    def metrics_from_matrix(matrix):
        tp = np.diag(matrix).astype(float)
        precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros_like(tp),
                              where=matrix.sum(axis=0) != 0)
        recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros_like(tp),
                           where=matrix.sum(axis=1) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) != 0)
        return np.column_stack([precision, recall, f1])

    point = metrics_from_matrix(matrices.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(classes), 3), dtype=float)
    macro_draws = np.empty(replicates, dtype=float)
    for rep in range(replicates):
        chosen = rng.integers(0, len(group_ids), size=len(group_ids))
        values = metrics_from_matrix(matrices[chosen].sum(axis=0))
        draws[rep] = values
        macro_draws[rep] = values[:, 2].mean()
    rows = []
    for index, label in enumerate(classes):
        row = {"class": label, "precision": point[index, 0], "recall": point[index, 1],
               "f1": point[index, 2], "support": int(matrices[:, index, :].sum())}
        for metric_index, metric in enumerate(("precision", "recall", "f1")):
            low, high = np.quantile(draws[:, index, metric_index], [0.025, 0.975])
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    macro_ci = np.quantile(macro_draws, [0.025, 0.975])
    return pd.DataFrame(rows), {"macro_f1_ci95_low": macro_ci[0],
        "macro_f1_ci95_high": macro_ci[1], "cluster_bootstrap_replicates": replicates,
        "cluster_bootstrap_unit": "node/mote"}


def read_intel_temperature(path: Path | None = None, chunksize=200_000) -> pd.DataFrame:
    """Aggregate official whitespace readings into five-minute mote-bin means."""
    path = path or INTEL_DIR / "data.txt"
    columns = ["date", "clock", "epoch", "mote_id", "temperature",
               "humidity", "light", "voltage"]
    partials = []
    malformed_rows = 0
    for chunk in pd.read_csv(path, sep=r"\s+", names=columns, header=None,
                             usecols=[0, 1, 3, 4], chunksize=chunksize,
                             dtype={0: "string", 1: "string", 3: "float32", 4: "float32"}):
        chunk.columns = ["date", "clock", "mote_id", "temperature"]
        valid_mote = chunk.mote_id.notna() & chunk.mote_id.between(1, 54) & \
            (chunk.mote_id % 1 == 0)
        malformed_rows += int((~valid_mote).sum())
        chunk = chunk.loc[valid_mote].copy()
        chunk["mote_id"] = chunk.mote_id.astype("int16")
        timestamp = pd.to_datetime(chunk.date + " " + chunk.clock, errors="coerce")
        chunk = chunk.assign(time_bin=timestamp.dt.floor("5min"))
        chunk = chunk.dropna(subset=["time_bin", "temperature"])
        chunk = chunk[np.isfinite(chunk.temperature)]
        grouped = chunk.groupby(["mote_id", "time_bin"], sort=False).temperature.agg(
            temperature_sum="sum", reading_count="count").reset_index()
        partials.append(grouped)
    combined = pd.concat(partials, ignore_index=True)
    combined = combined.groupby(["mote_id", "time_bin"], as_index=False).agg(
        temperature_sum=("temperature_sum", "sum"), reading_count=("reading_count", "sum"))
    combined["temperature"] = combined.temperature_sum / combined.reading_count
    combined = combined.sort_values(["mote_id", "time_bin"]).reset_index(drop=True)
    if combined.mote_id.nunique() != 54:
        raise ValueError(f"Expected 54 Intel Lab motes, found {combined.mote_id.nunique()}")
    result = combined[["mote_id", "time_bin", "temperature", "reading_count"]]
    result.attrs["malformed_rows_skipped"] = malformed_rows
    return result


def inject_intel_faults(binned: pd.DataFrame, seed=SEED, episodes_per_type=3,
                        episode_bins=6, spikes_per_node=2) -> pd.DataFrame:
    """Create a derived set with reproducible bias, drift, stuck and spike labels."""
    result = binned.copy().sort_values(["mote_id", "time_bin"]).reset_index(drop=True)
    result["label"] = "normal"
    excluded_motes = []
    eligible_motes = []
    required_contiguous = episodes_per_type * 3 * episode_bins + 2 * spikes_per_node + 8
    for mote_id, node_rows in result.groupby("mote_id", sort=True):
        times = node_rows.time_bin.to_numpy()
        gaps = pd.Series(times).diff().dt.total_seconds().fillna(300).to_numpy()
        segment = np.cumsum(gaps != 300)
        if max(np.bincount(segment)) < required_contiguous:
            excluded_motes.append(int(mote_id))
        else:
            eligible_motes.append(int(mote_id))
    result = result[result.mote_id.isin(eligible_motes)].reset_index(drop=True)
    if result.empty:
        raise ValueError("No motes have a sufficiently long contiguous segment for injection")
    for mote_id, index_values in result.groupby("mote_id", sort=True).groups.items():
        indices = np.asarray(list(index_values), dtype=int)
        rng = np.random.default_rng(seed + int(mote_id) * 104_729)
        occupied = np.zeros(len(indices), dtype=bool)
        times = result.loc[indices, "time_bin"].to_numpy()
        gaps = pd.Series(times).diff().dt.total_seconds().fillna(300).to_numpy()
        segment = np.cumsum(gaps != 300)
        run_bounds = [(int(np.flatnonzero(segment == sid)[0]),
                       int(np.flatnonzero(segment == sid)[-1]) + 1)
                      for sid in np.unique(segment)]
        candidates = [start for lo, hi in run_bounds
                      for start in range(lo, hi - episode_bins + 1)]
        starts_by_type = {}
        for label in ("bias", "drift", "stuck"):
            starts = []
            order = rng.permutation(candidates)
            for start in order:
                if len(starts) == episodes_per_type:
                    break
                lo, hi = max(0, start - 1), min(len(indices), start + episode_bins + 1)
                if not occupied[lo:hi].any():
                    occupied[start:start + episode_bins] = True
                    starts.append(int(start))
            if len(starts) != episodes_per_type:
                raise RuntimeError(f"Could not place enough {label} episodes for mote {mote_id}")
            starts_by_type[label] = starts
        available = [i for i in range(len(indices)) if not occupied[i]
                     and not occupied[max(0, i - 1):min(len(indices), i + 2)].any()]
        spike_positions = []
        for position in rng.permutation(available):
            if len(spike_positions) == spikes_per_node:
                break
            if all(abs(int(position) - prior) > 1 for prior in spike_positions):
                spike_positions.append(int(position))
        if len(spike_positions) != spikes_per_node:
            raise RuntimeError(f"Could not place isolated spikes for mote {mote_id}")
        base = result.loc[indices, "temperature"].to_numpy(dtype=float)
        modified = base.copy()
        labels = np.full(len(indices), "normal", dtype=object)
        for start in starts_by_type["bias"]:
            modified[start:start + episode_bins] += 3.0
            labels[start:start + episode_bins] = "bias"
        for start in starts_by_type["drift"]:
            modified[start:start + episode_bins] += np.arange(episode_bins) * 0.5
            labels[start:start + episode_bins] = "drift"
        for start in starts_by_type["stuck"]:
            modified[start:start + episode_bins] = base[start]
            labels[start:start + episode_bins] = "stuck"
        modified[spike_positions] += 8.0
        labels[spike_positions] = "spike"
        result.loc[indices, "temperature"] = modified
        result.loc[indices, "label"] = labels
    result.attrs["excluded_motes_too_short"] = excluded_motes
    return result


def intel_features(injected: pd.DataFrame):
    """Make temperature-only rolling features; keep mote ID solely for grouping."""
    frame = injected.sort_values(["mote_id", "time_bin"]).copy()
    gaps = frame.groupby("mote_id", sort=False).time_bin.diff().dt.total_seconds()
    frame["segment_id"] = (gaps.ne(300) | gaps.isna()).groupby(frame.mote_id).cumsum()
    grouped = frame.groupby(["mote_id", "segment_id"], sort=False).temperature
    frame["temp_delta"] = grouped.diff().fillna(0.0)
    frame["temp_mean_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    frame["temp_std_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0.0))
    frame["temp_range_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).max() -
                                               s.rolling(3, min_periods=1).min())
    columns = ["temperature", "temp_delta", "temp_mean_3", "temp_std_3", "temp_range_3"]
    return frame[columns].reset_index(drop=True), frame.label.astype(str).reset_index(drop=True), \
        frame.mote_id.to_numpy()


def run_wsn_ds(out_dir=OUT):
    features, labels, groups, audit = load_wsn_ds()
    summary, classes, folds, predictions = grouped_oof_metrics(
        features, labels, groups, CLASSES_WSN)
    classes, intervals = cluster_bootstrap_metrics(labels, predictions, groups, CLASSES_WSN)
    summary.update(intervals)
    summary.update({"dataset": "WSN-DS", "evaluation": "5-fold StratifiedGroupKFold by node ID",
        "raw_rows": audit["source_rows"], "exact_duplicate_rows_removed": audit["exact_duplicate_rows_removed"],
        "analysis_rows": audit["analysis_rows"], "n_features": len(audit["features"]),
        "features_excluded": ["id", "Time", "who CH", "Attack type"]})
    classes.to_csv(out_dir / "wsn_ds_class_metrics.csv", index=False)
    folds.to_csv(out_dir / "wsn_ds_fold_metrics.csv", index=False)
    pd.DataFrame({"label": labels, "prediction": predictions, "node_group": groups}).to_csv(
        out_dir / "wsn_ds_oof_predictions.csv", index=False)
    (out_dir / "wsn_ds_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    (out_dir / "wsn_ds_audit.json").write_text(pd.Series(audit).to_json(indent=2), encoding="utf-8")
    return summary, classes, folds


def run_intel(out_dir=OUT):
    source = INTEL_DIR / "data.txt"
    binned = read_intel_temperature(source)
    injected = inject_intel_faults(binned, seed=SEED, episodes_per_type=3,
                                   episode_bins=6, spikes_per_node=2)
    features, labels, groups = intel_features(injected)
    summary, classes, folds, predictions = grouped_oof_metrics(
        features, labels, groups, CLASSES_INTEL)
    classes, intervals = cluster_bootstrap_metrics(labels, predictions, groups, CLASSES_INTEL)
    summary.update(intervals)
    summary.update({"dataset": "Intel Lab with synthetic temperature fault injection",
        "aggregation": "five-minute mote-bin mean; no empty bins imputed",
        "fault_injection": "3 disjoint six-observation episodes per mote per bias/drift/stuck; 2 isolated spikes per mote",
        "bias_celsius": 3.0, "drift_celsius_per_observation": 0.5,
        "spike_celsius": 8.0, "seed": SEED,
        "evaluation": "5-fold StratifiedGroupKFold by mote ID",
        "n_bins": len(injected), "n_motes": int(injected.mote_id.nunique()),
        "motes_excluded_too_short_for_fixed_injection_plan": injected.attrs.get(
            "excluded_motes_too_short", []),
        "malformed_source_rows_skipped": int(binned.attrs.get("malformed_rows_skipped", 0)),
        "raw_rows_not_used_as_independent_replicates": True})
    classes.to_csv(out_dir / "intel_lab_injected_class_metrics.csv", index=False)
    folds.to_csv(out_dir / "intel_lab_injected_fold_metrics.csv", index=False)
    injected.assign(prediction=predictions).to_csv(
        out_dir / "intel_lab_injected_oof_predictions.csv", index=False)
    injected.to_csv(out_dir / "intel_lab_injected_derived.csv", index=False)
    (out_dir / "intel_lab_injected_summary.json").write_text(
        pd.Series(summary).to_json(indent=2), encoding="utf-8")
    return summary, classes, folds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [WSN_DS_PATH, INTEL_DIR / "data.txt", INTEL_DIR / "connectivity.txt",
             INTEL_DIR / "mote_locs.txt"]
    manifest = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Required raw dataset file not found: {path}")
        manifest.append({"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(OUT / "source_manifest.csv", index=False)
    wsn_summary, _, _ = run_wsn_ds()
    intel_summary, _, _ = run_intel()
    pd.DataFrame([wsn_summary, intel_summary]).to_json(
        OUT / "dataset_summary.json", orient="records", indent=2)
    print("WSN-DS:", wsn_summary)
    print("Intel Lab injected:", intel_summary)
    print(f"Wrote classifier-only dataset results to {OUT}")


if __name__ == "__main__":
    main()
