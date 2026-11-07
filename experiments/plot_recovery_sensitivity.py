"""Create a paper-ready PDR sensitivity figure from the saved study outputs."""
from pathlib import Path
import os

MPL_CONFIG = Path("results/recovery_sensitivity_v1/mplconfig")
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG.resolve())
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/recovery_sensitivity_v1")
SUMMARY = RESULTS / "sensitivity_summary.csv"
FIGURES = RESULTS / "figures"

PANELS = [
    ("Radio range", "radio_range", [35.0, 48.0, 60.0, 75.0],
     {35.0: "S02", 48.0: "S01", 60.0: "S03", 75.0: "S04"}),
    ("Network size", "num_nodes", [20, 30, 50],
     {20: "S01", 30: "S05", 50: "S06"}),
    ("Blackhole forwarding probability", "blackhole_p", [0.05, 0.08, 0.20],
     {0.05: "S07", 0.08: "S01", 0.20: "S08"}),
    ("Fault forwarding probability", "fault_p", [0.40, 0.62, 0.80],
     {0.40: "S09", 0.62: "S01", 0.80: "S10"}),
]
COLORS = {"no_recovery": "#525866", "generic_rerouting": "#2F6F8F",
          "diagnosis_aware": "#B46A32"}
LABELS = {"no_recovery": "No recovery", "generic_rerouting": "Generic rerouting",
          "diagnosis_aware": "Diagnosis-aware"}


def main():
    data = pd.read_csv(SUMMARY)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    for ax, (title, field, values, scenario_ids) in zip(axes.flat, PANELS):
        for strategy in ("no_recovery", "generic_rerouting", "diagnosis_aware"):
            means, stds = [], []
            for value in values:
                sid = scenario_ids[value]
                row = data[(data.scenario_id == sid) & (data.strategy == strategy)].iloc[0]
                means.append(row.packet_delivery_ratio_mean)
                stds.append(row.packet_delivery_ratio_std)
            ax.errorbar(values, means, yerr=stds, marker="o", markersize=4,
                        linewidth=1.7, capsize=2.5, color=COLORS[strategy],
                        label=LABELS[strategy])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(field.replace("_", " ").title())
        ax.set_ylabel("Packet delivery ratio")
        ax.set_ylim(0.75, 1.015)
        ax.grid(axis="y", color="#D9DDE3", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Packet delivery across one-factor sensitivity conditions",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, 0.01,
             "Points are 30-seed means; error bars show ±1 sample SD across seed-level runs. "
             "Source: results/recovery_sensitivity_v1/sensitivity_summary.csv.",
             ha="center", fontsize=8.5, color="#4D5360")
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
    path = FIGURES / "packet_delivery_sensitivity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
