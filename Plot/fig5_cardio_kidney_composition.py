from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rebuild_lib import FIGURE_DATA_DIR, FIGURE_DIR, ensure_dirs


COMPOSITION_ORDER = ["neither", "single_organ_only", "both"]
COMPOSITION_LABELS = {
    "neither": "Neither abnormality",
    "single_organ_only": "Cardiac/Kidney only",
    "both": "Both abnormalities",
}
COMPOSITION_COLORS = {
    "neither": "#cfa8a7",
    "single_organ_only": "#b8c5d4",
    "both": "#7d626f",
}
def _panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    comp = data.copy()
    comp["composition_group"] = comp["cardio_kidney_composition"].replace(
        {"kidney_only": "single_organ_only", "cardiac_only": "single_organ_only"}
    )

    burden = (
        comp.groupby(["non_glycemic_burden_group", "composition_group"], observed=False)["n"]
        .sum()
        .reset_index()
    )
    burden_totals = burden.groupby("non_glycemic_burden_group", observed=False)["n"].sum().rename("total")
    burden = burden.merge(burden_totals, on="non_glycemic_burden_group")
    burden["composition_pct"] = burden["n"] / burden["total"] * 100
    burden["axis_group"] = "Metabolic burden"
    burden["axis_level"] = burden["non_glycemic_burden_group"].astype(str)

    stea = (
        comp.groupby(["steatosis_label", "composition_group"], observed=False)["n"]
        .sum()
        .reset_index()
    )
    stea_totals = stea.groupby("steatosis_label", observed=False)["n"].sum().rename("total")
    stea = stea.merge(stea_totals, on="steatosis_label")
    stea["composition_pct"] = stea["n"] / stea["total"] * 100
    stea["axis_group"] = "Steatosis severity"
    stea["axis_level"] = stea["steatosis_label"].astype(str)

    plot_df = pd.concat(
        [
            burden[["axis_group", "axis_level", "composition_group", "composition_pct"]],
            stea[["axis_group", "axis_level", "composition_group", "composition_pct"]],
        ],
        ignore_index=True,
    )

    x_positions = {
        ("Metabolic burden", "low"): 0,
        ("Metabolic burden", "intermediate"): 1,
        ("Metabolic burden", "high"): 2,
        ("Steatosis severity", "No steatosis"): 4,
        ("Steatosis severity", "Mild steatosis"): 5,
        ("Steatosis severity", "Moderate/severe steatosis"): 6,
    }
    tick_positions = [0, 1, 2, 4, 5, 6]
    tick_labels = ["Low", "Intermediate", "High", "No steatosis", "Mild steatosis", "Moderate/severe"]

    bottom = pd.Series(0.0, index=tick_positions, dtype=float)
    for key in COMPOSITION_ORDER:
        sub = plot_df.loc[plot_df["composition_group"] == key].copy()
        sub["x"] = [x_positions[(g, l)] for g, l in zip(sub["axis_group"], sub["axis_level"])]
        sub = sub.sort_values("x")
        ax.bar(
            sub["x"],
            sub["composition_pct"],
            bottom=bottom.loc[sub["x"]].to_numpy(),
            color=COMPOSITION_COLORS[key],
            edgecolor="white",
            linewidth=0.6,
            label=COMPOSITION_LABELS[key],
        )
        text_color = "white" if key == "both" else "#374151"
        for row in sub.itertuples(index=False):
            y_center = bottom.loc[row.x] + row.composition_pct / 2
            ax.text(
                row.x,
                y_center,
                f"{row.composition_pct:.0f}%",
                ha="center",
                va="center",
                fontsize=9.2,
                color=text_color,
            )
        bottom.loc[sub["x"]] = bottom.loc[sub["x"]].to_numpy() + sub["composition_pct"].to_numpy()

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 6.6)
    ax.set_xticks(tick_positions, tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Composition within group (%)")
    ax.set_title("Cardio-kidney composition", fontsize=12.5, pad=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=9.5)
    ax.text(1.0, -0.22, "Metabolic burden", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.2)
    ax.text(5.0, -0.22, "Steatosis severity", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.2)
    ax.axvline(3.0, color="#9ca3af", linestyle=":", linewidth=1.2)
def main() -> None:
    ensure_dirs()
    composition = pd.read_csv(FIGURE_DATA_DIR / "figure5_cardio_kidney_composition.csv")

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    _panel_a(ax, composition)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.82)
    fig.savefig(FIGURE_DIR / "figure5_cardio_kidney_composition.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure5_cardio_kidney_composition.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
