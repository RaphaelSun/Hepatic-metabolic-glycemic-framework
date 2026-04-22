from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rebuild_lib import FIGURE_DATA_DIR, FIGURE_DIR, RESULT_DIR, STEATOSIS_LABELS, ensure_dirs


COLOR_MAP = {
    "No steatosis": "#355070",
    "Mild steatosis": "#e07a5f",
    "Moderate/severe steatosis": "#8d0801",
}
BURDEN_ORDER = ["low", "intermediate", "high"]
BURDEN_TICKS = ["Low", "Intermediate", "High"]
COMPARISON_ORDER = ["2_high vs 0_low", "2_high vs 0_high", "0_high vs 0_low", "1_high vs 1_low"]
CONTRAST_COLORS = {
    "2_high vs 0_low": "#8d0801",
    "2_high vs 0_high": "#c1121f",
    "0_high vs 0_low": "#577590",
    "1_high vs 1_low": "#6d597a",
}
DISPLAY_LABELS = {
    "2_high vs 0_low": "0_low -> 2_high",
    "2_high vs 0_high": "0_high -> 2_high",
    "0_high vs 0_low": "0_low -> 0_high",
    "1_high vs 1_low": "1_low -> 1_high",
}


def main() -> None:
    ensure_dirs()
    data = pd.read_csv(FIGURE_DATA_DIR / "figure4_overall_proteinuria_lines.csv")
    contrasts = pd.read_csv(RESULT_DIR / "overall_proteinuria_key_contrasts.csv")
    x_map = {name: idx for idx, name in enumerate(BURDEN_ORDER)}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.8, 5.3), gridspec_kw={"width_ratios": [1.15, 1.0]})

    for stea_grade in [0, 1, 2]:
        sub = data.loc[data["steatosis_grade"] == stea_grade].copy()
        sub["x"] = sub["non_glycemic_burden_group"].map(x_map)
        label = STEATOSIS_LABELS[stea_grade]
        ax.plot(
            sub["x"],
            sub["adjusted_probability_pct"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=COLOR_MAP[label],
            label=label,
        )

    ax.set_xticks([0, 1, 2], BURDEN_TICKS)
    ax.set_xlabel("Non-glycemic burden")
    ax.set_ylabel("Adjusted proteinuria probability (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=10.5)
    ax.set_title("A. Overall phenotype pattern", fontsize=12.5, pad=10)

    plot_rows = []
    for idx, comparison in enumerate(COMPARISON_ORDER[::-1]):
        row = contrasts.loc[contrasts["comparison"] == comparison].iloc[0]
        plot_rows.append(
            {
                "comparison": comparison,
                "display_label": DISPLAY_LABELS[comparison],
                "y": idx,
                "ard": float(row["absolute_risk_difference_pp"]),
                "ref_pct": float(row["ref_adjusted_pct"]),
                "comp_pct": float(row["comp_adjusted_pct"]),
                "or_label": f"{row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f})",
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    x_max = max(14.5, plot_df["ard"].max() + 4.6)

    for row in plot_df.itertuples(index=False):
        bx.hlines(y=row.y, xmin=0, xmax=row.ard, color=CONTRAST_COLORS[row.comparison], linewidth=2.6, alpha=0.85)
        bx.scatter(row.ard, row.y, color=CONTRAST_COLORS[row.comparison], s=82, zorder=3)
        if row.comparison == "2_high vs 0_low":
            text_x = row.ard - 0.35
            ha = "right"
        else:
            text_x = row.ard + 0.35
            ha = "left"
        bx.text(
            text_x,
            row.y,
            f"{row.ref_pct:.1f}% -> {row.comp_pct:.1f}%",
            va="center",
            ha=ha,
            fontsize=9,
        )

    bx.axvline(0, color="#666666", linewidth=1, linestyle="--")
    bx.set_xlim(-0.3, x_max)
    bx.set_ylim(-0.7, len(plot_df) - 0.3)
    bx.set_yticks(plot_df["y"], plot_df["display_label"])
    bx.set_xlabel("Absolute risk difference (pp)")
    bx.set_title("B. Key phenotype contrasts", fontsize=12.5, pad=10)
    bx.grid(axis="x", alpha=0.2)

    inset = inset_axes(
        bx,
        width="34%",
        height="64%",
        loc="lower right",
        bbox_to_anchor=(0.0, 0.03, 1.0, 1.0),
        bbox_transform=bx.transAxes,
        borderpad=0.8,
    )
    for row in plot_df.itertuples(index=False):
        inset.hlines(
            y=row.y,
            xmin=float(contrasts.loc[contrasts["comparison"] == row.comparison, "or_ci_low"].iloc[0]),
            xmax=float(contrasts.loc[contrasts["comparison"] == row.comparison, "or_ci_high"].iloc[0]),
            color=CONTRAST_COLORS[row.comparison],
            linewidth=2.0,
            alpha=0.9,
        )
        inset.scatter(
            float(contrasts.loc[contrasts["comparison"] == row.comparison, "or"].iloc[0]),
            row.y,
            color=CONTRAST_COLORS[row.comparison],
            s=38,
            zorder=3,
        )
        inset.text(
            float(contrasts.loc[contrasts["comparison"] == row.comparison, "or_ci_high"].iloc[0]) * 1.03,
            row.y,
            row.or_label,
            va="center",
            ha="left",
            fontsize=7.6,
        )
    inset.axvline(1, color="#666666", linewidth=1, linestyle="--")
    inset.set_xscale("log")
    inset.set_xlim(0.9, 9.0)
    inset.set_xticks([1, 2, 4, 6])
    inset.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}" if x in [1, 2, 4, 6] else ""))
    inset.minorticks_off()
    inset.set_ylim(bx.get_ylim())
    inset.set_yticks([])
    inset.set_title("OR (95% CI)", fontsize=9, pad=4)
    inset.tick_params(axis="x", labelsize=8)
    inset.grid(axis="x", alpha=0.15)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure4_overall_proteinuria.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure4_overall_proteinuria.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
