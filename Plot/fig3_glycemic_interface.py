from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rebuild_lib import FIGURE_DATA_DIR, FIGURE_DIR, GLYCEMIC_LABELS, GLYCEMIC_ORDER, ensure_dirs


def _style_axes(ax: plt.Axes, fontsize: float = 15) -> None:
    ax.tick_params(axis="both", labelsize=fontsize - 1)


def _save_panel(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")


def _make_heatmap_panel(
    full_predictions: pd.DataFrame,
    status: str,
    panel_letter: str,
    vmax: float,
) -> plt.Figure:
    burden_order = ["low", "intermediate", "high"]
    sub = full_predictions.loc[full_predictions["glycemic_status"] == status].copy()
    pt = sub.pivot(index="steatosis_grade", columns="non_glycemic_burden_group", values="adjusted_probability_pct")
    pt = pt.loc[[0, 1, 2], burden_order]
    ann = pt.map(lambda x: f"{x:.1f}")

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    cmap = sns.color_palette("Reds", as_cmap=True)
    hm = sns.heatmap(
        pt,
        annot=ann,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        cbar=True,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 15},
    )

    ax.text(
        -0.10,
        1.08,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_title(f"{GLYCEMIC_LABELS[status]}\n(n = {int(sub['stratum_n'].iloc[0]):,})", fontsize=17, pad=10)
    ax.set_xlabel("Metabolic burden", fontsize=16)
    ax.set_ylabel("Steatosis severity", fontsize=16)
    ax.set_yticklabels(["No steatosis", "Mild steatosis", "Moderate/severe"], rotation=0)
    _style_axes(ax, fontsize=15)
    hm.collections[0].colorbar.set_label("Adjusted probability (%)", fontsize=16)
    hm.collections[0].colorbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    return fig


def _make_trajectory_panel(lines: pd.DataFrame) -> plt.Figure:
    phenotype_colors = {
        "0_low": "#355070",
        "0_high": "#6d597a",
        "2_high": "#b56576",
    }
    x_map = {status: idx for idx, status in enumerate(GLYCEMIC_ORDER)}
    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    for phenotype_label, sub in lines.groupby("phenotype_label", sort=False):
        sub = sub.copy().sort_values("glycemic_status")
        sub["x"] = sub["glycemic_status"].map(x_map)
        ax.plot(
            sub["x"],
            sub["adjusted_probability_pct"],
            marker="o",
            linewidth=2.8,
            markersize=7.5,
            color=phenotype_colors[phenotype_label],
            label=phenotype_label.replace("_", "-"),
        )

    ax.text(
        -0.10,
        1.08,
        "D",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_xticks([0, 1, 2], [GLYCEMIC_LABELS[s] for s in GLYCEMIC_ORDER], rotation=0, ha="center")
    ax.set_ylabel("Adjusted proteinuria probability (%)", fontsize=16)
    ax.set_xlabel("Glycemic status", fontsize=16)
    ax.set_title("Selected phenotype trajectories", fontsize=17, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, title="Phenotype", fontsize=13.5, title_fontsize=14, loc="upper left")
    _style_axes(ax, fontsize=15)
    fig.tight_layout()
    return fig


def _make_contrast_panel(contrasts: pd.DataFrame) -> plt.Figure:
    contrast_colors = {
        "0_high vs 0_low": "#6d597a",
        "2_high vs 0_low": "#b56576",
    }
    y_base = {status: idx for idx, status in enumerate(reversed(GLYCEMIC_ORDER))}
    offset_map = {"0_high vs 0_low": -0.12, "2_high vs 0_low": 0.12}

    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    for comparison in ["0_high vs 0_low", "2_high vs 0_low"]:
        sub = contrasts.loc[contrasts["comparison"] == comparison].copy()
        y = [y_base[s] + offset_map[comparison] for s in sub["glycemic_status"]]
        ax.scatter(
            sub["absolute_risk_difference_pp"],
            y,
            s=70,
            color=contrast_colors[comparison],
            label=comparison,
            zorder=3,
        )
        for x, yy in zip(sub["absolute_risk_difference_pp"], y):
            ax.plot([0, x], [yy, yy], color=contrast_colors[comparison], linewidth=2, alpha=0.8)

    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(list(y_base.values()), [GLYCEMIC_LABELS[s] for s in reversed(GLYCEMIC_ORDER)])
    ax.text(
        -0.08,
        1.08,
        "E",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_xlabel("Absolute risk difference for proteinuria (pp)", fontsize=16)
    ax.set_title("Key contrast amplification", fontsize=17, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=13)
    ax.grid(axis="x", alpha=0.2)
    _style_axes(ax, fontsize=15)
    fig.tight_layout()
    return fig


def main() -> None:
    ensure_dirs()
    full_predictions = pd.read_csv(FIGURE_DATA_DIR / "figure3_proteinuria_glycemic_stratified_predictions.csv")
    lines = pd.read_csv(FIGURE_DATA_DIR / "figure3_interface_selected_phenotypes.csv")
    contrasts = pd.read_csv(FIGURE_DATA_DIR / "figure3_interface_selected_contrasts.csv")
    vmax = full_predictions["adjusted_probability_pct"].max()

    panel_specs = [
        ("normoglycemia", "A", "figure3_A_normoglycemia_heatmap"),
        ("intermediate_hyperglycemia", "B", "figure3_B_intermediate_hyperglycemia_heatmap"),
        ("diabetes_level_hyperglycemia", "C", "figure3_C_diabetes_level_hyperglycemia_heatmap"),
    ]
    for status, letter, stem in panel_specs:
        fig = _make_heatmap_panel(full_predictions, status, letter, vmax)
        _save_panel(fig, stem)
        plt.close(fig)

    fig_d = _make_trajectory_panel(lines)
    _save_panel(fig_d, "figure3_D_selected_phenotype_trajectories")
    plt.close(fig_d)

    fig_e = _make_contrast_panel(contrasts)
    _save_panel(fig_e, "figure3_E_key_contrast_amplification")
    plt.close(fig_e)


if __name__ == "__main__":
    main()
