from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.special import expit
from scipy.stats import chi2

from rebuild_lib import FIGURE_DATA_DIR, FIGURE_DIR, RESULT_DIR, STEATOSIS_LABELS, ensure_dirs, load_source_cohort, prepare_full_analytic_cohort, save_csv


COLOR_MAP = {
    "No steatosis": "#355070",
    "Mild steatosis": "#e07a5f",
    "Moderate/severe steatosis": "#8d0801",
}
BURDEN_ORDER = ["low", "intermediate", "high"]
BURDEN_TICKS = ["Low", "Intermediate", "High"]
PROXY_CONFIG = [
    ("tyg", "TyG", "#355070"),
    ("mets_ir", "METS-IR", "#8d0801"),
]
ANNOTATION_OFFSETS = {
    0: (0, -12),
    1: (12, 0),
    2: (0, 12),
}


def _fit_rcs_curve(df: pd.DataFrame, outcome_label: str, proxy_col: str, proxy_label: str, color: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    formula = f"proteinuria ~ cr({proxy_col}, df=4) + age + sex_binary"
    spline_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    linear_model = smf.glm(
        f"proteinuria ~ {proxy_col} + age + sex_binary",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    null_model = smf.glm("proteinuria ~ age + sex_binary", data=df, family=sm.families.Binomial()).fit()

    grid = np.linspace(df[proxy_col].quantile(0.01), df[proxy_col].quantile(0.99), 140)
    ref = pd.DataFrame(
        {
            proxy_col: grid,
            "age": float(df["age"].mean()),
            "sex_binary": float(df["sex_binary"].mean()),
        }
    )
    design = build_design_matrices([spline_model.model.data.design_info], ref, return_type="dataframe")[0].to_numpy()
    params = spline_model.params.to_numpy(dtype=float)
    cov = spline_model.cov_params().to_numpy(dtype=float)
    lp = design @ params
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    curve = pd.DataFrame(
        {
            "outcome_label": outcome_label,
            "proxy_col": proxy_col,
            "proxy_label": proxy_label,
            "x": grid,
            "predicted_probability_pct": expit(lp) * 100,
            "ci_low_pct": expit(lp - 1.96 * se) * 100,
            "ci_high_pct": expit(lp + 1.96 * se) * 100,
        }
    )
    tests = pd.DataFrame(
        [
            {
                "outcome_label": outcome_label,
                "proxy_col": proxy_col,
                "proxy_label": proxy_label,
                "overall_p_value": float(
                    chi2.sf(2.0 * (spline_model.llf - null_model.llf), int(spline_model.df_model - null_model.df_model))
                ),
                "nonlinearity_p_value": float(
                    chi2.sf(2.0 * (spline_model.llf - linear_model.llf), int(spline_model.df_model - linear_model.df_model))
                ),
                "n": int(len(df)),
            }
        ]
    )
    return curve, tests


def main() -> None:
    ensure_dirs()
    data = pd.read_csv(FIGURE_DATA_DIR / "figure2_ir_proxy_adjusted_grid.csv")
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)

    rcs_curves = []
    rcs_tests = []
    for proxy_col, proxy_label, color in PROXY_CONFIG:
        curve, tests = _fit_rcs_curve(full_cohort, "proteinuria", proxy_col, proxy_label, color)
        rcs_curves.append(curve)
        rcs_tests.append(tests)

    curve_df = pd.concat(rcs_curves, ignore_index=True)
    tests_df = pd.concat(rcs_tests, ignore_index=True)
    save_csv(curve_df, FIGURE_DATA_DIR / "figure2_ir_proxy_rcs_curve.csv")
    save_csv(tests_df, RESULT_DIR / "figure2_ir_proxy_rcs_tests.csv")

    fig = plt.figure(figsize=(13.8, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.38, wspace=0.28)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    x_map = {name: idx for idx, name in enumerate(BURDEN_ORDER)}

    for ax, outcome_label, title_letter in zip(axes[:2], ["TyG", "METS-IR"], ["A", "B"]):
        sub = data.loc[data["outcome_label"] == outcome_label].copy()
        for stea_grade in [0, 1, 2]:
            stea_sub = sub.loc[sub["steatosis_grade"] == stea_grade].copy()
            stea_sub["x"] = stea_sub["non_glycemic_burden_group"].map(x_map)
            label = STEATOSIS_LABELS[stea_grade]
            ax.plot(
                stea_sub["x"],
                stea_sub["adjusted_value"],
                marker="o",
                linewidth=2.4,
                markersize=7,
                color=COLOR_MAP[label],
                label=label,
            )
            value_fmt = "{:.2f}" if outcome_label == "TyG" else "{:.1f}"
            dx, dy = ANNOTATION_OFFSETS[stea_grade]
            for row in stea_sub.itertuples(index=False):
                ax.annotate(
                    value_fmt.format(row.adjusted_value),
                    (row.x, row.adjusted_value),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=COLOR_MAP[label],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.15),
                )
        ax.set_xticks([0, 1, 2], BURDEN_TICKS)
        ax.set_xlabel("Non-glycemic burden")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{title_letter}. {outcome_label} across phenotype", fontsize=11.5, pad=8)

    axes[0].set_ylabel("Age/sex-adjusted TyG")
    axes[1].set_ylabel("Age/sex-adjusted METS-IR")

    for ax, (proxy_col, proxy_label, color), letter in zip(axes[2:], PROXY_CONFIG, ["C", "D"]):
        curve = curve_df.loc[curve_df["proxy_col"] == proxy_col].copy()
        ax.fill_between(
            curve["x"],
            curve["ci_low_pct"],
            curve["ci_high_pct"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(curve["x"], curve["predicted_probability_pct"], color=color, linewidth=2.4)
        ax.set_xlabel(proxy_label)
        ax.set_ylabel("Adjusted proteinuria probability (%)")
        ax.set_title(f"{letter}. {proxy_label} spline vs proteinuria", fontsize=11.5, pad=8)
        ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.90, wspace=0.28, hspace=0.4)
    fig.savefig(FIGURE_DIR / "figure2_ir_support.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure2_ir_support.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
