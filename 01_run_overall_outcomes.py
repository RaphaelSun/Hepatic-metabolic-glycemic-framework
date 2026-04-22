from __future__ import annotations

import pandas as pd

from rebuild_lib import (
    FIGURE_DATA_DIR,
    RESULT_DIR,
    TABLE_DIR,
    cardio_kidney_composition_summary,
    ensure_dirs,
    load_source_cohort,
    prepare_echo_subset,
    prepare_full_analytic_cohort,
    run_overall_outcome_analysis,
    save_csv,
    sparse_cells,
)


def main() -> None:
    ensure_dirs()
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)
    echo_subset = prepare_echo_subset(full_cohort)

    proteinuria = run_overall_outcome_analysis(full_cohort, "proteinuria", "full_analytic_cohort")
    cardio_kidney = run_overall_outcome_analysis(
        echo_subset,
        "cardio_kidney_coabnormality",
        "echo_overlap_subset",
    )
    diastolic = run_overall_outcome_analysis(
        echo_subset,
        "diastolic_dysfunction",
        "echo_subset",
    )

    for analysis in [proteinuria, cardio_kidney, diastolic]:
        prefix = analysis.outcome
        save_csv(analysis.grid_summary, RESULT_DIR / f"overall_{prefix}_grid_summary.csv")
        save_csv(analysis.marginal_predictions, RESULT_DIR / f"overall_{prefix}_marginal_predictions.csv")
        save_csv(analysis.key_contrasts, RESULT_DIR / f"overall_{prefix}_key_contrasts.csv")
        save_csv(analysis.interaction_test, RESULT_DIR / f"overall_{prefix}_interaction_test.csv")
        save_csv(sparse_cells(analysis.grid_summary), RESULT_DIR / f"overall_{prefix}_sparse_cells.csv")

    save_csv(
        proteinuria.marginal_predictions,
        FIGURE_DATA_DIR / "figure4_overall_proteinuria_lines.csv",
    )
    save_csv(
        cardio_kidney_composition_summary(echo_subset),
        FIGURE_DATA_DIR / "figure5_cardio_kidney_composition.csv",
    )
    save_csv(
        diastolic.marginal_predictions,
        FIGURE_DATA_DIR / "figure6_diastolic_grouped.csv",
    )

    table2 = pd.concat(
        [
            proteinuria.key_contrasts.assign(section="Co-primary: Proteinuria"),
            cardio_kidney.key_contrasts.assign(section="Co-primary: Cardio-kidney co-abnormality"),
            diastolic.key_contrasts.assign(section="Secondary: Diastolic dysfunction"),
        ],
        ignore_index=True,
    )
    interaction_lookup = {
        "proteinuria": float(proteinuria.interaction_test["p_value"].iloc[0]),
        "cardio_kidney_coabnormality": float(cardio_kidney.interaction_test["p_value"].iloc[0]),
        "diastolic_dysfunction": float(diastolic.interaction_test["p_value"].iloc[0]),
    }
    cohort_lookup = {
        "proteinuria": ("Full analytic cohort", len(full_cohort)),
        "cardio_kidney_coabnormality": ("Echo overlap subset", len(echo_subset)),
        "diastolic_dysfunction": ("Echo subset", len(echo_subset)),
    }
    table2["cohort_label"] = table2["outcome"].map(lambda x: cohort_lookup[x][0])
    table2["cohort_n"] = table2["outcome"].map(lambda x: cohort_lookup[x][1])
    table2["interaction_p_value"] = table2["outcome"].map(interaction_lookup)
    table2["or_95ci"] = table2.apply(
        lambda row: f"{row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f})",
        axis=1,
    )
    table2 = table2[
        [
            "section",
            "cohort_label",
            "cohort_n",
            "comparison",
            "ref_adjusted_pct",
            "comp_adjusted_pct",
            "absolute_risk_difference_pp",
            "risk_ratio",
            "or_95ci",
            "interaction_p_value",
        ]
    ].rename(
        columns={
            "section": "Section",
            "cohort_label": "Cohort",
            "cohort_n": "N",
            "comparison": "Comparison",
            "ref_adjusted_pct": "Reference adjusted probability (%)",
            "comp_adjusted_pct": "Comparison adjusted probability (%)",
            "absolute_risk_difference_pp": "Absolute risk difference (pp)",
            "risk_ratio": "Risk ratio",
            "or_95ci": "Odds ratio (95% CI)",
            "interaction_p_value": "Interaction p-value",
        }
    )
    save_csv(table2, TABLE_DIR / "table2_overall_key_contrasts.csv")


if __name__ == "__main__":
    main()
