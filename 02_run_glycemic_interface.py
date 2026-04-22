from __future__ import annotations

import pandas as pd

from rebuild_lib import (
    FIGURE_DATA_DIR,
    GLYCEMIC_ORDER,
    GLYCEMIC_LABELS,
    RESULT_DIR,
    SELECTED_INTERFACE_PHENOTYPES,
    TABLE_DIR,
    collapsed_cardio_kidney_glycemic_summary,
    ensure_dirs,
    load_source_cohort,
    prepare_echo_subset,
    prepare_full_analytic_cohort,
    run_proteinuria_glycemic_strata,
    run_proteinuria_three_way_interaction,
    save_csv,
)


def main() -> None:
    ensure_dirs()
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)
    echo_subset = prepare_echo_subset(full_cohort)

    stratified_grids, stratified_predictions, stratified_contrasts, stratified_tests = run_proteinuria_glycemic_strata(
        full_cohort
    )
    save_csv(stratified_grids, RESULT_DIR / "glycemic_stratified_proteinuria_grid_summary.csv")
    save_csv(stratified_predictions, RESULT_DIR / "glycemic_stratified_proteinuria_marginal_predictions.csv")
    save_csv(stratified_contrasts, RESULT_DIR / "glycemic_stratified_proteinuria_key_contrasts.csv")
    save_csv(stratified_tests, RESULT_DIR / "glycemic_stratified_proteinuria_interaction_tests.csv")
    save_csv(run_proteinuria_three_way_interaction(full_cohort), RESULT_DIR / "proteinuria_three_way_interaction_test.csv")
    save_csv(
        collapsed_cardio_kidney_glycemic_summary(echo_subset),
        RESULT_DIR / "supplement_cardio_kidney_collapsed_glycemic_summary.csv",
    )

    save_csv(
        stratified_predictions,
        FIGURE_DATA_DIR / "figure3_proteinuria_glycemic_stratified_predictions.csv",
    )
    figure3_lines = []
    for phenotype_label, stea_grade, burden in SELECTED_INTERFACE_PHENOTYPES:
        sub = stratified_predictions.loc[
            (stratified_predictions["steatosis_grade"] == stea_grade)
            & (stratified_predictions["non_glycemic_burden_group"] == burden)
        ].copy()
        sub["phenotype_label"] = phenotype_label
        figure3_lines.append(sub)
    save_csv(
        pd.concat(figure3_lines, ignore_index=True),
        FIGURE_DATA_DIR / "figure3_interface_selected_phenotypes.csv",
    )
    save_csv(
        stratified_contrasts.loc[stratified_contrasts["comparison"].isin(["0_high vs 0_low", "2_high vs 0_low"])].copy(),
        FIGURE_DATA_DIR / "figure3_interface_selected_contrasts.csv",
    )
    supplement_27 = stratified_grids.merge(
        stratified_predictions[
            [
                "glycemic_status",
                "steatosis_grade",
                "non_glycemic_burden_group",
                "adjusted_probability",
                "adjusted_probability_pct",
            ]
        ],
        on=["glycemic_status", "steatosis_grade", "non_glycemic_burden_group"],
        how="left",
    )
    save_csv(supplement_27, TABLE_DIR / "supplementary_table_s1_27cell_proteinuria_summary.csv")

    rows = []
    for status in GLYCEMIC_ORDER:
        label = GLYCEMIC_LABELS[status]
        pred_sub = stratified_predictions.loc[stratified_predictions["glycemic_status"] == status].copy()
        con_sub = stratified_contrasts.loc[stratified_contrasts["glycemic_status"] == status].copy()
        row = {
            "Glycemic stratum": label,
            "N": int(pred_sub["stratum_n"].iloc[0]),
        }
        for phenotype_label, stea_grade, burden in SELECTED_INTERFACE_PHENOTYPES:
            prob = float(
                pred_sub.loc[
                    (pred_sub["steatosis_grade"] == stea_grade)
                    & (pred_sub["non_glycemic_burden_group"] == burden),
                    "adjusted_probability_pct",
                ].iloc[0]
            )
            row[f"{phenotype_label} adjusted probability (%)"] = prob
        for comparison in ["0_high vs 0_low", "2_high vs 0_low"]:
            con_row = con_sub.loc[con_sub["comparison"] == comparison].iloc[0]
            row[f"{comparison} ARD (pp)"] = float(con_row["absolute_risk_difference_pp"])
            row[f"{comparison} OR (95% CI)"] = (
                f"{con_row['or']:.2f} ({con_row['or_ci_low']:.2f}-{con_row['or_ci_high']:.2f})"
            )
        row["Within-stratum interaction p-value"] = float(
            stratified_tests.loc[stratified_tests["glycemic_status"] == status, "p_value"].iloc[0]
        )
        rows.append(row)
    table3 = pd.DataFrame(rows)
    save_csv(table3, TABLE_DIR / "table3_proteinuria_glycemic_key_contrasts.csv")


if __name__ == "__main__":
    main()
