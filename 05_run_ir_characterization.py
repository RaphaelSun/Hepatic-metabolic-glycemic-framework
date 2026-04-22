from __future__ import annotations

import pandas as pd

from rebuild_lib import (
    FIGURE_DATA_DIR,
    RESULT_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_source_cohort,
    prepare_full_analytic_cohort,
    run_ir_characterization,
    save_csv,
)


def main() -> None:
    ensure_dirs()
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)

    crude_grid, adjusted_grid, interaction_tests = run_ir_characterization(full_cohort)
    save_csv(crude_grid, RESULT_DIR / "ir_proxy_crude_grid_summary.csv")
    save_csv(adjusted_grid, RESULT_DIR / "ir_proxy_adjusted_grid_summary.csv")
    save_csv(interaction_tests, RESULT_DIR / "ir_proxy_interaction_tests.csv")

    save_csv(adjusted_grid, FIGURE_DATA_DIR / "figure2_ir_proxy_adjusted_grid.csv")

    table = adjusted_grid.pivot_table(
        index=["steatosis_label", "non_glycemic_burden_group"],
        columns="outcome_label",
        values="adjusted_value",
    ).reset_index()
    save_csv(table, TABLE_DIR / "supplementary_table_s2_ir_proxy_by_phenotype.csv")


if __name__ == "__main__":
    main()
