from __future__ import annotations

from rebuild_lib import (
    RESULT_DIR,
    TABLE_DIR,
    build_baseline_table,
    ensure_dirs,
    flow_summary,
    load_source_cohort,
    prepare_echo_subset,
    prepare_full_analytic_cohort,
    save_csv,
)


def main() -> None:
    ensure_dirs()
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)
    echo_subset = prepare_echo_subset(full_cohort)

    save_csv(flow_summary(source_df, full_cohort, echo_subset), RESULT_DIR / "flow_summary.csv")
    save_csv(build_baseline_table(full_cohort), TABLE_DIR / "table1_baseline_by_glycemic_status.csv")


if __name__ == "__main__":
    main()

