from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import chi2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COHORT_PATH = PROJECT_ROOT / "outputs" / "cohorts" / "hospital_main_cohort_frozen.csv.gz"
SOURCE_FLOW_TABLE_PATH = PROJECT_ROOT / "outputs" / "tables" / "tableS1_variable_availability_and_flow.csv"

RESULT_DIR = NEW_ROOT / "outputs" / "results"
FIGURE_DATA_DIR = NEW_ROOT / "outputs" / "figure_data"
FIGURE_DIR = NEW_ROOT / "outputs" / "figures"
TABLE_DIR = NEW_ROOT / "outputs" / "tables"
DRAFT_DIR = NEW_ROOT / "drafts"

STEATOSIS_ORDER = [0, 1, 2]
STEATOSIS_LABELS = {
    0: "No steatosis",
    1: "Mild steatosis",
    2: "Moderate/severe steatosis",
}
BURDEN_ORDER = ["low", "intermediate", "high"]
BURDEN_LABELS = {
    "low": "Low",
    "intermediate": "Intermediate",
    "high": "High",
}
GLYCEMIC_ORDER = [
    "normoglycemia",
    "intermediate_hyperglycemia",
    "diabetes_level_hyperglycemia",
]
GLYCEMIC_LABELS = {
    "normoglycemia": "Normoglycemia",
    "intermediate_hyperglycemia": "Intermediate hyperglycemia",
    "diabetes_level_hyperglycemia": "Diabetes-level hyperglycemia",
}
GLYCEMIC_COLLAPSED_ORDER = [
    "non_diabetes_level_hyperglycemia",
    "diabetes_level_hyperglycemia",
]
GLYCEMIC_COLLAPSED_LABELS = {
    "non_diabetes_level_hyperglycemia": "Non-diabetes-level hyperglycemia",
    "diabetes_level_hyperglycemia": "Diabetes-level hyperglycemia",
}
PHENOTYPE_ORDER = [
    "0_low",
    "0_intermediate",
    "0_high",
    "1_low",
    "1_intermediate",
    "1_high",
    "2_low",
    "2_intermediate",
    "2_high",
]
SELECTED_INTERFACE_PHENOTYPES = [
    ("0_low", 0, "low"),
    ("0_high", 0, "high"),
    ("2_high", 2, "high"),
]
KEY_COMPARISONS = [
    ("2_high vs 0_low", 0, "low", 2, "high"),
    ("2_high vs 0_high", 0, "high", 2, "high"),
    ("0_high vs 0_low", 0, "low", 0, "high"),
    ("1_high vs 1_low", 1, "low", 1, "high"),
]
FULL_REQUIRED_COLUMNS = [
    "hepatic_steatosis_grade_final",
    "proteinuria_pos_final",
    "age",
    "sex_binary",
    "bmi",
    "sbp",
    "dbp",
    "glucose",
    "tg",
    "hdl",
    "ua",
]
IR_OUTCOMES = [("tyg", "TyG"), ("mets_ir", "METS-IR")]


@dataclass
class OutcomeAnalysis:
    outcome: str
    cohort_name: str
    model: sm.GLM
    reduced_model: sm.GLM
    grid_summary: pd.DataFrame
    marginal_predictions: pd.DataFrame
    key_contrasts: pd.DataFrame
    interaction_test: pd.DataFrame


def ensure_dirs() -> None:
    for path in [RESULT_DIR, FIGURE_DATA_DIR, FIGURE_DIR, TABLE_DIR, DRAFT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_source_cohort() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_COHORT_PATH)
    numeric_cols = [
        "age",
        "sex_binary",
        "hepatic_steatosis_grade_final",
        "proteinuria_pos_final",
        "echo_diastolic_abnormal_final",
        "bmi",
        "sbp",
        "dbp",
        "glucose",
        "tg",
        "hdl",
        "ua",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def derive_non_glycemic_burden(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ng_high_bmi"] = (out["bmi"] >= 25).astype(int)
    out["ng_high_bp"] = ((out["sbp"] >= 130) | (out["dbp"] >= 85)).astype(int)
    out["ng_high_tg_low_hdl"] = (
        (out["tg"] >= 1.7)
        | (((out["sex_binary"] == 1) & (out["hdl"] < 1.0)) | ((out["sex_binary"] == 0) & (out["hdl"] < 1.3)))
    ).astype(int)
    out["ng_hyperuricemia"] = (
        ((out["sex_binary"] == 1) & (out["ua"] > 420))
        | ((out["sex_binary"] == 0) & (out["ua"] > 360))
    ).astype(int)
    out["non_glycemic_burden_score"] = out[
        ["ng_high_bmi", "ng_high_bp", "ng_high_tg_low_hdl", "ng_hyperuricemia"]
    ].sum(axis=1)
    out["non_glycemic_burden_group"] = pd.Categorical(
        pd.cut(
            out["non_glycemic_burden_score"],
            bins=[-1, 1, 2, 4],
            labels=BURDEN_ORDER,
        ),
        categories=BURDEN_ORDER,
        ordered=True,
    )
    return out


def add_shared_phenotype_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_non_glycemic_burden(df)
    out["steatosis_grade"] = pd.Categorical(
        out["hepatic_steatosis_grade_final"].astype(int),
        categories=STEATOSIS_ORDER,
        ordered=True,
    )
    out["steatosis_label"] = out["steatosis_grade"].astype(int).map(STEATOSIS_LABELS)
    out["glycemic_status"] = pd.Categorical(
        pd.cut(
            out["glucose"],
            bins=[-np.inf, 5.6, 7.0, np.inf],
            right=False,
            labels=GLYCEMIC_ORDER,
        ),
        categories=GLYCEMIC_ORDER,
        ordered=True,
    )
    out["glycemic_label"] = out["glycemic_status"].astype(str).map(GLYCEMIC_LABELS)
    out["glycemic_status_collapsed"] = pd.Categorical(
        np.where(out["glucose"] >= 7.0, "diabetes_level_hyperglycemia", "non_diabetes_level_hyperglycemia"),
        categories=GLYCEMIC_COLLAPSED_ORDER,
        ordered=True,
    )
    out["proteinuria"] = out["proteinuria_pos_final"].astype(int)
    out["phenotype_id"] = (
        out["steatosis_grade"].astype(int).astype(str) + "_" + out["non_glycemic_burden_group"].astype(str)
    )
    glucose_mg = out["glucose"] * 18.0
    tg_mg = out["tg"] * 88.57
    hdl_mg = out["hdl"] * 38.67
    out["tyg"] = np.log((tg_mg * glucose_mg) / 2.0)
    out["mets_ir"] = np.log(2.0 * glucose_mg + tg_mg) * out["bmi"] / np.log(hdl_mg)
    return out


def prepare_full_analytic_cohort(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_source_cohort()
    cohort = df.dropna(subset=FULL_REQUIRED_COLUMNS).copy()
    cohort = add_shared_phenotype_columns(cohort)
    return cohort


def prepare_echo_subset(full_cohort: pd.DataFrame) -> pd.DataFrame:
    subset = full_cohort.loc[full_cohort["echo_diastolic_abnormal_final"].isin([0, 1])].copy()
    subset["diastolic_dysfunction"] = subset["echo_diastolic_abnormal_final"].astype(int)
    subset["cardio_kidney_coabnormality"] = (
        (subset["diastolic_dysfunction"] == 1) & (subset["proteinuria"] == 1)
    ).astype(int)
    subset["cardio_kidney_composition"] = np.select(
        [
            (subset["proteinuria"] == 0) & (subset["diastolic_dysfunction"] == 0),
            (subset["proteinuria"] == 1) & (subset["diastolic_dysfunction"] == 0),
            (subset["proteinuria"] == 0) & (subset["diastolic_dysfunction"] == 1),
            (subset["proteinuria"] == 1) & (subset["diastolic_dysfunction"] == 1),
        ],
        [
            "neither",
            "kidney_only",
            "cardiac_only",
            "both",
        ],
        default="neither",
    )
    return subset


def flow_summary(source_df: pd.DataFrame, full_cohort: pd.DataFrame, echo_subset: pd.DataFrame) -> pd.DataFrame:
    echo_text_available = int(source_df["echo_text"].notna().sum())
    raw_participants = int(len(source_df))
    if SOURCE_FLOW_TABLE_PATH.exists():
        legacy_flow = pd.read_csv(SOURCE_FLOW_TABLE_PATH)
        raw_row = legacy_flow.loc[
            (legacy_flow["panel"] == "Flow: hospital") & (legacy_flow["item"] == "raw_hospital_visits"),
            "participants",
        ]
        if not raw_row.empty and pd.notna(raw_row.iloc[0]):
            raw_participants = int(raw_row.iloc[0])
    excluded_missing_main = raw_participants - int(len(source_df))
    excluded_missing_burden = int(len(source_df)) - int(len(full_cohort))
    excluded_unclassifiable_echo = echo_text_available - int(len(echo_subset))
    return pd.DataFrame(
        [
            {"step_order": 1, "step": "Participants in hospital dataset", "n": raw_participants},
            {
                "step_order": 2,
                "step": "Excluded: missing ultrasound/proteinuria data",
                "n": excluded_missing_main,
            },
            {
                "step_order": 3,
                "step": "Main hospital cohort",
                "n": int(len(source_df)),
            },
            {
                "step_order": 4,
                "step": "Excluded: missing non-glycemic burden data",
                "n": excluded_missing_burden,
            },
            {
                "step_order": 5,
                "step": "Phenotype-based proteinuria, glycemic, and IR analyses",
                "n": int(len(full_cohort)),
            },
            {
                "step_order": 6,
                "step": "Echocardiographic subset",
                "n": echo_text_available,
            },
            {
                "step_order": 7,
                "step": "Excluded: incomplete or unclassifiable echo data",
                "n": excluded_unclassifiable_echo,
            },
            {
                "step_order": 8,
                "step": "Cardio-kidney co-abnormality and diastolic dysfunction analyses",
                "n": int(len(echo_subset)),
            },
        ]
    )


def summarize_series(series: pd.Series) -> str:
    return f"{series.mean():.2f} ({series.std(ddof=1):.2f})"


def summarize_count_pct(mask: pd.Series) -> str:
    return f"{int(mask.sum())} ({mask.mean() * 100:.1f}%)"


def build_baseline_table(full_cohort: pd.DataFrame) -> pd.DataFrame:
    columns = [("Overall", full_cohort)] + [
        (GLYCEMIC_LABELS[level], full_cohort.loc[full_cohort["glycemic_status"] == level].copy())
        for level in GLYCEMIC_ORDER
    ]
    rows: list[dict[str, str]] = []

    def add_row(variable: str, formatter) -> None:
        row = {"Variable": variable}
        for label, sub in columns:
            row[label] = formatter(sub)
        rows.append(row)

    add_row("N", lambda sub: f"{len(sub):,}")
    add_row("Age, years", lambda sub: summarize_series(sub["age"]))
    add_row("Female, n (%)", lambda sub: summarize_count_pct(sub["sex_binary"] == 0))
    add_row("BMI, kg/m^2", lambda sub: summarize_series(sub["bmi"]))
    add_row("SBP, mmHg", lambda sub: summarize_series(sub["sbp"]))
    add_row("DBP, mmHg", lambda sub: summarize_series(sub["dbp"]))
    add_row("Fasting glucose, mmol/L", lambda sub: summarize_series(sub["glucose"]))
    add_row("Triglycerides, mmol/L", lambda sub: summarize_series(sub["tg"]))
    add_row("HDL-C, mmol/L", lambda sub: summarize_series(sub["hdl"]))
    add_row("Uric acid, umol/L", lambda sub: summarize_series(sub["ua"]))
    add_row("High BMI, n (%)", lambda sub: summarize_count_pct(sub["ng_high_bmi"] == 1))
    add_row("High BP, n (%)", lambda sub: summarize_count_pct(sub["ng_high_bp"] == 1))
    add_row("High TG or low HDL-C, n (%)", lambda sub: summarize_count_pct(sub["ng_high_tg_low_hdl"] == 1))
    add_row("Hyperuricemia, n (%)", lambda sub: summarize_count_pct(sub["ng_hyperuricemia"] == 1))
    add_row("No steatosis, n (%)", lambda sub: summarize_count_pct(sub["steatosis_grade"].astype(int) == 0))
    add_row("Mild steatosis, n (%)", lambda sub: summarize_count_pct(sub["steatosis_grade"].astype(int) == 1))
    add_row(
        "Moderate/severe steatosis, n (%)",
        lambda sub: summarize_count_pct(sub["steatosis_grade"].astype(int) == 2),
    )
    add_row(
        "Non-glycemic burden low, n (%)",
        lambda sub: summarize_count_pct(sub["non_glycemic_burden_group"] == "low"),
    )
    add_row(
        "Non-glycemic burden intermediate, n (%)",
        lambda sub: summarize_count_pct(sub["non_glycemic_burden_group"] == "intermediate"),
    )
    add_row(
        "Non-glycemic burden high, n (%)",
        lambda sub: summarize_count_pct(sub["non_glycemic_burden_group"] == "high"),
    )
    add_row("Proteinuria positive, n (%)", lambda sub: summarize_count_pct(sub["proteinuria"] == 1))
    add_row("TyG", lambda sub: summarize_series(sub["tyg"]))
    add_row("METS-IR", lambda sub: summarize_series(sub["mets_ir"]))
    return pd.DataFrame(rows)


def outcome_formula(outcome: str, interaction: bool = True) -> str:
    base = "C(steatosis_grade, Treatment(reference=0)) + C(non_glycemic_burden_group, Treatment(reference='low'))"
    if interaction:
        base = base.replace(" + ", " * ", 1)
    return f"{outcome} ~ {base} + age + sex_binary"


def fit_glm_binomial(df: pd.DataFrame, outcome: str, interaction: bool = True):
    return smf.glm(formula=outcome_formula(outcome, interaction=interaction), data=df, family=sm.families.Binomial()).fit()


def interaction_test(full_model, reduced_model, outcome: str, cohort_name: str) -> pd.DataFrame:
    lr = 2 * (full_model.llf - reduced_model.llf)
    df_diff = int(full_model.df_model - reduced_model.df_model)
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "cohort": cohort_name,
                "test": "steatosis_by_burden_interaction",
                "statistic": float(lr),
                "df_diff": df_diff,
                "p_value": float(chi2.sf(lr, df_diff)),
                "n": int(full_model.nobs),
            }
        ]
    )


def standardized_prediction(model, df: pd.DataFrame, steatosis_grade: int, burden: str) -> float:
    cf = df.copy()
    cf["steatosis_grade"] = pd.Categorical([steatosis_grade] * len(cf), categories=STEATOSIS_ORDER, ordered=True)
    cf["non_glycemic_burden_group"] = pd.Categorical(
        [burden] * len(cf),
        categories=BURDEN_ORDER,
        ordered=True,
    )
    return float(model.predict(cf).mean())


def build_grid_summary(df: pd.DataFrame, outcome: str, cohort_name: str) -> pd.DataFrame:
    grid = (
        df.groupby(["steatosis_grade", "non_glycemic_burden_group"], observed=False)
        .agg(
            n=("participant_id", "size"),
            events=(outcome, "sum"),
            crude_rate=(outcome, "mean"),
            mean_age=("age", "mean"),
            female_pct=("sex_binary", lambda s: float((s == 0).mean() * 100)),
        )
        .reset_index()
    )
    grid["outcome"] = outcome
    grid["cohort"] = cohort_name
    grid["steatosis_label"] = grid["steatosis_grade"].astype(int).map(STEATOSIS_LABELS)
    grid["burden_label"] = grid["non_glycemic_burden_group"].astype(str).map(BURDEN_LABELS)
    grid["crude_rate_pct"] = grid["crude_rate"] * 100
    grid["grid_id"] = (
        grid["steatosis_grade"].astype(int).astype(str)
        + "_"
        + grid["non_glycemic_burden_group"].astype(str)
    )
    return grid


def build_marginal_predictions(df: pd.DataFrame, model, outcome: str, cohort_name: str) -> pd.DataFrame:
    rows = []
    for steatosis_grade in STEATOSIS_ORDER:
        for burden in BURDEN_ORDER:
            adjusted = standardized_prediction(model, df, steatosis_grade, burden)
            rows.append(
                {
                    "outcome": outcome,
                    "cohort": cohort_name,
                    "steatosis_grade": steatosis_grade,
                    "non_glycemic_burden_group": burden,
                    "adjusted_probability": adjusted,
                    "adjusted_probability_pct": adjusted * 100,
                    "steatosis_label": STEATOSIS_LABELS[steatosis_grade],
                    "burden_label": BURDEN_LABELS[burden],
                }
            )
    return pd.DataFrame(rows)


def _reference_values(df: pd.DataFrame, steatosis_grade: int, burden: str) -> dict[str, float | str]:
    return {
        "steatosis_grade": steatosis_grade,
        "non_glycemic_burden_group": burden,
        "age": float(df["age"].mean()),
        "sex_binary": 0,
    }


def _design_row(model, values: dict[str, float | str]) -> np.ndarray:
    design_info = model.model.data.design_info
    row = build_design_matrices([design_info], pd.DataFrame([values]), return_type="dataframe")[0]
    return row.iloc[0].to_numpy(dtype=float)


def model_or_contrast(
    model,
    df: pd.DataFrame,
    comparison: str,
    ref_grade: int,
    ref_burden: str,
    comp_grade: int,
    comp_burden: str,
) -> dict[str, float | str]:
    ref_row = _design_row(model, _reference_values(df, ref_grade, ref_burden))
    comp_row = _design_row(model, _reference_values(df, comp_grade, comp_burden))
    diff = comp_row - ref_row
    params = model.params.to_numpy(dtype=float)
    cov = model.cov_params().to_numpy(dtype=float)
    estimate = float(diff @ params)
    se = float(np.sqrt(diff @ cov @ diff))
    ci_low = estimate - 1.96 * se
    ci_high = estimate + 1.96 * se
    return {
        "comparison": comparison,
        "or": float(np.exp(estimate)),
        "or_ci_low": float(np.exp(ci_low)),
        "or_ci_high": float(np.exp(ci_high)),
    }


def build_key_contrasts(
    predictions: pd.DataFrame,
    model,
    df: pd.DataFrame,
    outcome: str,
    cohort_name: str,
    glycemic_status: str | None = None,
) -> pd.DataFrame:
    rows = []
    for comparison, ref_grade, ref_burden, comp_grade, comp_burden in KEY_COMPARISONS:
        ref = float(
            predictions.loc[
                (predictions["steatosis_grade"] == ref_grade)
                & (predictions["non_glycemic_burden_group"] == ref_burden),
                "adjusted_probability",
            ].iloc[0]
        )
        comp = float(
            predictions.loc[
                (predictions["steatosis_grade"] == comp_grade)
                & (predictions["non_glycemic_burden_group"] == comp_burden),
                "adjusted_probability",
            ].iloc[0]
        )
        or_info = model_or_contrast(model, df, comparison, ref_grade, ref_burden, comp_grade, comp_burden)
        rows.append(
            {
                "outcome": outcome,
                "cohort": cohort_name,
                "glycemic_status": glycemic_status,
                "comparison": comparison,
                "ref_grade": ref_grade,
                "ref_burden": ref_burden,
                "comp_grade": comp_grade,
                "comp_burden": comp_burden,
                "ref_adjusted_probability": ref,
                "comp_adjusted_probability": comp,
                "ref_adjusted_pct": ref * 100,
                "comp_adjusted_pct": comp * 100,
                "absolute_risk_difference": comp - ref,
                "absolute_risk_difference_pp": (comp - ref) * 100,
                "risk_ratio": comp / ref if ref > 0 else np.nan,
                **or_info,
            }
        )
    return pd.DataFrame(rows)


def run_overall_outcome_analysis(df: pd.DataFrame, outcome: str, cohort_name: str) -> OutcomeAnalysis:
    full_model = fit_glm_binomial(df, outcome, interaction=True)
    reduced_model = fit_glm_binomial(df, outcome, interaction=False)
    grid_summary = build_grid_summary(df, outcome, cohort_name)
    marginal_predictions = build_marginal_predictions(df, full_model, outcome, cohort_name)
    key_contrasts = build_key_contrasts(marginal_predictions, full_model, df, outcome, cohort_name)
    tests = interaction_test(full_model, reduced_model, outcome, cohort_name)
    return OutcomeAnalysis(
        outcome=outcome,
        cohort_name=cohort_name,
        model=full_model,
        reduced_model=reduced_model,
        grid_summary=grid_summary,
        marginal_predictions=marginal_predictions,
        key_contrasts=key_contrasts,
        interaction_test=tests,
    )


def run_proteinuria_glycemic_strata(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_grids = []
    all_predictions = []
    all_contrasts = []
    all_tests = []
    for status in GLYCEMIC_ORDER:
        sub = df.loc[df["glycemic_status"] == status].copy()
        analysis = run_overall_outcome_analysis(sub, outcome="proteinuria", cohort_name="full_analytic_cohort")
        grd = analysis.grid_summary.copy()
        grd["glycemic_status"] = status
        grd["glycemic_label"] = GLYCEMIC_LABELS[status]
        grd["stratum_n"] = len(sub)
        all_grids.append(grd)

        pred = analysis.marginal_predictions.copy()
        pred["glycemic_status"] = status
        pred["glycemic_label"] = GLYCEMIC_LABELS[status]
        pred["stratum_n"] = len(sub)
        all_predictions.append(pred)

        con = analysis.key_contrasts.copy()
        con["glycemic_status"] = status
        con["glycemic_label"] = GLYCEMIC_LABELS[status]
        con["stratum_n"] = len(sub)
        all_contrasts.append(con)

        tst = analysis.interaction_test.copy()
        tst["glycemic_status"] = status
        tst["glycemic_label"] = GLYCEMIC_LABELS[status]
        tst["stratum_n"] = len(sub)
        all_tests.append(tst)
    return (
        pd.concat(all_grids, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_contrasts, ignore_index=True),
        pd.concat(all_tests, ignore_index=True),
    )


def run_proteinuria_three_way_interaction(df: pd.DataFrame) -> pd.DataFrame:
    full_formula = (
        "proteinuria ~ C(steatosis_grade, Treatment(reference=0))"
        " * C(non_glycemic_burden_group, Treatment(reference='low'))"
        " * C(glycemic_status, Treatment(reference='normoglycemia'))"
        " + age + sex_binary"
    )
    reduced_formula = (
        "proteinuria ~ (C(steatosis_grade, Treatment(reference=0))"
        " + C(non_glycemic_burden_group, Treatment(reference='low'))"
        " + C(glycemic_status, Treatment(reference='normoglycemia'))) ** 2"
        " + age + sex_binary"
    )
    full_model = smf.glm(formula=full_formula, data=df, family=sm.families.Binomial()).fit()
    reduced_model = smf.glm(formula=reduced_formula, data=df, family=sm.families.Binomial()).fit()
    lr = 2 * (full_model.llf - reduced_model.llf)
    df_diff = int(full_model.df_model - reduced_model.df_model)
    return pd.DataFrame(
        [
            {
                "outcome": "proteinuria",
                "test": "three_way_interaction",
                "statistic": float(lr),
                "df_diff": df_diff,
                "p_value": float(chi2.sf(lr, df_diff)),
                "n": int(full_model.nobs),
            }
        ]
    )


def collapsed_cardio_kidney_glycemic_summary(echo_subset: pd.DataFrame) -> pd.DataFrame:
    return (
        echo_subset.groupby(
            ["glycemic_status_collapsed", "steatosis_grade", "non_glycemic_burden_group"],
            observed=False,
        )
        .agg(
            n=("participant_id", "size"),
            events=("cardio_kidney_coabnormality", "sum"),
            crude_rate=("cardio_kidney_coabnormality", "mean"),
        )
        .reset_index()
        .assign(
            crude_rate_pct=lambda d: d["crude_rate"] * 100,
            glycemic_label=lambda d: d["glycemic_status_collapsed"].astype(str).map(GLYCEMIC_COLLAPSED_LABELS),
            steatosis_label=lambda d: d["steatosis_grade"].astype(int).map(STEATOSIS_LABELS),
            burden_label=lambda d: d["non_glycemic_burden_group"].astype(str).map(BURDEN_LABELS),
        )
    )


def cardio_kidney_composition_summary(echo_subset: pd.DataFrame) -> pd.DataFrame:
    summary = (
        echo_subset.groupby(
            ["steatosis_grade", "non_glycemic_burden_group", "cardio_kidney_composition"],
            observed=False,
        )
        .agg(n=("participant_id", "size"))
        .reset_index()
    )
    totals = (
        summary.groupby(["steatosis_grade", "non_glycemic_burden_group"], observed=False)["n"]
        .sum()
        .reset_index(name="cell_total")
    )
    summary = summary.merge(totals, on=["steatosis_grade", "non_glycemic_burden_group"], how="left")
    summary["composition_pct"] = summary["n"] / summary["cell_total"] * 100
    summary["phenotype_id"] = (
        summary["steatosis_grade"].astype(int).astype(str)
        + "_"
        + summary["non_glycemic_burden_group"].astype(str)
    )
    summary["steatosis_label"] = summary["steatosis_grade"].astype(int).map(STEATOSIS_LABELS)
    summary["burden_label"] = summary["non_glycemic_burden_group"].astype(str).map(BURDEN_LABELS)
    summary["composition_label"] = summary["cardio_kidney_composition"].map(
        {
            "neither": "Neither abnormality",
            "kidney_only": "Kidney only",
            "cardiac_only": "Cardiac only",
            "both": "Both abnormalities",
        }
    )
    summary["phenotype_id"] = pd.Categorical(summary["phenotype_id"], categories=PHENOTYPE_ORDER, ordered=True)
    summary["cardio_kidney_composition"] = pd.Categorical(
        summary["cardio_kidney_composition"],
        categories=["neither", "kidney_only", "cardiac_only", "both"],
        ordered=True,
    )
    return summary.sort_values(["phenotype_id", "cardio_kidney_composition"]).reset_index(drop=True)


def sparse_cells(grid_summary: pd.DataFrame, event_min: int = 5, n_min: int = 20) -> pd.DataFrame:
    out = grid_summary.copy()
    out["sparse_n"] = out["n"] < n_min
    out["sparse_events"] = out["events"] < event_min
    out["is_sparse"] = out["sparse_n"] | out["sparse_events"]
    out["sparse_reason"] = np.where(
        out["sparse_n"] & out["sparse_events"],
        "n_lt_20_and_events_lt_5",
        np.where(out["sparse_n"], "n_lt_20", np.where(out["sparse_events"], "events_lt_5", "")),
    )
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def standardized_linear_prediction(model, df: pd.DataFrame, steatosis_grade: int, burden: str) -> float:
    cf = df.copy()
    cf["steatosis_grade"] = pd.Categorical([steatosis_grade] * len(cf), categories=STEATOSIS_ORDER, ordered=True)
    cf["non_glycemic_burden_group"] = pd.Categorical(
        [burden] * len(cf),
        categories=BURDEN_ORDER,
        ordered=True,
    )
    return float(model.predict(cf).mean())


def run_ir_characterization(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interaction_rows = []
    marginal_rows = []
    by_grid_rows = []

    for outcome, label in IR_OUTCOMES:
        full = smf.ols(
            f"{outcome} ~ C(steatosis_grade, Treatment(reference=0)) * C(non_glycemic_burden_group, Treatment(reference='low')) + age + sex_binary",
            data=df,
        ).fit()
        reduced = smf.ols(
            f"{outcome} ~ C(steatosis_grade, Treatment(reference=0)) + C(non_glycemic_burden_group, Treatment(reference='low')) + age + sex_binary",
            data=df,
        ).fit()

        lr = 2.0 * (full.llf - reduced.llf)
        df_diff = int(full.df_model - reduced.df_model)
        interaction_rows.append(
            {
                "outcome": outcome,
                "outcome_label": label,
                "n": int(len(df)),
                "interaction_statistic": float(lr),
                "df_diff": df_diff,
                "p_value": float(chi2.sf(lr, df_diff)),
            }
        )

        crude_grid = (
            df.groupby(["steatosis_grade", "non_glycemic_burden_group"], observed=False)
            .agg(
                n=("participant_id", "size"),
                mean_value=(outcome, "mean"),
                sd_value=(outcome, "std"),
            )
            .reset_index()
        )
        crude_grid["outcome"] = outcome
        crude_grid["outcome_label"] = label
        crude_grid["steatosis_label"] = crude_grid["steatosis_grade"].astype(int).map(STEATOSIS_LABELS)
        crude_grid["burden_label"] = crude_grid["non_glycemic_burden_group"].astype(str).map(BURDEN_LABELS)
        by_grid_rows.append(crude_grid)

        for steatosis_grade in STEATOSIS_ORDER:
            for burden in BURDEN_ORDER:
                marginal_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": label,
                        "steatosis_grade": steatosis_grade,
                        "non_glycemic_burden_group": burden,
                        "steatosis_label": STEATOSIS_LABELS[steatosis_grade],
                        "burden_label": BURDEN_LABELS[burden],
                        "adjusted_value": standardized_linear_prediction(full, df, steatosis_grade, burden),
                    }
                )

    return (
        pd.concat(by_grid_rows, ignore_index=True),
        pd.DataFrame(marginal_rows),
        pd.DataFrame(interaction_rows),
    )
