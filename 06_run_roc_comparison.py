from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rebuild_lib import (
    FIGURE_DIR,
    RESULT_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_source_cohort,
    prepare_echo_subset,
    prepare_full_analytic_cohort,
)


FIGURE_OUT = FIGURE_DIR / "figure6_roc_comparison.png"
PDF_OUT = FIGURE_DIR / "figure6_roc_comparison.pdf"
AUC_TABLE_OUT = TABLE_DIR / "supplementary_table_s3_auc_comparison.csv"

MODEL_SPECS = [
    ("steatosis_alone", "Steatosis alone", ["steatosis_grade", "age", "sex_binary"], "#577590"),
    (
        "burden_alone",
        "Metabolic burden",
        ["non_glycemic_burden_group", "age", "sex_binary"],
        "#6d597a",
    ),
    (
        "phenotype",
        "Phenotype",
        ["phenotype_id", "age", "sex_binary"],
        "#c1121f",
    ),
    (
        "phenotype_plus_glycemic",
        "Phenotype + glycemic status",
        ["phenotype_id", "glycemic_status", "age", "sex_binary"],
        "#1b4332",
    ),
]


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 20260420) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        samples.append(roc_auc_score(y_b, y_score[idx]))
    if not samples:
        return np.nan, np.nan
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def fit_and_score(df: pd.DataFrame, outcome: str, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    y = df[outcome].to_numpy(dtype=int)
    categorical = [col for col in features if X[col].dtype == "object" or str(X[col].dtype).startswith("category")]
    numeric = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return scores


def evaluate_outcome(df: pd.DataFrame, outcome: str, outcome_label: str) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    curve_rows = []
    y_true = df[outcome].to_numpy(dtype=int)

    for model_id, model_label, features, color in MODEL_SPECS:
        y_score = fit_and_score(df, outcome, features)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        ci_low, ci_high = bootstrap_auc_ci(y_true, np.asarray(y_score))

        rows.append(
            {
                "outcome": outcome,
                "outcome_label": outcome_label,
                "model_id": model_id,
                "model_label": model_label,
                "auc": float(auc_value),
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "n": int(len(df)),
                "events": int(df[outcome].sum()),
                "evaluation": "5-fold cross-validated",
            }
        )
        for x, y in zip(fpr, tpr):
            curve_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "model_id": model_id,
                    "model_label": model_label,
                    "fpr": float(x),
                    "tpr": float(y),
                    "auc": float(auc_value),
                    "color": color,
                }
            )

    return pd.DataFrame(rows), curve_rows


def main() -> None:
    ensure_dirs()
    source_df = load_source_cohort()
    full_cohort = prepare_full_analytic_cohort(source_df)
    echo_subset = prepare_echo_subset(full_cohort)

    proteinuria_auc, proteinuria_curves = evaluate_outcome(full_cohort, "proteinuria", "Proteinuria")
    ck_auc, ck_curves = evaluate_outcome(
        echo_subset, "cardio_kidney_coabnormality", "Cardio-kidney co-abnormality"
    )
    auc_df = pd.concat([proteinuria_auc, ck_auc], ignore_index=True)
    curve_df = pd.DataFrame(proteinuria_curves + ck_curves)

    auc_df.to_csv(AUC_TABLE_OUT, index=False)
    curve_df.to_csv(RESULT_DIR / "roc_curve_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
    outcome_order = ["Proteinuria", "Cardio-kidney co-abnormality"]
    title_letters = ["A", "B"]

    for ax, outcome_label, letter in zip(axes, outcome_order, title_letters):
        sub = curve_df.loc[curve_df["outcome_label"] == outcome_label].copy()
        auc_sub = auc_df.loc[auc_df["outcome_label"] == outcome_label].copy()
        for model_id, model_label, _, color in MODEL_SPECS:
            curve = sub.loc[sub["model_id"] == model_id].copy()
            auc_row = auc_sub.loc[auc_sub["model_id"] == model_id].iloc[0]
            legend_label = f"{model_label}: {auc_row['auc']:.3f} ({auc_row['auc_ci_low']:.3f}-{auc_row['auc_ci_high']:.3f})"
            ax.plot(curve["fpr"], curve["tpr"], color=color, linewidth=2.2, label=legend_label)

        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2)
        ax.set_title(f"{letter}. {outcome_label}", fontsize=12.5, pad=8)
        ax.set_xlabel("False positive rate")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8.9, loc="lower right")

    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
