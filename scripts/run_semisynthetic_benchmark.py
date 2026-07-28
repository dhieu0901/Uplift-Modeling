# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.data.semisynthetic import generate_semisynthetic_uplift
from src.evaluation.ground_truth import (
    ground_truth_cate_metrics,
    ground_truth_policy_table,
)
from src.experiments.honest_uplift import run_honest_uplift_experiment
from src.models.registry import select_model_factories
from src.reporting import dataframe_to_markdown


DEFAULT_MODELS = (
    "response_model,s_learner,t_learner,x_learner,cvt,"
    "transformed_outcome,r_learner,dr_learner"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark uplift models on real Criteo covariates with known "
            "semi-synthetic response surfaces and CATE."
        )
    )
    parser.add_argument(
        "--sample-path",
        default="data/processed/criteo_sample_500k.parquet",
    )
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--control-rate", type=float, default=0.05)
    parser.add_argument("--average-effect", type=float, default=0.015)
    parser.add_argument("--heterogeneity-scale", type=float, default=0.035)
    parser.add_argument("--treatment-propensity", type=float, default=0.50)
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary-budget", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report-path",
        default="outputs/semisynthetic_benchmark.md",
    )
    parser.add_argument(
        "--metrics-path",
        default="outputs/tables/semisynthetic_cate_metrics.csv",
    )
    parser.add_argument(
        "--selection-path",
        default="outputs/tables/semisynthetic_selection.csv",
    )
    parser.add_argument(
        "--policy-path",
        default="outputs/tables/semisynthetic_policy_truth.csv",
    )
    parser.add_argument(
        "--figure-path",
        default="outputs/figures/semisynthetic_policy_truth.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = _parse_strings(args.models)
    budgets = _parse_floats(args.budgets)
    source = load_criteo(ROOT / args.sample_path, outcome="visit")
    source = subsample_criteo(source, args.max_rows, args.random_state)
    synthetic = generate_semisynthetic_uplift(
        source.X,
        control_rate=args.control_rate,
        average_effect=args.average_effect,
        heterogeneity_scale=args.heterogeneity_scale,
        treatment_propensity=args.treatment_propensity,
        random_state=args.random_state,
    )
    factories = select_model_factories(
        model_names,
        crossfit_folds=args.crossfit_folds,
    )
    result = run_honest_uplift_experiment(
        synthetic.dataset,
        model_factories=factories,
        budgets=budgets,
        primary_budget=args.primary_budget,
        random_state=args.random_state,
        selection_folds=args.selection_folds,
        evaluate_all_test=True,
        progress=print,
    )

    test_truth = synthetic.true_cate[result.splits.test.indices]
    evaluation_scores = {**result.test_scores, "oracle": test_truth}
    cate_metrics = ground_truth_cate_metrics(test_truth, evaluation_scores)
    true_policy_values = ground_truth_policy_table(
        test_truth,
        evaluation_scores,
        fractions=budgets,
    )
    selection_table = build_selection_table(result, args.primary_budget)

    metrics_path = ROOT / args.metrics_path
    selection_path = ROOT / args.selection_path
    policy_path = ROOT / args.policy_path
    figure_path = ROOT / args.figure_path
    report_path = ROOT / args.report_path
    for path in (
        metrics_path,
        selection_path,
        policy_path,
        figure_path,
        report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    cate_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    selection_table.to_csv(
        selection_path,
        index=False,
        encoding="utf-8-sig",
    )
    true_policy_values.to_csv(policy_path, index=False, encoding="utf-8-sig")
    plot_true_policy_value(true_policy_values, figure_path)
    report_path.write_text(
        build_report(
            args,
            result,
            synthetic,
            cate_metrics,
            selection_table,
            true_policy_values,
            metrics_path,
            selection_path,
            policy_path,
            figure_path,
        ),
        encoding="utf-8",
    )

    print(f"Out-of-sample-selected champion: {result.champion}")
    print(f"Semi-synthetic report: {report_path}")
    print(cate_metrics.to_string(index=False))


def _parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("models must not be empty.")
    return list(dict.fromkeys(values))


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError("budgets must be comma-separated numbers.") from exc
    if not values:
        raise ValueError("budgets must not be empty.")
    return values


def build_selection_table(result, primary_budget: float):
    primary_budget_pct = 100.0 * primary_budget
    selection = result.validation_policy_values[
        np.isclose(
            result.validation_policy_values["budget_pct"],
            primary_budget_pct,
        )
    ].merge(result.validation_metrics, on="policy", how="left")
    contrasts = result.validation_contrasts[
        np.isclose(
            result.validation_contrasts["budget_pct"],
            primary_budget_pct,
        )
    ][["policy", "difference", "ci_lower", "ci_upper"]].rename(
        columns={
            "difference": "difference_vs_response",
            "ci_lower": "difference_ci_lower",
            "ci_upper": "difference_ci_upper",
        }
    )
    return (
        selection.merge(contrasts, on="policy", how="left")
        .sort_values(
            "difference_ci_lower",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def plot_true_policy_value(table, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for policy, group in table.groupby("policy", sort=False):
        group = group.sort_values("budget_pct")
        axis.plot(
            group["budget_pct"],
            group["true_incremental_outcome"],
            marker="o",
            linewidth=2.5 if policy == "oracle" else 1.6,
            linestyle="--" if policy == "oracle" else "-",
            label=policy,
        )
    axis.set_title("Semi-synthetic exact policy value")
    axis.set_xlabel("Targeting budget (%)")
    axis.set_ylabel("True incremental outcomes")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_report(
    args,
    result,
    synthetic,
    cate_metrics,
    selection_table,
    true_policy_values,
    metrics_path: Path,
    selection_path: Path,
    policy_path: Path,
    figure_path: Path,
) -> str:
    primary_budget_pct = 100.0 * args.primary_budget
    primary_truth = true_policy_values[
        np.isclose(true_policy_values["budget_pct"], primary_budget_pct)
    ].sort_values("policy_regret")
    observed_primary = result.test_policy_values[
        np.isclose(result.test_policy_values["budget_pct"], primary_budget_pct)
    ].sort_values("incremental_outcome", ascending=False)

    return f"""# Semi-Synthetic Uplift Benchmark with Known CATE

## Data-Generating Process

- Covariates: `{args.sample_path}` ({len(synthetic.dataset.X):,} rows).
- Target control outcome rate: `{args.control_rate:.4f}`.
- Realized mean control response surface: `{synthetic.mu0.mean():.6f}`.
- Realized average CATE: `{synthetic.true_cate.mean():.6f}`.
- CATE standard deviation: `{synthetic.true_cate.std():.6f}`.
- Treatment propensity: `{args.treatment_propensity:.4f}`.
- The response surfaces contain nonlinear terms and feature interactions.
- Treatment is randomized and both potential-outcome probabilities are known.

## Honest Selection

The model is selected only from out-of-sample development paired AIPW lower confidence bounds
against response targeting at the pre-specified `{primary_budget_pct:.2f}%`
budget. The selected model is
**{result.champion}**. All candidates are refit on development data and evaluated
on test only to compare them against known ground truth; this does not change
the development-selected champion.

{dataframe_to_markdown(selection_table)}

## CATE Recovery

{dataframe_to_markdown(cate_metrics)}

PEHE, CATE MAE, and bias evaluate score magnitude. Pearson and Spearman
correlations evaluate linear and rank recovery. The response model is included
as an operational ranking baseline, not as a calibrated CATE estimator.

## Exact Policy Value at the Primary Budget

{dataframe_to_markdown(primary_truth)}

`policy_regret` is the exact difference from targeting the users with the
largest true CATE. `oracle_value_fraction` measures how much of the attainable
oracle gain each ranking captures.

## Observed AIPW Estimate at the Same Budget

{dataframe_to_markdown(observed_primary)}

This comparison checks whether the observed-data estimator and its uncertainty
lead to decisions that agree with the known response surfaces.

![Exact policy value](figures/{figure_path.name})

## Reproducible Outputs

- CATE metrics: `{metrics_path.relative_to(ROOT).as_posix()}`
- Development selection: `{selection_path.relative_to(ROOT).as_posix()}`
- Exact policy values: `{policy_path.relative_to(ROOT).as_posix()}`
"""


if __name__ == "__main__":
    main()
