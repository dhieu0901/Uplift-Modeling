from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.evaluation.online_experiment import (
    OnlineExperimentAnalysis,
    analyze_online_experiment,
)
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SRM, ITT lift, and net value for an online experiment."
    )
    parser.add_argument("--input-path", default="data/online_experiment_results.csv")
    parser.add_argument(
        "--arms-path", default="reports/tables/online_experiment_arms.csv"
    )
    parser.add_argument("--arm-a", default="A")
    parser.add_argument("--arm-b", default="B")
    parser.add_argument("--holdout-arm", default="H")
    parser.add_argument("--outcome-value", type=float, default=100.0)
    parser.add_argument("--treatment-cost", type=float, default=5.0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--srm-alpha", type=float, default=0.01)
    parser.add_argument("--max-missing-rate-gap", type=float, default=0.005)
    parser.add_argument(
        "--report-path", default="reports/online_experiment_results.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input_path
    arms_path = ROOT / args.arms_path
    if not input_path.exists():
        raise FileNotFoundError(f"Experiment results not found: {input_path}")
    if not arms_path.exists():
        raise FileNotFoundError(f"Experiment design not found: {arms_path}")

    results = pd.read_csv(input_path)
    design = pd.read_csv(arms_path)
    is_synthetic = bool(
        "is_synthetic" in results
        and results["is_synthetic"].fillna(False).astype(bool).all()
    )
    analysis = analyze_online_experiment(
        results,
        design,
        arm_a=args.arm_a,
        arm_b=args.arm_b,
        holdout_arm=args.holdout_arm,
        outcome_value=args.outcome_value,
        treatment_cost=args.treatment_cost,
        confidence_level=args.confidence_level,
        srm_alpha=args.srm_alpha,
        max_missing_rate_gap=args.max_missing_rate_gap,
    )
    decision = make_decision(
        analysis,
        confidence_alpha=1.0 - args.confidence_level,
        is_synthetic=is_synthetic,
    )

    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            analysis=analysis,
            is_synthetic=is_synthetic,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(f"Report written to: {report_path}")
    print(f"Conclusion: {decision}")


def make_decision(
    analysis: OnlineExperimentAnalysis,
    confidence_alpha: float,
    is_synthetic: bool,
) -> str:
    if is_synthetic:
        return "Dry run successful; do not use synthetic data to make a rollout decision."
    if not analysis.quality_checks["passed"].all():
        return "Stop interpreting lift and investigate the failed data-quality checks."

    primary = analysis.comparisons[
        analysis.comparisons["comparison"] == "primary_A_vs_B"
    ].iloc[0]
    if (
        primary["absolute_difference"] > 0
        and primary["p_value"] < confidence_alpha
        and primary["net_value_difference_ci_lower"] > 0
    ):
        return "Arm A is a rollout candidate, pending validation of all guardrails."
    return "The primary result is insufficient to change policy; retain the status quo or collect more data."


def build_report(
    args: argparse.Namespace,
    analysis: OnlineExperimentAnalysis,
    is_synthetic: bool,
    decision: str,
) -> str:
    title = (
        "Online-Experiment Analysis Dry Run"
        if is_synthetic
        else "Online-Experiment Results"
    )
    warning = (
        "**Warning: this synthetic data is only for pipeline testing and is not business evidence.**"
        if is_synthetic
        else "The data is treated as real results; validate the data contract and guardrails before rollout."
    )
    primary = analysis.comparisons[
        analysis.comparisons["comparison"] == "primary_A_vs_B"
    ].iloc[0]

    return f"""# {title}

{warning}

## Setup

- Results: `{args.input_path}`.
- Design: `{args.arms_path}`.
- Primary comparison: `{args.arm_a}` versus `{args.arm_b}` under ITT.
- Holdout: `{args.holdout_arm}`.
- Confidence level: `{args.confidence_level:.0%}`; SRM alpha `{args.srm_alpha:.3f}`.
- Outcome value / treatment cost: `{args.outcome_value:.2f}` / `{args.treatment_cost:.2f}`.

## Data-quality checks

{dataframe_to_markdown(analysis.quality_checks)}

SRM chi-square statistic: `{analysis.srm_statistic:.6f}`; p-value:
`{analysis.srm_p_value:.6f}`. If the SRM or contamination check fails, do not
interpret the treatment-effect p-value until the cause is identified.

### Arm Allocation

{dataframe_to_markdown(analysis.srm_table)}

## Results by Arm

{dataframe_to_markdown(analysis.arm_summary)}

The primary outcome always uses all users according to randomized assignment.
`target_rate` and `delivery_rate` are diagnostics only and are not used to filter
the analysis population.

## ITT comparisons

{dataframe_to_markdown(analysis.comparisons)}

The primary A-B absolute difference is `{primary['absolute_difference']:.6f}`, with
a {args.confidence_level:.0%} CI from `{primary['ci_lower']:.6f}` to
`{primary['ci_upper']:.6f}`, p-value `{primary['p_value']:.6f}`.

## Policy Value versus Holdout

{dataframe_to_markdown(analysis.policy_value)}

Net value uses the ITT outcome rate relative to the holdout and actual campaign
cost when the input includes `total_campaign_cost`. Unit economics must be locked
before results are reviewed to avoid selecting favorable assumptions afterward.

## Automated Conclusion

**{decision}**

The automated conclusion considers only the primary KPI, data quality, and net
value. The actual decision must also check guardrails, logging audits, subgroup
consistency, and novelty effects.

## Data contract

Each arm has exactly one row with the columns `arm`, `assigned_n`,
`outcome_observed_n`, `outcomes`, `targeted_n`, and `treatment_received_n`.
`total_campaign_cost` is optional; when absent, cost is inferred from `targeted_n`.
"""


if __name__ == "__main__":
    main()
