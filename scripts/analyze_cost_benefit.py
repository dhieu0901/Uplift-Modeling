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

from src.evaluation.policy_value import (
    monetize_policy_table,
    select_best_campaign,
    uplift_score_threshold,
)
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert incremental outcomes into campaign net value."
    )
    parser.add_argument(
        "--input-path",
        default="reports/tables/visit_policy_summary.csv",
        help="CSV containing policy, budget_pct, n_targeted, and incremental_outcome.",
    )
    parser.add_argument("--outcome-name", default="visit")
    parser.add_argument("--outcome-value", type=float, default=100.0)
    parser.add_argument("--treatment-cost", type=float, default=5.0)
    parser.add_argument("--champion", default="transformed_outcome")
    parser.add_argument("--report-path", default="reports/cost_benefit_policy.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Policy summary not found: {input_path}")

    base_table = pd.read_csv(input_path)
    policy_table = monetize_policy_table(
        base_table,
        outcome_value=args.outcome_value,
        treatment_cost=args.treatment_cost,
    )
    best = select_best_campaign(policy_table)
    comparison = compare_with_response(policy_table, args.champion)
    threshold = uplift_score_threshold(args.outcome_value, args.treatment_cost)

    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            policy_table=policy_table,
            comparison=comparison,
            best=best,
            threshold=threshold,
        ),
        encoding="utf-8",
    )
    print(f"Report written to: {report_path}")
    print(
        f"Best option: {best['policy']} at a budget of "
        f"{best['budget_pct']:.1f}%, net value {best['net_value']:.2f}."
    )


def compare_with_response(
    policy_table: pd.DataFrame,
    champion: str,
) -> pd.DataFrame:
    required = {champion, "response_model"}
    missing = sorted(required - set(policy_table["policy"]))
    if missing:
        raise ValueError(f"Missing policies for comparison: {missing}")
    pivot = policy_table.pivot(
        index="budget_pct", columns="policy", values="net_value"
    ).reset_index()
    result = pivot[["budget_pct", champion, "response_model"]].rename(
        columns={champion: "champion_net_value"}
    )
    result["gain_vs_response"] = (
        result["champion_net_value"] - result["response_model"]
    )
    return result


def build_report(
    args: argparse.Namespace,
    policy_table: pd.DataFrame,
    comparison: pd.DataFrame,
    best: pd.Series,
    threshold: float,
) -> str:
    efficiency_columns = [
        "policy",
        "budget_pct",
        "n_targeted",
        "incremental_outcome",
        "break_even_cost_per_target",
        "minimum_value_to_cost_ratio",
    ]
    value_columns = [
        "policy",
        "budget_pct",
        "gross_value",
        "campaign_cost",
        "net_value",
        "net_value_per_1k_targeted",
        "profitable",
    ]
    best_description = (
        "do not run a campaign"
        if best["policy"] == "no_campaign"
        else (
            f"`{best['policy']}` at a {best['budget_pct']:.1f}% budget "
            f"with a net value of `{best['net_value']:.2f}`"
        )
    )
    value_cost_ratio = args.outcome_value / args.treatment_cost if args.treatment_cost else float("inf")

    return f"""# Targeting Policy Cost-Benefit Analysis

## Inputs

- Source: `{args.input_path}`.
- Incremental outcome is averaged across three seeds from the Criteo `visit` experiment.
- Illustrative value of one incremental {args.outcome_name}: `{args.outcome_value:.2f}` currency units.
- Cost per targeted user: `{args.treatment_cost:.2f}` currency units.
- Scenario value-to-cost ratio: `{value_cost_ratio:.2f}`.

The monetary figures are an **illustrative scenario**, not validated business
assumptions. Rerun the script with `--outcome-value` and `--treatment-cost`.

## Break-Even Thresholds

{dataframe_to_markdown(policy_table[efficiency_columns])}

`minimum_value_to_cost_ratio` directly answers how many times greater the value
of one incremental outcome must be than the cost of one targeting action for the
policy to break even. With the current parameters, the individual uplift threshold
is `{threshold:.6f}`: target only when estimated uplift exceeds this threshold and
the score is sufficiently well calibrated.

## Net Value by Budget

{dataframe_to_markdown(policy_table[value_columns])}

The option that maximizes the point estimate is **{best_description}**. The
selection function always includes `no_campaign`, with a net value of zero,
among the options, so it will not recommend a campaign with a negative point estimate.

## Comparison with Response Targeting

{dataframe_to_markdown(comparison)}

At the same budget, campaign costs are identical; therefore, `gain_vs_response`
comes from the additional incremental {args.outcome_name} generated by uplift ranking.

## Recommendations

- Replace the illustrative value and cost with contribution margin and all actual costs.
- Prioritize budgets with positive net value and gains stable across seeds/bootstrap samples.
- Do not apply a score threshold directly until uplift-score calibration has been validated.
- Validate the final policy with a randomized holdout or A/B test before rollout.
"""


if __name__ == "__main__":
    main()
