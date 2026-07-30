# ruff: noqa: E402

from __future__ import annotations

import argparse
from math import ceil
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

from src.evaluation.experiment_design import (
    buffered_sample_size,
    policy_outcome_rate,
    two_proportion_sample_size,
)
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design a randomized online experiment for two targeting policies."
    )
    parser.add_argument(
        "--input-path",
        default="outputs/tables/audit_visit_test.csv",
    )
    parser.add_argument("--policy-a", default="s_learner")
    parser.add_argument("--policy-b", default="response_model")
    parser.add_argument("--budget-pct", type=float, default=5.0)
    parser.add_argument("--no-campaign-rate", type=float, default=0.038201)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--planning-effect-fraction", type=float, default=0.75)
    parser.add_argument("--buffer-fraction", type=float, default=0.15)
    parser.add_argument(
        "--report-path", default="outputs/online_experiment_design.md"
    )
    parser.add_argument(
        "--arms-path", default="outputs/tables/online_experiment_arms.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.planning_effect_fraction <= 1.0:
        raise ValueError("planning_effect_fraction must be in the interval (0, 1].")

    input_path = ROOT / args.input_path
    policy_summary = pd.read_csv(input_path)
    policy_a = get_policy_row(policy_summary, args.policy_a, args.budget_pct)
    policy_b = get_policy_row(policy_summary, args.policy_b, args.budget_pct)
    population_a = infer_population_size(policy_a)
    population_b = infer_population_size(policy_b)
    if population_a != population_b:
        raise ValueError("Both policies must be estimated on the same population size.")
    population_size = population_a

    rate_a = policy_outcome_rate(
        args.no_campaign_rate,
        policy_a["incremental_outcome"],
        population_size,
    )
    rate_b = policy_outcome_rate(
        args.no_campaign_rate,
        policy_b["incremental_outcome"],
        population_size,
    )
    offline_difference = rate_a - rate_b
    planned_difference = offline_difference * args.planning_effect_fraction
    planned_rate_a = rate_b + planned_difference

    primary_unbuffered = two_proportion_sample_size(
        planned_rate_a,
        rate_b,
        alpha=args.alpha,
        power=args.power,
    )
    primary_sample = buffered_sample_size(
        primary_unbuffered, args.buffer_fraction
    )

    response_effect_for_planning = (
        rate_b - args.no_campaign_rate
    ) * args.planning_effect_fraction
    planned_response_rate = args.no_campaign_rate + response_effect_for_planning
    holdout_unbuffered = two_proportion_sample_size(
        planned_response_rate,
        args.no_campaign_rate,
        alpha=args.alpha,
        power=args.power,
    )
    holdout_sample = buffered_sample_size(
        holdout_unbuffered, args.buffer_fraction
    )
    arms = pd.DataFrame(
        [
            {
                "arm": "A",
                "policy": args.policy_a,
                "target_rate_pct": args.budget_pct,
                "sample_size": primary_sample,
                "expected_visit_rate_offline": rate_a,
            },
            {
                "arm": "B",
                "policy": args.policy_b,
                "target_rate_pct": args.budget_pct,
                "sample_size": primary_sample,
                "expected_visit_rate_offline": rate_b,
            },
            {
                "arm": "H",
                "policy": "no_campaign_holdout",
                "target_rate_pct": 0.0,
                "sample_size": holdout_sample,
                "expected_visit_rate_offline": args.no_campaign_rate,
            },
        ]
    )
    sensitivity = build_sensitivity_table(
        rate_a=rate_a,
        rate_b=rate_b,
        alpha=args.alpha,
        power=args.power,
        buffer_fraction=args.buffer_fraction,
    )

    arms_path = ROOT / args.arms_path
    arms_path.parent.mkdir(parents=True, exist_ok=True)
    arms.to_csv(arms_path, index=False, encoding="utf-8-sig")
    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            population_size=population_size,
            rate_a=rate_a,
            rate_b=rate_b,
            planned_difference=planned_difference,
            primary_unbuffered=primary_unbuffered,
            holdout_unbuffered=holdout_unbuffered,
            arms=arms,
            sensitivity=sensitivity,
        ),
        encoding="utf-8",
    )
    print(f"Design written to: {report_path}")
    print(arms.to_string(index=False))


def get_policy_row(
    policy_summary: pd.DataFrame,
    policy: str,
    budget_pct: float,
) -> pd.Series:
    selected = policy_summary[
        (policy_summary["policy"] == policy)
        & (policy_summary["budget_pct"] == budget_pct)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Exactly one row is required for policy={policy}, budget_pct={budget_pct}."
        )
    return selected.iloc[0]


def infer_population_size(policy_row: pd.Series) -> int:
    budget_fraction = float(policy_row["budget_pct"]) / 100.0
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError("budget_pct must be in the interval (0, 100].")
    return int(round(float(policy_row["n_targeted"]) / budget_fraction))


def build_sensitivity_table(
    rate_a: float,
    rate_b: float,
    alpha: float,
    power: float,
    buffer_fraction: float,
) -> pd.DataFrame:
    rows = []
    offline_difference = rate_a - rate_b
    for fraction in [1.0, 0.75, 0.50]:
        planned_rate_a = rate_b + offline_difference * fraction
        unbuffered = two_proportion_sample_size(
            planned_rate_a,
            rate_b,
            alpha=alpha,
            power=power,
        )
        rows.append(
            {
                "effect_retained_pct": 100.0 * fraction,
                "planned_absolute_difference": offline_difference * fraction,
                "sample_per_policy_arm": unbuffered,
                "sample_with_buffer": buffered_sample_size(
                    unbuffered, buffer_fraction
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    args: argparse.Namespace,
    population_size: int,
    rate_a: float,
    rate_b: float,
    planned_difference: float,
    primary_unbuffered: int,
    holdout_unbuffered: int,
    arms: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    total_sample = int(arms["sample_size"].sum())
    primary_sample = int(arms.loc[arms["arm"] == "A", "sample_size"].iloc[0])
    holdout_sample = int(arms.loc[arms["arm"] == "H", "sample_size"].iloc[0])
    approximate_targets_a = ceil(primary_sample * args.budget_pct / 100.0)
    approximate_targets_b = ceil(primary_sample * args.budget_pct / 100.0)
    arms_display = arms.rename(
        columns={
            "arm": "Arm",
            "policy": "Policy",
            "target_rate_pct": "Target rate",
            "sample_size": "Users",
            "expected_visit_rate_offline": "Offline visit rate",
        }
    ).copy()
    arms_display["Target rate"] = arms_display["Target rate"].map(
        lambda value: f"{value:.0f}%"
    )
    arms_display["Users"] = arms_display["Users"].map(
        lambda value: f"{value:,}"
    )
    arms_display["Offline visit rate"] = arms_display[
        "Offline visit rate"
    ].map(lambda value: f"{value:.6f}")
    sensitivity_display = sensitivity.rename(
        columns={
            "effect_retained_pct": "Effect retained",
            "planned_absolute_difference": "Difference",
            "sample_per_policy_arm": "Users per arm",
            "sample_with_buffer": "Users per arm with buffer",
        }
    ).copy()
    sensitivity_display["Effect retained"] = sensitivity_display[
        "Effect retained"
    ].map(lambda value: f"{value:.0f}%")
    sensitivity_display["Difference"] = sensitivity_display["Difference"].map(
        lambda value: f"{value:.6f}"
    )
    for column in ("Users per arm", "Users per arm with buffer"):
        sensitivity_display[column] = sensitivity_display[column].map(
            lambda value: f"{value:,}"
        )

    return f"""# Randomized Experiment Design for the Targeting Policy

## Objective

Directly compare `{args.policy_a}` with `{args.policy_b}` at the same
`{args.budget_pct:.1f}%` budget. The primary estimand is the
**intention-to-treat difference** in visit rate across all users assigned to
each policy arm.

## Sample-Size Assumptions

- Offline source: `{args.input_path}` on a test population of {population_size:,} users.
- No-campaign visit rate: `{args.no_campaign_rate:.6f}`.
- Implied visit rates for arms A/B: `{rate_a:.6f}` / `{rate_b:.6f}`.
- Offline A-B difference: `{rate_a - rate_b:.6f}`.
- Planning effect retains `{args.planning_effect_fraction:.0%}` of the offline
  difference: `{planned_difference:.6f}`.
- Two-sided test, alpha `{args.alpha:.2f}`, power `{args.power:.0%}`.
- Buffer for attrition/logging loss: `{args.buffer_fraction:.0%}`.

The unbuffered sample size for each policy arm is {primary_unbuffered:,}. The
holdout size is calculated conservatively from the response-policy versus
no-campaign comparison: {holdout_unbuffered:,}.

## Proposed Allocation

{dataframe_to_markdown(arms_display)}

Proposed total cohort: **{total_sample:,} users**. Each policy arm has
{primary_sample:,} users and is expected to target approximately
{approximate_targets_a:,} / {approximate_targets_b:,} users. The holdout contains
{holdout_sample:,} users who receive no campaign during the measurement window.

## Sensitivity Analysis by Online Effect Size

{dataframe_to_markdown(sensitivity_display)}

The required sample size grows rapidly when the online effect is smaller than
the offline estimate. The default design assumes 75% effect retention; if traffic
allows, the 50% scenario is safer.

## Randomization Procedure

1. Finalize eligibility and the observation window; exclude users in
   conflicting campaigns.
2. Randomize deterministically by user ID into A, B, and H **before applying
   ranking policies**.
3. Score A and B independently, then target exactly the top
   {args.budget_pct:.1f}% within each arm.
4. Use the same channel, creative, send time, frequency cap, and treatment
   cost for A/B.
5. Keep assignment fixed; log assignment, score, treatment delivered, and
   outcome.

Do not compare only the two targeted subsets, because each policy selects a
different population and that comparison breaks randomization.

## Analysis Plan

- Primary: A-B visit-rate difference with a 95% confidence interval, analyzed
  by ITT.
- Secondary: incremental visits versus H, conversion rate, and net value for
  the full arm.
- Guardrails: unsubscribe/opt-out, complaints, contact frequency, and campaign
  cost.
- Report absolute difference, relative lift, and confidence interval, not only
  the p-value.
- Lock sample size and the measurement window before launch; do not stop early
  based on p-values.
- Check sample-ratio mismatch, contamination, and missing outcomes before
  interpreting lift.

## Decision Criteria

Roll out `{args.policy_a}` only when A beats B on the primary KPI, net value is
positive, guardrails do not deteriorate, and the result is not driven by a
small subgroup. If A-B is inconclusive but both beat H, retain the current
policy and collect more data instead of declaring the policies equivalent.

## Reproducible Output

- Allocation table: `outputs/tables/online_experiment_arms.csv`
"""


if __name__ == "__main__":
    main()
