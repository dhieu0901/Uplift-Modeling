# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import count_overlapping_rows, load_criteo
from src.experiments.honest_uplift import evaluate_locked_policies
from src.models.registry import select_model_factories
from src.reporting import dataframe_to_markdown, plot_policy_value_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an already-locked champion against the response baseline "
            "on a large confirmatory sample that shares no row with any sample "
            "used for training, selection, or earlier evaluation."
        )
    )
    parser.add_argument(
        "--fit-path",
        default="data/processed/criteo_audit_1m.parquet",
        help="Sample the locked policies are refit on.",
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/criteo_confirm_4m.parquet",
        help="Confirmatory sample, opened once.",
    )
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument(
        "--champion",
        default="s_learner",
        help="Learner locked by the earlier selection stage.",
    )
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary-budget", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=20260730)
    parser.add_argument(
        "--baseline-contrast-path",
        default="outputs/tables/audit_visit_contrasts.csv",
        help=(
            "Earlier locked-test contrasts, used only to quantify how much "
            "precision the larger sample buys."
        ),
    )
    parser.add_argument(
        "--report-path",
        default="outputs/confirmatory_visit_evaluation.md",
    )
    parser.add_argument(
        "--test-table-path",
        default="outputs/tables/confirmatory_visit_test.csv",
    )
    parser.add_argument(
        "--contrast-path",
        default="outputs/tables/confirmatory_visit_contrasts.csv",
    )
    parser.add_argument(
        "--figure-path",
        default="outputs/figures/confirmatory_visit_policy_value.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = _parse_floats(args.budgets)
    if args.primary_budget not in budgets:
        raise ValueError("primary-budget must be included in budgets.")

    fit_path = ROOT / args.fit_path
    test_path = ROOT / args.test_path
    print("Verifying that the confirmatory sample is disjoint from the fit sample...")
    overlap = count_overlapping_rows(test_path, fit_path)
    if overlap != 0:
        raise SystemExit(
            f"Refusing to evaluate: {overlap:,} confirmatory rows also appear in "
            f"{args.fit_path}. A confirmatory test on overlapping rows is not a "
            "confirmatory test."
        )
    print("Overlap with the fit sample: 0 rows.")

    fit_dataset = load_criteo(fit_path, outcome=args.outcome)
    test_dataset = load_criteo(test_path, outcome=args.outcome)
    factories = select_model_factories(
        [args.champion],
        crossfit_folds=args.crossfit_folds,
    )

    result = evaluate_locked_policies(
        fit_dataset,
        test_dataset,
        factories,
        budgets=budgets,
        confidence_level=args.confidence_level,
        random_state=args.random_state,
        progress=print,
    )

    primary_budget_pct = round(100.0 * args.primary_budget, 4)
    primary_contrast = result.contrasts[
        np.isclose(result.contrasts["budget_pct"], primary_budget_pct)
        & (result.contrasts["policy"] == args.champion)
    ].iloc[0]
    precision_table = build_precision_table(
        args,
        primary_contrast,
        n_test=result.n_test,
        confidence_level=args.confidence_level,
    )

    report_path = ROOT / args.report_path
    test_table_path = ROOT / args.test_table_path
    contrast_path = ROOT / args.contrast_path
    figure_path = ROOT / args.figure_path
    for path in (report_path, test_table_path, contrast_path, figure_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    result.policy_values.to_csv(test_table_path, index=False, encoding="utf-8-sig")
    result.contrasts.to_csv(contrast_path, index=False, encoding="utf-8-sig")
    plot_policy_value_curve(
        result.policy_values,
        figure_path,
        title=(
            f"Confirmatory AIPW incremental {args.outcome} "
            f"({result.n_test:,} untouched users)"
        ),
    )
    report_path.write_text(
        build_report(
            args,
            result,
            primary_contrast=primary_contrast,
            precision_table=precision_table,
            figure_path=figure_path,
            test_table_path=test_table_path,
            contrast_path=contrast_path,
        ),
        encoding="utf-8",
    )

    print(f"\nConfirmatory report: {report_path}")
    print(result.contrasts.to_string(index=False))


def build_precision_table(
    args: argparse.Namespace,
    primary_contrast: pd.Series,
    n_test: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Compare the confirmatory precision against the earlier locked test.

    Everything is reported per 100,000 evaluated users. Totals are not
    comparable across stages: the difference and its standard error both grow
    with the evaluation sample, so a raw standard-error ratio would make the
    larger, more precise sample look worse.
    """
    z_critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows = []
    baseline_path = ROOT / args.baseline_contrast_path
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        matched = baseline[
            np.isclose(baseline["budget_pct"], round(100.0 * args.primary_budget, 4))
            & (baseline["policy"] == args.champion)
        ]
        if not matched.empty:
            row = matched.iloc[0]
            rows.append(
                _precision_row(
                    "Earlier locked test",
                    row,
                    z_critical,
                    args.primary_budget,
                )
            )
    rows.append(
        _precision_row(
            "Confirmatory test",
            primary_contrast,
            z_critical,
            args.primary_budget,
        )
    )
    table = pd.DataFrame(rows)
    if len(table) == 2:
        table["standard_error_reduction"] = (
            table["standard_error_per_100k"].iloc[0]
            / table["standard_error_per_100k"]
        )
    return table


def _precision_row(
    label: str,
    row: pd.Series,
    z_critical: float,
    primary_budget: float,
) -> dict:
    standard_error = (float(row["ci_upper"]) - float(row["ci_lower"])) / (
        2.0 * z_critical
    )
    difference = float(row["difference"])
    n_evaluated = float(row["n_targeted"]) / float(primary_budget)
    per_100k = 100_000.0 / n_evaluated
    return {
        "stage": label,
        "n_evaluated": int(round(n_evaluated)),
        "difference_per_100k": difference * per_100k,
        "standard_error_per_100k": standard_error * per_100k,
        # The z-statistic is scale-free, so it is the one figure that compares
        # directly across stages.
        "z_statistic": difference / standard_error if standard_error > 0 else np.nan,
        "ci_lower_per_100k": float(row["ci_lower"]) * per_100k,
        "ci_upper_per_100k": float(row["ci_upper"]) * per_100k,
        "excludes_zero": bool(
            float(row["ci_lower"]) > 0.0 or float(row["ci_upper"]) < 0.0
        ),
    }


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


def build_report(
    args: argparse.Namespace,
    result,
    primary_contrast: pd.Series,
    precision_table: pd.DataFrame,
    figure_path: Path,
    test_table_path: Path,
    contrast_path: Path,
) -> str:
    if primary_contrast["ci_lower"] > 0:
        verdict = (
            "confirmed a positive advantage over response targeting at the "
            "pre-specified budget"
        )
    elif primary_contrast["ci_upper"] < 0:
        verdict = "showed a negative advantage relative to response targeting"
    else:
        verdict = "remained inconclusive relative to response targeting"

    return f"""# Confirmatory Locked Test: Criteo {args.outcome}

## Why This Stage Exists

The earlier locked test estimated a positive but inconclusive advantage for
`{args.champion}` at the `{100.0 * args.primary_budget:.2f}%` budget. That
interval was wide because the test partition held only a fifth of a
one-million-row audit, not because the effect was known to be zero. The source
experiment has 13,979,592 randomized rows, so the cheapest way to sharpen the
answer is to spend unused rows on evaluation rather than to re-argue the same
numbers.

Nothing is selected here. The champion, the operating budget, the confidence
level, and the estimator were all locked before this sample was drawn.

## Protocol

- Locked champion: `{args.champion}` (selected earlier, on development data only).
- Refit sample: `{args.fit_path}` ({result.n_fit:,} rows).
- Confirmatory sample: `{args.test_path}` ({result.n_test:,} rows), opened once.
- Verified overlap between the two samples: **0 rows**, by `row_id`.
- Reference policies: `response_model` (the incumbent) and `random_targeting`
  (the floor that shows how much of any gain is ranking skill).
- Treatment propensity used by AIPW: `{result.propensity:.6f}`.
- Primary budget `{100.0 * args.primary_budget:.2f}%`, confidence level
  `{100.0 * args.confidence_level:.1f}%`.

## Policy Value

{dataframe_to_markdown(result.policy_values)}

![Confirmatory policy value](figures/{figure_path.name})

## Paired Contrast Against Response Targeting

{dataframe_to_markdown(result.contrasts)}

At the pre-specified `{100.0 * args.primary_budget:.2f}%` budget the confirmatory
sample {verdict}. The estimated difference is
`{primary_contrast['difference']:.4f}` incremental {args.outcome} outcomes with a
`{100.0 * args.confidence_level:.1f}%` interval of
`[{primary_contrast['ci_lower']:.4f}, {primary_contrast['ci_upper']:.4f}]`.

## Precision Gained

{dataframe_to_markdown(precision_table)}

Both stages estimate the same locked policy, so the difference between them is
evaluation precision, not a different effect. Figures are normalized per 100,000
evaluated users because totals scale with the sample and are not comparable; the
z-statistic is scale-free and comparable as printed. Agreement between the two
per-100,000 point estimates is the check that the larger sample bought precision
rather than a new answer.

## Ranking Metrics

{dataframe_to_markdown(result.metrics)}

Relative AUUC scores the whole ranking. The decision is made on the
budget-specific paired contrast because the campaign can only treat a fixed
share of users.

## Runtime

{dataframe_to_markdown(result.timing_table)}

## Interpretation Boundaries

- The confirmatory sample is disjoint from every earlier sample but is drawn
  from the same source experiment and the same time period.
- The interval conditions on the fitted policies; it does not carry the
  uncertainty of having chosen `{args.champion}` in the first place. Selection
  stability is measured separately by the repeated-split experiment.
- AIPW treats the treatment propensity as known, as is standard for a
  randomized design; the propensity is estimated from the refit sample.
- Offline evidence bounds what a live campaign might do. It does not replace a
  randomized online test, and it carries no claim about revenue.

## Reproducible Outputs

- Policy values: `{test_table_path.relative_to(ROOT).as_posix()}`
- Paired contrasts: `{contrast_path.relative_to(ROOT).as_posix()}`
"""


if __name__ == "__main__":
    main()
