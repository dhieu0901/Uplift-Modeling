# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.experiments.honest_uplift import (
    multiplicity_adjusted_bounds,
    run_honest_uplift_experiment,
)
from src.models.registry import (
    REFERENCE_POLICIES,
    rare_outcome_model_factories,
    select_model_factories,
)
from src.reporting import dataframe_to_markdown, plot_policy_value_curve

DEFAULT_MODELS = (
    "response_model,s_learner,t_learner,x_learner,cvt,"
    "transformed_outcome,r_learner,dr_learner"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an uplift learner from out-of-sample development "
            "predictions and evaluate the locked champion against response "
            "targeting on an untouched test set."
        )
    )
    parser.add_argument(
        "--sample-path",
        default="data/processed/criteo_sample_500k.parquet",
    )
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--selection-folds",
        type=int,
        default=1,
        help=(
            "Use out-of-fold policy selection on all development data when "
            "greater than 1; 1 uses the explicit validation holdout."
        ),
    )
    parser.add_argument(
        "--undersampling-factors",
        default="",
        help=(
            "Optional comma-separated factors for logistic T/CVT candidates; "
            "intended for rare conversion outcomes."
        ),
    )
    parser.add_argument(
        "--undersampling-families",
        default="t,cvt",
        help=(
            "Comma-separated rare-outcome families to include: t,cvt. "
            "This can lock confirmatory evaluation to one selected family."
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary-budget", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--validation-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--contrast-path", default=None)
    parser.add_argument("--figure-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = _parse_strings(args.models)
    budgets = _parse_float_values(args.budgets, "budgets")
    undersampling_factors = _parse_float_values(
        args.undersampling_factors,
        "undersampling-factors",
        allow_empty=True,
    )
    undersampling_families = _parse_rare_families(
        args.undersampling_families
    )
    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, args.random_state)
    factories = select_model_factories(
        model_names,
        crossfit_folds=args.crossfit_folds,
    )
    factories.update(
        rare_outcome_model_factories(
            undersampling_factors,
            families=undersampling_families,
        )
    )

    result = run_honest_uplift_experiment(
        dataset,
        model_factories=factories,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        budgets=budgets,
        primary_budget=args.primary_budget,
        confidence_level=args.confidence_level,
        random_state=args.random_state,
        selection_folds=args.selection_folds,
        progress=print,
    )

    prefix = f"criteo_{args.outcome}_honest"
    validation_path = ROOT / (
        args.validation_path
        or f"outputs/{prefix}_validation.csv"
    )
    test_path = ROOT / (
        args.test_path or f"outputs/{prefix}_test.csv"
    )
    contrast_path = ROOT / (
        args.contrast_path
        or f"outputs/{prefix}_contrasts.csv"
    )
    figure_path = ROOT / (
        args.figure_path
        or f"outputs/figures/{prefix}_policy_value.png"
    )
    report_path = ROOT / (
        args.report_path or f"outputs/{prefix}.md"
    )
    for path in (validation_path, test_path, contrast_path, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    result.validation_policy_values.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )
    result.test_policy_values.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )
    result.test_contrasts.to_csv(
        contrast_path,
        index=False,
        encoding="utf-8-sig",
    )
    plot_policy_value_curve(result.test_policy_values, figure_path)
    report_path.write_text(
        build_report(
            args,
            result,
            dataset_size=len(dataset.X),
            validation_path=validation_path,
            test_path=test_path,
            contrast_path=contrast_path,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )

    print(f"Out-of-sample-selected champion: {result.champion}")
    print(f"Locked-test report: {report_path}")
    print(result.test_contrasts.to_string(index=False))


def _parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one model is required.")
    return list(dict.fromkeys(values))


def _parse_rare_families(value: str) -> tuple[str, ...]:
    families = tuple(_parse_strings(value))
    unknown = sorted(set(families) - {"t", "cvt"})
    if unknown:
        raise ValueError(
            "undersampling-families supports only: t,cvt; "
            f"received {unknown}"
        )
    return families


def _parse_float_values(
    value: str,
    label: str,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated numbers.") from exc
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty.")
    return values


def _adjusted_verdict(survivors: list[str], champion: str) -> str:
    """State what survives the correction, in the terms the table is read in."""
    if not survivors:
        return (
            "No candidate clears zero once the table is read as a whole, so the "
            "champion is the best of a set that the selection sample cannot "
            "separate from response targeting."
        )
    if survivors == [champion]:
        return (
            f"`{champion}` is the only candidate still clearing zero, so the "
            "claim survives being read as a statement about the whole table."
        )
    others = ", ".join(f"`{name}`" for name in survivors if name != champion)
    if champion in survivors:
        return (
            f"`{champion}` still clears zero, and so does {others} - the "
            "correction narrows the field without isolating one candidate."
        )
    return (
        f"`{champion}` no longer clears zero, while {others} does. The rule "
        "picked on the unadjusted bound, so this is a caveat on the choice."
    )


def build_report(
    args: argparse.Namespace,
    result,
    dataset_size: int,
    validation_path: Path,
    test_path: Path,
    contrast_path: Path,
    figure_path: Path,
) -> str:
    primary_budget_pct = round(100.0 * args.primary_budget, 4)
    validation_primary = result.validation_policy_values[
        np.isclose(
            result.validation_policy_values["budget_pct"],
            primary_budget_pct,
        )
    ].merge(result.validation_metrics, on="policy", how="left")
    validation_contrasts = result.validation_contrasts[
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
    validation_primary = validation_primary.merge(
        validation_contrasts,
        on="policy",
        how="left",
    )
    adjusted = multiplicity_adjusted_bounds(
        result.validation_contrasts,
        candidate_policies=[
            name for name in result.validation_scores
            if name not in REFERENCE_POLICIES
        ],
        primary_budget=args.primary_budget,
        confidence_level=args.confidence_level,
    )
    validation_primary = validation_primary.merge(
        adjusted[["policy", "ci_lower_adjusted"]],
        on="policy",
        how="left",
    )
    validation_primary = validation_primary.sort_values(
        "difference_ci_lower",
        ascending=False,
        na_position="last",
    )
    n_candidates = int(adjusted["n_candidates"].iloc[0])
    adjusted_level = float(adjusted["adjusted_confidence_level"].iloc[0])
    survivors = adjusted[adjusted["ci_lower_adjusted"] > 0.0]["policy"].tolist()
    # The verdict describes the champion. The contrast table lists the
    # reference policies first, so it must be filtered by name rather than
    # read positionally.
    primary_contrast = result.test_contrasts[
        np.isclose(result.test_contrasts["budget_pct"], primary_budget_pct)
        & (result.test_contrasts["policy"] == result.champion)
    ].iloc[0]
    if primary_contrast["ci_lower"] > 0:
        decision = "confirmed a positive advantage over response targeting"
    elif primary_contrast["ci_upper"] < 0:
        decision = "showed a negative advantage relative to response targeting"
    else:
        decision = "was inconclusive relative to response targeting"
    test_fraction = 1.0 - args.train_fraction - args.validation_fraction
    split_summary = (
        f"`{args.train_fraction:.2f}` / `{args.validation_fraction:.2f}` / "
        f"`{test_fraction:.2f}`"
    )
    selection_description = (
        "one explicit validation holdout"
        if result.selection_folds == 1
        else (
            f"{result.selection_folds}-fold out-of-fold predictions on the "
            "combined development sample"
        )
    )

    return f"""# Honest Uplift Model Selection: Criteo {args.outcome}

## Locked Protocol

- Data: `{Path(args.sample_path).as_posix()}` ({dataset_size:,} rows), outcome `{args.outcome}`.
- Base train/validation/test fractions: {split_summary}.
- Selection folds: `{result.selection_folds}` over
  `{result.selection_size:,}` selection observations.
- Primary targeting budget: `{100.0 * args.primary_budget:.2f}%`.
- Confidence level: `{100.0 * args.confidence_level:.1f}%`.
- Random seed: `{args.random_state}`. Every model seed is derived from it
  and the model's name, so the run is a function of this number alone.
- Candidate model and hyperparameter selection uses {selection_description};
  locked-test outcomes are excluded.
- The champion is the candidate with the largest paired AIPW lower confidence
  bound against response targeting at the primary budget.
- Reference policies `response_model` and `random_targeting` are always
  evaluated but can never win selection.
- After selection, the champion and reference policies are refit on
  train + validation.
- Locked-test outcomes are opened once for the final comparison.

## Out-of-Sample Development Selection

{dataframe_to_markdown(validation_primary)}

Selected champion: **{result.champion}**.

`ci_lower_adjusted` re-derives each bound at
`{100.0 * adjusted_level:.2f}%`, which spreads the same
`{100.0 * (1.0 - args.confidence_level):.0f}%` error rate across the
`{n_candidates}` candidates. The unadjusted column answers "is this candidate
above response targeting"; the adjusted one answers "does any candidate in this
table stand above it", which is the question the selection rule actually asks by
keeping the largest bound. {_adjusted_verdict(survivors, result.champion)}

Selection still uses the unadjusted rule fixed before the data was seen.
Adjusting penalizes wide intervals more than narrow ones and so can reorder the
table, and switching to it here would be choosing a rule after seeing the
result.

## Locked-Test Policy Value

{dataframe_to_markdown(result.test_policy_values)}

![Locked-test policy value](figures/{figure_path.name})

## Paired Contrast Against Response Targeting

{dataframe_to_markdown(result.test_contrasts)}

At the pre-specified `{100.0 * args.primary_budget:.2f}%` budget, the locked test
{decision}. The estimated difference is `{primary_contrast['difference']:.4f}`
incremental {args.outcome} outcomes with a confidence interval of
`[{primary_contrast['ci_lower']:.4f}, {primary_contrast['ci_upper']:.4f}]`.

## Ranking Metrics

{dataframe_to_markdown(result.test_metrics)}

AUUC is secondary. The decision is based on the budget-specific AIPW policy
contrast because the campaign has a fixed operating budget.

## Runtime

{dataframe_to_markdown(result.timing_table)}

## Statistical Scope

The AIPW intervals account for evaluation-sample uncertainty conditional on the
locked fitted policies. Repeated honest splits are required to quantify training
and selection instability. No production-impact claim is made without a live
randomized experiment.

## Reproducible Outputs

- Selection values: `{validation_path.relative_to(ROOT).as_posix()}`
- Locked-test values: `{test_path.relative_to(ROOT).as_posix()}`
- Paired contrasts: `{contrast_path.relative_to(ROOT).as_posix()}`
"""


if __name__ == "__main__":
    main()
