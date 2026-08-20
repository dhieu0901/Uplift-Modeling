# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.experiments.honest_uplift import (
    run_honest_uplift_experiment,
    summarize_selection,
)
from src.models.base import BASE_LEARNER_FAMILIES
from src.models.registry import REFERENCE_POLICIES, select_model_factories
from src.reporting import dataframe_to_markdown, plot_base_learner_comparison

DEFAULT_MODELS = (
    "response_model,s_learner,t_learner,x_learner,cvt,"
    "transformed_outcome,r_learner,dr_learner"
)

#: Below this a rerun counts as reproducing the published table. The quantities
#: compared are outcome counts in the thousands, so anything near float64
#: epsilon on them is a sum reassembled in a different order rather than a
#: model that changed.
REPRODUCTION_TOLERANCE = 1e-6


def report(message: str) -> None:
    """Print progress so it survives redirection to a file."""
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked selection protocol once per base learner family "
            "and report whether the ranking of uplift learners depends on the "
            "estimator those learners are built from."
        )
    )
    parser.add_argument(
        "--sample-path",
        default="data/processed/criteo_audit_1m.parquet",
        help=(
            "Defaults to the sample the locked selection ran on. Matching it "
            "is the point: a smaller sample would widen every interval and "
            "show up as a ranking that depends on the base learner when it "
            "actually depends on how much data the selection stage was given."
        ),
    )
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--families",
        default=",".join(BASE_LEARNER_FAMILIES),
        help="Comma-separated base learner families to compare.",
    )
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--selection-folds",
        type=int,
        default=3,
        help="Matches the locked run for the reason given under --sample-path.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary-budget", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--random-state",
        type=int,
        default=777,
        help="The seed the locked selection used, so one column reproduces it.",
    )
    parser.add_argument(
        "--baseline-selection-path",
        default="outputs/tables/audit_visit_selection.csv",
        help=(
            "Published selection table to check against. Setting this runs the "
            "locked configuration once more as a reproduction anchor. Pass an "
            "empty string to skip both."
        ),
    )
    parser.add_argument(
        "--report-path", default="outputs/base_learner_comparison.md"
    )
    parser.add_argument(
        "--results-path", default="outputs/tables/base_learner_comparison.csv"
    )
    parser.add_argument(
        "--summary-path", default="outputs/tables/base_learner_selection_summary.csv"
    )
    parser.add_argument(
        "--figure-path", default="outputs/figures/base_learner_comparison.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = _parse_strings(args.models)
    families = _parse_families(args.families)
    budgets = _parse_floats(args.budgets)
    if args.primary_budget not in budgets:
        raise ValueError("primary-budget must be included in budgets.")
    budget_pct = 100.0 * args.primary_budget

    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, args.random_state)
    report(
        f"Sample: {args.sample_path} ({len(dataset.X):,} rows), "
        f"outcome {args.outcome}, {len(families)} families"
    )

    contrast_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    anchor_values = None
    if args.baseline_selection_path:
        report("\n=== Reproduction anchor: the locked configuration ===")
        anchor = run_honest_uplift_experiment(
            dataset,
            model_factories=select_model_factories(
                model_names, crossfit_folds=args.crossfit_folds
            ),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            budgets=budgets,
            primary_budget=args.primary_budget,
            confidence_level=args.confidence_level,
            random_state=args.random_state,
            selection_folds=args.selection_folds,
            progress=report,
        )
        anchor_values = anchor.validation_policy_values
        report(f"anchor champion: {anchor.champion}")

    for position, family in enumerate(families, start=1):
        report(f"\n=== Base learner family {position}/{len(families)}: {family} ===")
        factories = select_model_factories(
            model_names,
            crossfit_folds=args.crossfit_folds,
            base_family=family,
        )
        started = perf_counter()
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
            progress=report,
        )
        elapsed = perf_counter() - started

        contrasts = _selection_contrasts(result, budget_pct, family)
        contrast_rows.append(contrasts)
        summary = summarize_selection(result, budget_pct)
        summary_rows.append(
            {
                "base_family": family,
                "champion": result.champion,
                **summary,
                "margin_over_halfwidth": _ratio(
                    summary.get("selection_margin"),
                    summary.get("champion_selection_halfwidth"),
                ),
                "wall_seconds": elapsed,
            }
        )
        report(
            f"{family}: champion {result.champion} "
            f"in {elapsed / 60:.1f} min"
        )

    results = pd.concat(contrast_rows, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)

    results_path = ROOT / args.results_path
    summary_path = ROOT / args.summary_path
    figure_path = ROOT / args.figure_path
    report_path = ROOT / args.report_path
    for path in (results_path, summary_path, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_base_learner_comparison(results, figure_path)

    baseline = _baseline_deviation(args.baseline_selection_path, anchor_values)
    report_path.write_text(
        build_report(
            args,
            results=results,
            summary=summary_table,
            families=families,
            model_names=model_names,
            n_rows=len(dataset.X),
            budget_pct=budget_pct,
            baseline=baseline,
            results_path=results_path,
            summary_path=summary_path,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )
    report(f"\nComparison report: {report_path}")
    report(summary_table.to_string(index=False))


def _selection_contrasts(result, budget_pct: float, family: str) -> pd.DataFrame:
    """Take the selection-stage contrasts the rule reads, for one family.

    The selection stage is the whole study. The locked test split is left
    unopened here: opening it once per family would be several looks at one
    holdout, and no question this report asks needs it.
    """
    candidates = [
        name for name in result.validation_scores if name not in REFERENCE_POLICIES
    ]
    contrasts = result.validation_contrasts
    selected = contrasts[
        np.isclose(contrasts["budget_pct"], budget_pct)
        & contrasts["policy"].isin(candidates)
    ].copy()
    selected = selected.sort_values("ci_lower", ascending=False).reset_index(drop=True)
    selected.insert(0, "base_family", family)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _ratio(
    numerator: float | np.floating | None,
    denominator: float | np.floating | None,
) -> float:
    """Divide, returning a plain float and never raising.

    The operands reach here off a summary dictionary, so they are numpy scalars
    rather than Python floats. The annotation says so, and the cast on the way
    out is what keeps a numpy type out of the report and the CSV.
    """
    if numerator is None or denominator is None:
        return float("nan")
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _baseline_deviation(
    baseline_path: str,
    selection_values: pd.DataFrame | None,
) -> dict | None:
    """Compare the anchor run against the published selection table.

    Base learner families were added by threading an estimator choice through
    learners that used to build their own. That refactor is only safe if the
    path taken when no family is named is untouched, and the strongest
    available evidence is that a fresh run of it reproduces a table on disk
    that predates the change.

    The anchor is the locked configuration, not the boosted column. Those are
    not the same thing, and the Protocol section says where they part.
    """
    if not baseline_path or selection_values is None:
        return None
    path = ROOT / baseline_path
    if not path.exists():
        return None
    published = pd.read_csv(path, encoding="utf-8-sig")
    keys = ["policy", "budget_pct"]
    compared = published.merge(
        selection_values,
        on=keys,
        suffixes=("_published", "_rerun"),
        how="inner",
    )
    if compared.empty:
        return None
    columns = [
        column
        for column in ("incremental_outcome", "ci_lower", "ci_upper")
        if f"{column}_published" in compared.columns
    ]
    deviations = {
        column: float(
            (compared[f"{column}_published"] - compared[f"{column}_rerun"]).abs().max()
        )
        for column in columns
    }
    worst = max(deviations.values()) if deviations else 0.0
    moved = sorted(
        {
            str(policy)
            for column in columns
            for policy in compared.loc[
                (
                    compared[f"{column}_published"] - compared[f"{column}_rerun"]
                ).abs()
                > REPRODUCTION_TOLERANCE,
                "policy",
            ]
        }
    )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(compared),
        "max_absolute_deviation": worst,
        "by_column": deviations,
        "policies_that_moved": moved,
    }


def build_report(
    args: argparse.Namespace,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    families: list[str],
    model_names: list[str],
    n_rows: int,
    budget_pct: float,
    baseline: dict | None,
    results_path: Path,
    summary_path: Path,
    figure_path: Path,
) -> str:
    matrix = _bound_matrix(results, families)
    champion_view = summary[
        [
            "base_family",
            "champion",
            "runner_up",
            "selection_margin",
            "champion_selection_halfwidth",
            "margin_over_halfwidth",
            "n_candidates_with_positive_bound",
        ]
    ]
    lines = [
        "# Does the Champion Depend on the Base Learner?",
        "",
        "## Why This Exists",
        "",
        "Every uplift learner in this project is a recipe that turns ordinary",
        "supervised estimators into an effect estimate, and every published run",
        "gave all of them the same boosted trees. A ranking produced that way is",
        "a statement about the recipes *at one estimator*, which is weaker than",
        "the statement the README makes on the strength of it.",
        "",
        "The repeated-splits experiment in `visit_stability.md` varies the data",
        "the recipes see. It holds the estimator fixed, so it cannot answer this.",
        "This report varies the estimator and holds everything else fixed.",
        "",
        "It is a development-stage study. It re-reads a decision that has already",
        "been made rather than making a new one, so it changes no locked policy",
        "and opens no confirmatory sample.",
        "",
        "## Protocol",
        "",
        f"- Sample: `{args.sample_path}`, `{n_rows:,}` rows, outcome `{args.outcome}`.",
        f"- Selection: `{args.selection_folds}`-fold out-of-fold policy selection,",
        f"  seed `{args.random_state}`, decided at the `{budget_pct:g}%` budget.",
        f"- Candidates: `{len(model_names) - 1}` uplift learners, `{len(families)}`",
        f"  base learner families ({', '.join(f'`{name}`' for name in families)}).",
        "- Sample, folds, seed, and budget all match the run that chose the locked",
        "  champion. A cheaper selection stage would widen every interval and show",
        "  up here as a ranking that moves with the base learner when it actually",
        "  moved with the sample size.",
        "- Reference policies stay on boosted trees in every column. Response",
        "  targeting stands for what the business already does, so it has to be one",
        "  fixed bar rather than a bar that moves with the family under test.",
        "- The AIPW nuisance models that score every candidate also stay on boosted",
        "  trees. The estimator under test is the one inside the candidates, not the",
        "  one doing the measuring, and moving both at once would confound them.",
        "- The `gradient_boosting` column is not the locked configuration. The locked",
        "  one was never uniform: `transformed_outcome` uses ridge on purpose, because",
        "  at a propensity of 0.85 its regression target takes three values and a",
        "  flexible learner fits the spikes. Making the column uniform overrides that",
        "  choice, which is a claim worth testing rather than one to preserve. Every",
        "  other candidate in that column is the locked one, to the last bit.",
        "",
        "## Selection Bound by Base Learner",
        "",
        f"Lower bound of the {args.confidence_level:.0%} interval on incremental",
        f"outcomes against response targeting at the {budget_pct:g}% budget, which",
        "is the number the selection rule reads. Rank within the family is in",
        "brackets. Rows are ordered by the boosted-tree bound, so a family that",
        "reorders the candidates shows up as brackets out of sequence.",
        "",
        dataframe_to_markdown(matrix),
        "",
        # Relative to the report, which sits one level above the figures.
        f"![Base learner comparison]("
        f"{figure_path.relative_to(ROOT / 'outputs').as_posix()})",
        "",
        "## Who Wins in Each Family",
        "",
        dataframe_to_markdown(champion_view),
        "",
        "`margin_over_halfwidth` is the gap between first and second place divided",
        "by the half-width of the winner's own interval. Below 1 the sample cannot",
        "resolve the gap it is ranking by, so the ordering is real but the winner is",
        "not separated from the runner-up.",
        "",
        "## What This Shows",
        "",
        *_verdict(results, summary, families),
        "",
    ]
    if baseline is not None:
        lines.extend(
            [
                "## Reproduction Check",
                "",
                "Before any column was compared, the locked configuration was run",
                "again from the current code and checked row for row against",
                f"`{baseline['path']}`, which was written before base learner",
                f"families existed. Across `{baseline['rows']}` matched rows the",
                "largest absolute difference in any reported quantity is",
                f"`{baseline['max_absolute_deviation']:.10g}`.",
                "",
                _reproduction_verdict(
                    baseline["max_absolute_deviation"],
                    baseline["policies_that_moved"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Reproducible Outputs",
            "",
            f"- Every candidate in every family: `{results_path.relative_to(ROOT).as_posix()}`",
            f"- One row per family: `{summary_path.relative_to(ROOT).as_posix()}`",
            f"- Figure: `{figure_path.relative_to(ROOT).as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def _bound_matrix(results: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Lay the selection bounds out as candidates by families, with ranks."""
    ordering = (
        results[results["base_family"] == families[0]]
        .sort_values("ci_lower", ascending=False)["policy"]
        .tolist()
    )
    ordering += [
        str(policy) for policy in dict.fromkeys(results["policy"]) if policy not in ordering
    ]
    rows = []
    for policy in ordering:
        row = {"policy": policy}
        for family in families:
            match = results[
                (results["base_family"] == family) & (results["policy"] == policy)
            ]
            if match.empty:
                row[family] = "-"
                continue
            bound = float(match["ci_lower"].iloc[0])
            rank = int(match["selection_rank"].iloc[0])
            row[family] = f"{bound:.1f} ({rank})"
        rows.append(row)
    return pd.DataFrame(rows)


def _verdict(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    families: list[str],
) -> list[str]:
    """State what the table supports, including when that is little."""
    champions = dict(zip(summary["base_family"], summary["champion"], strict=True))
    distinct = sorted(set(champions.values()))
    named = ", ".join(f"`{family}` picks `{champions[family]}`" for family in families)

    lines = []
    if len(distinct) == 1:
        lines.append(
            f"**The same learner wins in every family.** {named}. The locked "
            f"champion is therefore not an artifact of the estimator it was "
            f"built on, which is the specific thing this report was written to "
            f"check."
        )
    else:
        lines.append(
            f"**The winner changes with the base learner.** {named}. The locked "
            f"champion was selected at one estimator, so its lead is a property "
            f"of that pairing rather than of the learner alone."
        )

    ranks = _rank_matrix(results, families)
    stable = [policy for policy, values in ranks.items() if len(set(values)) == 1]
    movers = {
        policy: values
        for policy, values in ranks.items()
        if max(values) - min(values) >= 3
    }
    lines.append("")
    if stable:
        lines.append(
            "Holding the same rank in every family: "
            + ", ".join(f"`{policy}`" for policy in sorted(stable))
            + "."
        )
    else:
        lines.append("No candidate holds the same rank in every family.")
    if movers:
        described = ", ".join(
            f"`{policy}` ({min(values)} to {max(values)})"
            for policy, values in sorted(movers.items())
        )
        lines.append(
            f"Moving three places or more between families: {described}. A "
            "candidate that swings this far is being judged on its estimator as "
            "much as on its recipe."
        )

    ratios = pd.to_numeric(summary["margin_over_halfwidth"], errors="coerce").dropna()
    lines.append("")
    if ratios.empty:
        lines.append("No selection margin was recorded, so closeness cannot be read.")
    elif (ratios < 1.0).all():
        lines.append(
            f"In every family the winning margin is smaller than the half-width "
            f"of the winner's own interval (largest ratio "
            f"`{ratios.max():.2f}`). So none of these columns separates its "
            f"first place from its second, and a reordering between families is "
            f"movement inside the noise rather than evidence that one estimator "
            f"suits one recipe better. The defensible reading is about the "
            f"policy class, not about a single best learner."
        )
    else:
        resolved = summary[ratios.ge(1.0).reindex(summary.index, fill_value=False)]
        lines.append(
            "The winning margin exceeds the winner's own half-width in: "
            + ", ".join(f"`{name}`" for name in resolved["base_family"])
            + ". In those families the selection sample can resolve the gap it "
            "ranks by, so first place there is a measured result rather than a "
            "coin flip among equals."
        )
    return lines


def _rank_matrix(
    results: pd.DataFrame,
    families: list[str],
) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for policy in dict.fromkeys(results["policy"]):
        values = []
        for family in families:
            match = results[
                (results["base_family"] == family) & (results["policy"] == policy)
            ]
            if not match.empty:
                values.append(int(match["selection_rank"].iloc[0]))
        if len(values) == len(families):
            ranks[str(policy)] = values
    return ranks


def _reproduction_verdict(deviation: float, moved: list[str]) -> str:
    if deviation <= REPRODUCTION_TOLERANCE:
        return (
            "No policy moved. What remains is float round-off on counts in the "
            "thousands, which is what a sum reassembled in a different order "
            "costs, so threading the estimator choice through the learners left "
            "the published path as it was and the columns above can be read "
            "against it."
        )
    named = ", ".join(f"`{policy}`" for policy in moved) or "no named policy"
    return (
        f"The values differ, on {named}. The locked path is therefore not the "
        "one that produced the published table, and nothing above can be read "
        "against it until that is explained."
    )


def _parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one model is required.")
    return list(dict.fromkeys(values))


def _parse_families(value: str) -> list[str]:
    families = _parse_strings(value)
    unknown = sorted(set(families) - set(BASE_LEARNER_FAMILIES))
    if unknown:
        known = ", ".join(sorted(BASE_LEARNER_FAMILIES))
        raise ValueError(f"Unsupported families: {unknown}. Known: {known}.")
    return families


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


if __name__ == "__main__":
    main()
