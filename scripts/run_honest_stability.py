# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.experiments.honest_uplift import run_honest_uplift_experiment
from src.models.registry import (
    REFERENCE_POLICIES,
    rare_outcome_model_factories,
    select_model_factories,
)
from src.reporting import dataframe_to_markdown

DEFAULT_MODELS = "response_model,s_learner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the complete out-of-sample selection and locked-test protocol "
            "to measure training and split instability."
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
    parser.add_argument("--selection-folds", type=int, default=1)
    parser.add_argument("--undersampling-factors", default="")
    parser.add_argument(
        "--seeds",
        default="42,123,2026,730,991,1201,1601,2401,3301,4401",
    )
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary-budget", type=float, default=0.05)
    parser.add_argument(
        "--report-path",
        default="outputs/visit_stability.md",
    )
    parser.add_argument(
        "--results-path",
        default="outputs/tables/visit_stability.csv",
    )
    parser.add_argument(
        "--figure-path",
        default="outputs/figures/visit_stability.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_integers(args.seeds, "seeds")
    budgets = _parse_floats(args.budgets, "budgets")
    factors = _parse_floats(
        args.undersampling_factors,
        "undersampling-factors",
        allow_empty=True,
    )
    model_names = _parse_strings(args.models)
    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, seeds[0])
    factories = select_model_factories(
        model_names,
        crossfit_folds=args.crossfit_folds,
    )
    factories.update(rare_outcome_model_factories(factors))

    rows = []
    for seed in seeds:
        print(f"\n=== Honest split seed {seed} ===")
        result = run_honest_uplift_experiment(
            dataset,
            model_factories=factories,
            budgets=budgets,
            primary_budget=args.primary_budget,
            random_state=seed,
            selection_folds=args.selection_folds,
            progress=print,
        )
        budget_pct = 100.0 * args.primary_budget
        contrast = result.test_contrasts[
            np.isclose(result.test_contrasts["budget_pct"], budget_pct)
            & (result.test_contrasts["policy"] == result.champion)
        ].iloc[0]
        policy_values = result.test_policy_values[
            np.isclose(result.test_policy_values["budget_pct"], budget_pct)
        ].set_index("policy")
        metrics = result.test_metrics.set_index("policy")
        selection = summarize_selection(result, budget_pct)
        rows.append(
            {
                "seed": seed,
                "champion": result.champion,
                **selection,
                "difference_vs_response": contrast["difference"],
                "ci_lower": contrast["ci_lower"],
                "ci_upper": contrast["ci_upper"],
                "champion_incremental_outcome": policy_values.loc[
                    result.champion, "incremental_outcome"
                ],
                "response_incremental_outcome": policy_values.loc[
                    "response_model", "incremental_outcome"
                ],
                "champion_auuc": metrics.loc[
                    result.champion, "benchmark_relative_auuc"
                ],
                "response_auuc": metrics.loc[
                    "response_model", "benchmark_relative_auuc"
                ],
                "fit_seconds": result.timing_table["fit_seconds"].sum(),
            }
        )

    results = pd.DataFrame(rows)
    champion_frequency = (
        results.groupby("champion", as_index=False)
        .agg(
            runs=("seed", "size"),
            mean_difference=("difference_vs_response", "mean"),
            positive_rate=("difference_vs_response", lambda x: (x > 0).mean()),
        )
        .sort_values(["runs", "mean_difference"], ascending=False)
    )
    summary = pd.DataFrame(
        [
            {
                "runs": len(results),
                "mean_difference": results["difference_vs_response"].mean(),
                "std_difference": results["difference_vs_response"].std(),
                "min_difference": results["difference_vs_response"].min(),
                "max_difference": results["difference_vs_response"].max(),
                "positive_point_rate": (
                    results["difference_vs_response"] > 0
                ).mean(),
                "positive_ci_rate": (results["ci_lower"] > 0).mean(),
                "negative_ci_rate": (results["ci_upper"] < 0).mean(),
            }
        ]
    )

    prefix = f"criteo_{args.outcome}_honest_stability"
    results_path = ROOT / (
        args.results_path or f"outputs/{prefix}.csv"
    )
    figure_path = ROOT / (
        args.figure_path
        or f"outputs/figures/{prefix}.png"
    )
    report_path = ROOT / (
        args.report_path or f"outputs/{prefix}.md"
    )
    for path in (results_path, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    plot_stability(results, figure_path)
    report_path.write_text(
        build_report(
            args,
            len(dataset.X),
            results,
            summary,
            champion_frequency,
            results_path,
            figure_path,
        ),
        encoding="utf-8",
    )
    print(f"Stability report: {report_path}")
    print(summary.to_string(index=False))


def summarize_selection(result, budget_pct: float) -> dict:
    """Record how close the selection was, not just who won.

    A champion that changes across splits can mean two very different things:
    the rule is unstable, or several candidates are genuinely tied and the
    rule is picking arbitrarily among equals. Those call for different
    responses, and the champion name alone cannot tell them apart. Recording
    the runner-up and the margin does.
    """
    contrasts = result.validation_contrasts
    ranked = contrasts[
        np.isclose(contrasts["budget_pct"], budget_pct)
        & contrasts["policy"].isin(
            [name for name in result.validation_scores if name not in REFERENCE_POLICIES]
        )
    ].dropna(subset=["ci_lower"]).sort_values("ci_lower", ascending=False)

    if ranked.empty:
        return {}

    champion_row = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None
    order = list(ranked["policy"])

    return {
        "runner_up": None if runner_up is None else str(runner_up["policy"]),
        "selection_margin": (
            float("nan")
            if runner_up is None
            else float(champion_row["ci_lower"] - runner_up["ci_lower"])
        ),
        "champion_selection_ci_lower": float(champion_row["ci_lower"]),
        "runner_up_selection_ci_lower": (
            float("nan") if runner_up is None else float(runner_up["ci_lower"])
        ),
        # The margin only means something next to the uncertainty in the
        # quantity being ranked, so the champion's own interval travels with it.
        "champion_selection_halfwidth": float(
            (champion_row["ci_upper"] - champion_row["ci_lower"]) / 2.0
        ),
        "n_candidates_with_positive_bound": int((ranked["ci_lower"] > 0).sum()),
        "s_learner_selection_rank": (
            order.index("s_learner") + 1 if "s_learner" in order else -1
        ),
    }


def _parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("models must not be empty.")
    return list(dict.fromkeys(values))


def _parse_integers(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers.") from exc
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain at least two unique values.")
    return values


def _parse_floats(
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


def plot_stability(results: pd.DataFrame, output_path: Path) -> None:
    ordered = results.reset_index(drop=True)
    center = ordered["difference_vs_response"].to_numpy(dtype=float)
    lower = ordered["ci_lower"].to_numpy(dtype=float)
    upper = ordered["ci_upper"].to_numpy(dtype=float)
    x = np.arange(len(ordered))
    colors = np.where(lower > 0.0, "#147d64", "#7a8793")

    figure, axis = plt.subplots(figsize=(10, 5.4))
    for index in x:
        axis.errorbar(
            index,
            center[index],
            yerr=np.array(
                [
                    [center[index] - lower[index]],
                    [upper[index] - center[index]],
                ]
            ),
            marker="o",
            capsize=4,
            color=colors[index],
        )
    axis.axhline(0.0, color="#b43c59", linewidth=1.2)
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["seed"].astype(str))
    axis.set_title("End-to-end honest-split stability at the primary budget")
    axis.set_xlabel("Split seed")
    axis.set_ylabel("Champion minus response incremental outcomes")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read an optional diagnostic column, empty when it was not recorded."""
    if name not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").dropna()


def describe_selection_closeness(results: pd.DataFrame) -> str:
    """State whether a changing champion means instability or a tie.

    The margin is read against the width of the champion's own selection
    interval rather than against the effect size. A gap the estimator cannot
    resolve means the candidates are tied; a gap it can resolve means the
    ranking really does move between splits.
    """
    if "selection_margin" not in results.columns:
        return "Selection margins were not recorded for this run."

    margins = pd.to_numeric(results["selection_margin"], errors="coerce").dropna()
    if margins.empty:
        return "Selection margins were not recorded for this run."

    modal = results["champion"].mode()
    modal_champion = str(modal.iloc[0]) if not modal.empty else "the modal champion"
    lost = results[results["champion"] != modal_champion]
    halfwidths = _numeric_column(results, "champion_selection_halfwidth")
    ranks = _numeric_column(results, "s_learner_selection_rank")
    ranks = ranks[ranks > 0]

    lines = [
        f"- Median margin between the champion and the runner-up: "
        f"`{margins.median():.1f}` incremental outcomes.",
        f"- Runs where `{modal_champion}` was not selected: "
        f"**{len(lost)} of {len(results)}**.",
    ]
    if not halfwidths.empty:
        ratio = margins.median() / halfwidths.median()
        lines.append(
            f"- Median half-width of the champion's own selection interval: "
            f"`{halfwidths.median():.1f}`. The margin is `{ratio:.2f}` times "
            f"that width."
        )
    cleared = _numeric_column(results, "n_candidates_with_positive_bound")
    if not cleared.empty:
        lines.append(
            f"- Runs in which no candidate reached a positive selection bound: "
            f"**{int((cleared == 0).sum())} of {len(cleared)}**. The rule still "
            f"names a champion in those runs, because it ranks candidates rather "
            f"than requiring one to clear a bar."
        )
    if not ranks.empty:
        lines.append(
            f"- `s_learner` finished first or second in "
            f"**{int((ranks <= 2).sum())} of {len(ranks)}** runs "
            f"(median rank {ranks.median():.0f})."
        )

    if not halfwidths.empty and margins.median() < halfwidths.median():
        verdict = (
            "The gap between first and second place is smaller than the "
            "uncertainty attached to first place itself, so the leaderboard "
            "reorders under resampling without any candidate being measurably "
            "better. A changing champion here reflects candidates the selection "
            "sample cannot separate, and the frequency table should be read as "
            "a ranking tendency rather than as evidence that one architecture "
            "wins."
        )
    else:
        verdict = (
            "The gap between first and second place exceeds the uncertainty "
            "attached to first place, so the changes in champion are not "
            "explained by selection noise alone and the ranking is genuinely "
            "sensitive to the split."
        )
    return "\n".join(lines) + "\n\n" + verdict


def build_report(
    args,
    dataset_size: int,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    champion_frequency: pd.DataFrame,
    results_path: Path,
    figure_path: Path,
) -> str:
    selection_note = describe_selection_closeness(results)
    return f"""# End-to-End Honest-Split Stability: Criteo {args.outcome}

## Protocol

- Data: `{args.sample_path}` ({dataset_size:,} rows).
- Seeds: `{args.seeds}`.
- Primary budget: `{100.0 * args.primary_budget:.2f}%`.
- Candidate models: `{args.models}`.
- Every run repeats training, out-of-sample model selection, development
  refitting, nuisance estimation, and locked-test evaluation.
- Each run uses the same pre-specified candidate set and selection rule.

## Aggregate Stability

{dataframe_to_markdown(summary)}

## Champion Frequency

{dataframe_to_markdown(champion_frequency)}

## How Close Was Each Selection

{selection_note}

## Results by Split

{dataframe_to_markdown(results)}

![Honest-split stability](figures/{figure_path.name})

These repeated splits overlap and are therefore correlated robustness checks,
not independent experiments. They measure sensitivity to training and partition
variation; the canonical locked test remains the primary result.

Raw results: `{results_path.relative_to(ROOT).as_posix()}`.
"""


if __name__ == "__main__":
    main()
