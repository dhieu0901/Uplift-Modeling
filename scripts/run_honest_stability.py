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
from src.experiments.honest_uplift import (
    run_honest_uplift_experiment,
    summarize_selection,
)
from src.models.registry import (
    rare_outcome_model_factories,
    select_model_factories,
)
from src.reporting import dataframe_to_markdown

DEFAULT_MODELS = "response_model,s_learner"


def report(message: str) -> None:
    """Print progress so it survives redirection to a file.

    A run of this length is normally watched through a log, and Python buffers
    stdout when it is not a terminal. Without the flush, nothing appears until
    the process exits, which is exactly when the progress stops being useful.
    """
    print(message, flush=True)


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
    parser.add_argument(
        "--rebuild-report",
        action="store_true",
        help=(
            "Redraw the report and figure from the saved table without "
            "repeating the experiment. Changes no number."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_report:
        rebuild_report(args)
        return
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
    for position, seed in enumerate(seeds, start=1):
        report(f"\n=== Honest split {position}/{len(seeds)}, seed {seed} ===")
        result = run_honest_uplift_experiment(
            dataset,
            model_factories=factories,
            budgets=budgets,
            primary_budget=args.primary_budget,
            random_state=seed,
            selection_folds=args.selection_folds,
            progress=report,
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
                # Carried on every row so the table records the protocol that
                # produced it. A report rebuilt from the table then describes
                # the run that happened rather than whatever defaults the
                # rebuilding command was invoked with.
                "sample_path": args.sample_path,
                "models": args.models,
                "primary_budget": args.primary_budget,
                "dataset_rows": len(dataset.X),
                "selection_folds": result.selection_folds,
                "selection_size": result.selection_size,
            }
        )

    write_outputs(args, pd.DataFrame(rows))


def summarize_runs(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return summary, champion_frequency


PROTOCOL_COLUMNS = (
    "sample_path",
    "models",
    "primary_budget",
    "dataset_rows",
    "selection_folds",
    "selection_size",
)


def rebuild_report(args: argparse.Namespace) -> None:
    """Rewrite the report and figure from the saved table.

    The experiment behind this table takes hours and its wording is the part
    most likely to need another pass, so redrawing has to be possible without
    repeating it. The protocol is read from the table rather than from this
    invocation: falling back to command-line defaults would let a rebuilt
    report describe a run that never happened.
    """
    results_path = ROOT / args.results_path
    if not results_path.exists():
        raise SystemExit(f"No saved results at {results_path}.")
    results = pd.read_csv(results_path)
    missing = [name for name in PROTOCOL_COLUMNS if name not in results.columns]
    if missing:
        raise SystemExit(
            f"{results_path.name} does not record its own protocol "
            f"(missing {', '.join(missing)}), so the report cannot be rebuilt "
            "from it without guessing. Re-run the experiment instead."
        )
    write_outputs(args, results, save_table=False)


def write_outputs(
    args: argparse.Namespace,
    results: pd.DataFrame,
    save_table: bool = True,
) -> None:
    summary, champion_frequency = summarize_runs(results)
    first = results.iloc[0]
    protocol = {
        "sample_path": str(first["sample_path"]),
        "models": str(first["models"]),
        "primary_budget": float(first["primary_budget"]),
        "dataset_rows": int(first["dataset_rows"]),
        "selection_folds": int(first["selection_folds"]),
        "selection_size": int(first["selection_size"]),
        "seeds": ",".join(str(int(value)) for value in results["seed"]),
    }

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
    if save_table:
        results.to_csv(results_path, index=False, encoding="utf-8-sig")
    plot_stability(results, figure_path)
    report_path.write_text(
        build_report(
            args,
            protocol,
            results,
            summary,
            champion_frequency,
            results_path,
            figure_path,
        ),
        encoding="utf-8",
    )
    report(f"Stability report: {report_path}")
    report(summary.to_string(index=False))


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
        starved = int((cleared == 0).sum())
        if starved:
            lines.append(
                f"- Runs in which no candidate reached a positive selection "
                f"bound: **{starved} of {len(cleared)}**. The rule names a "
                f"champion anyway, because it ranks candidates rather than "
                f"requiring one to clear a bar."
            )
        else:
            lines.append(
                f"- Every run had at least one candidate clear zero, a median "
                f"of **{cleared.median():.0f}**. The rule ranks rather than "
                f"requiring a bar to be cleared, so this is a property of the "
                f"selection sample rather than something the rule enforces."
            )
    always_near_top = False
    if not ranks.empty:
        top_two = int((ranks <= 2).sum())
        always_near_top = top_two == len(ranks)
        lines.append(
            f"- `s_learner` finished first or second in "
            f"**{top_two} of {len(ranks)}** runs "
            f"(median rank {ranks.median():.0f})."
        )

    if halfwidths.empty:
        return "\n".join(lines)
    if margins.median() >= halfwidths.median():
        return "\n".join(lines) + "\n\n" + (
            "The gap between first and second place exceeds the uncertainty "
            "attached to first place, so the changes in champion are not "
            "explained by selection noise alone and the ranking is genuinely "
            "sensitive to the split."
        )

    verdict = (
        "The gap between first and second place is smaller than the "
        "uncertainty attached to first place itself, so no candidate is "
        "measurably better than the one immediately below it."
    )
    if always_near_top:
        verdict += (
            f" The ordering is not arbitrary either: `{modal_champion}` never "
            "leaves the top two. Pairwise gaps inside the noise and a stable "
            "leader are consistent with each other, and together they say the "
            "sample can rank these candidates without being able to separate "
            "them. The defensible claim is about the policy class rather than "
            "about one architecture."
        )
    else:
        verdict += (
            " A changing champion here reflects candidates the selection "
            "sample cannot separate, so the frequency table is a ranking "
            "tendency rather than evidence that one architecture wins."
        )
    return "\n".join(lines) + "\n\n" + verdict


def build_report(
    args,
    protocol: dict,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    champion_frequency: pd.DataFrame,
    results_path: Path,
    figure_path: Path,
) -> str:
    selection_note = describe_selection_closeness(results)
    folds = int(protocol["selection_folds"])
    size = int(protocol["selection_size"])
    selection_description = (
        f"one explicit validation holdout over `{size:,}` observations"
        if folds == 1
        else f"`{folds}`-fold out-of-fold predictions over `{size:,}` observations"
    )
    return f"""# End-to-End Honest-Split Stability: Criteo {args.outcome}

## Protocol

- Data: `{protocol["sample_path"]}` ({int(protocol["dataset_rows"]):,} rows).
- Seeds: `{protocol["seeds"]}`.
- Primary budget: `{100.0 * float(protocol["primary_budget"]):.2f}%`.
- Candidate models: `{protocol["models"]}`.
- Selection: {selection_description}.
- Every run repeats training, out-of-sample model selection, development
  refitting, nuisance estimation, and locked-test evaluation.
- Each run uses the same pre-specified candidate set and selection rule.

The selection stage is the one being measured, so it has to match the run that
produced the headline champion. A smaller selection sample widens every
candidate's interval, which would show up here as instability that belongs to
the sample size rather than to the rule.

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
