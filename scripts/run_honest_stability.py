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
        rows.append(
            {
                "seed": seed,
                "champion": result.champion,
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


def build_report(
    args,
    dataset_size: int,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    champion_frequency: pd.DataFrame,
    results_path: Path,
    figure_path: Path,
) -> str:
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
