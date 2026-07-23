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
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import CriteoDataset, load_criteo, subsample_criteo
from src.evaluation.uplift import (
    auuc,
    budget_policy_table,
    cumulative_uplift_curve,
    qini_coefficient,
    separate_relative_auuc,
    uplift_by_quantile,
)
from src.experiments.criteo import run_criteo_experiment
from src.reporting import dataframe_to_markdown


UPLIFT_POLICIES = [
    "s_learner",
    "t_learner",
    "x_learner",
    "cvt",
    "transformed_outcome",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an uplift experiment on a Criteo sample.")
    parser.add_argument("--sample-path", default="data/processed/criteo_sample_500k.parquet")
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated policy list; random is always added automatically.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--figure-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, args.random_state)
    policies = parse_policies(args.policies)
    experiment = run_criteo_experiment(
        dataset,
        test_fraction=args.test_size,
        random_state=args.random_state,
        progress=lambda name: print(f"Training {name}..."),
        policies=policies,
    )
    y_test = experiment.y_test
    w_test = experiment.treatment_test
    scores = experiment.scores

    curves = {
        name: cumulative_uplift_curve(y_test, w_test, score)
        for name, score in scores.items()
    }
    budget_table = budget_policy_table(y_test, w_test, scores)
    metric_table = build_metric_table(y_test, w_test, scores, curves)
    champion = choose_champion(metric_table)
    comparison_table = build_business_comparison(budget_table, champion)
    decile_table = uplift_by_quantile(y_test, w_test, scores[champion], n_bins=10)
    test_summary = summarize_test_set(y_test, w_test)
    timing_table = experiment.timing_table

    figure_path = ROOT / (
        args.figure_path
        or (
            "reports/generated/figures/criteo_uplift_curves.png"
            if args.outcome == "visit"
            else f"reports/generated/figures/criteo_{args.outcome}_uplift_curves.png"
        )
    )
    plot_uplift_curves(curves, figure_path)

    report_path = ROOT / (
        args.report_path
        or (
            "reports/generated/criteo_experiment.md"
            if args.outcome == "visit"
            else f"reports/generated/criteo_{args.outcome}_experiment.md"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            dataset=dataset,
            train_size=experiment.train_size,
            test_size=experiment.test_size,
            test_summary=test_summary,
            metric_table=metric_table,
            budget_table=budget_table,
            comparison_table=comparison_table,
            decile_table=decile_table,
            timing_table=timing_table,
            champion=champion,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )

    print("Criteo experiment complete.")
    print(f"Champion by benchmark AUUC: {champion}")
    print(f"Report: {report_path}")
    print(metric_table.to_string(index=False))


def parse_policies(value: str | None) -> list[str] | None:
    if value is None:
        return None
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        raise ValueError("policies must not be empty.")
    if "response_model" not in policies:
        policies.insert(0, "response_model")
    return policies
def build_metric_table(
    y: pd.Series,
    treatment: pd.Series,
    scores: dict[str, np.ndarray],
    curves: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for policy, score in scores.items():
        rows.append(
            {
                "policy": policy,
                "benchmark_relative_auuc": separate_relative_auuc(y, treatment, score),
                "business_auuc": auuc(curves[policy]),
                "qini_coefficient": qini_coefficient(curves[policy]),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_relative_auuc", ascending=False)


def choose_champion(metric_table: pd.DataFrame) -> str:
    candidates = metric_table[metric_table["policy"].isin(UPLIFT_POLICIES)].dropna(
        subset=["benchmark_relative_auuc"]
    )
    if candidates.empty:
        raise RuntimeError("Cannot select a champion because benchmark AUUC could not be calculated.")
    return str(candidates.iloc[0]["policy"])


def build_business_comparison(budget_table: pd.DataFrame, champion: str) -> pd.DataFrame:
    pivot = budget_table.pivot(
        index="budget_pct", columns="policy", values="incremental_outcome"
    ).reset_index()
    comparison = pivot[
        ["budget_pct", champion, "response_model", "random"]
    ].rename(columns={champion: "uplift_champion"})
    comparison["gain_vs_response"] = (
        comparison["uplift_champion"] - comparison["response_model"]
    )
    comparison["gain_vs_random"] = comparison["uplift_champion"] - comparison["random"]
    comparison["ratio_vs_response"] = np.where(
        comparison["response_model"] != 0,
        comparison["uplift_champion"] / comparison["response_model"],
        np.nan,
    )
    return comparison


def summarize_test_set(y: pd.Series, treatment: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.to_numpy(), "treatment": treatment.to_numpy()})
    summary = (
        frame.groupby("treatment", observed=True)
        .agg(n=("y", "size"), outcome_rate=("y", "mean"), outcomes=("y", "sum"))
        .reset_index()
    )
    return summary


def plot_uplift_curves(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "random": "#7a8793",
        "response_model": "#d97706",
        "s_learner": "#147d64",
        "t_learner": "#2563a6",
        "x_learner": "#b43c59",
        "cvt": "#6b5ca5",
        "transformed_outcome": "#8a6d1d",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for policy, curve in curves.items():
        axis.plot(
            curve["fraction"] * 100,
            curve["incremental_outcome"],
            label=policy,
            color=colors.get(policy),
            linewidth=2 if policy != "random" else 1.5,
        )
    axis.set_title("Criteo: cumulative incremental outcome by budget")
    axis.set_xlabel("Targeted population (%)")
    axis.set_ylabel("Estimated incremental outcome")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_report(
    args: argparse.Namespace,
    dataset: CriteoDataset,
    train_size: int,
    test_size: int,
    test_summary: pd.DataFrame,
    metric_table: pd.DataFrame,
    budget_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    decile_table: pd.DataFrame,
    timing_table: pd.DataFrame,
    champion: str,
    figure_path: Path,
) -> str:
    rates = test_summary.set_index("treatment")["outcome_rate"]
    ate = float(rates.loc[1] - rates.loc[0])
    figure_relative = Path("figures") / figure_path.name

    return f"""# Criteo Uplift Experiment: {args.outcome}

## Setup

- Sample: `{args.sample_path}`
- Rows used: `{len(dataset.X):,}`
- Outcome: `{args.outcome}`
- Train/test: `{train_size:,}` / `{test_size:,}`
- Random seed: `{args.random_state}`
- Features: `f0` through `f11`; exclude `exposure` to avoid post-treatment leakage

## Test-Set Checks

{dataframe_to_markdown(test_summary)}

Observed ATE on the test set: `{ate:.6f}`.

## Ranking Comparison

{dataframe_to_markdown(metric_table)}

`benchmark_relative_auuc` implements `auuc_sep_rel_prop1` from the official Criteo benchmark repository. `qini_coefficient` is the area between the cumulative-uplift curve and the random-targeting line. The champion among uplift models is **{champion}**.

![Cumulative uplift curve]({figure_relative.as_posix()})

## Comparison by Budget

{dataframe_to_markdown(budget_table)}

## Champion versus Baselines

{dataframe_to_markdown(comparison_table)}

`gain_vs_response` is the additional incremental outcome from `{champion}` compared with response targeting at the same budget. This is an offline estimate on the test set, not a validated production impact.

## Champion Uplift by Decile

{dataframe_to_markdown(decile_table)}

## Training Time

{dataframe_to_markdown(timing_table)}

## Interpreting the Results

- Prioritize `benchmark_relative_auuc` for comparison with the Criteo paper.
- Use `gain_vs_response` and `gain_vs_random` to interpret budget-level impact.
- `cvt` uses Class Variable Transformation with inverse-propensity weights to balance the 85% treatment rate.
- `transformed_outcome` is the Modified Outcome Method: Ridge regression of `Y(T-e)/(e(1-e))` on standardized features.
- Do not use outcome accuracy or ROC-AUC as the primary conclusion because these metrics do not measure treatment-effect ranking quality.
- Repeat the analysis across multiple random seeds or use bootstrap confidence intervals before finalizing a model.
"""


if __name__ == "__main__":
    main()
