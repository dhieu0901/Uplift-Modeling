from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.evaluation.bootstrap import BootstrapUpliftResult, bootstrap_uplift_uncertainty
from src.evaluation.uplift import budget_policy_table, separate_relative_auuc
from src.experiments.criteo import CriteoExperimentResult, run_criteo_experiment
from src.reporting import dataframe_to_markdown


UPLIFT_POLICIES = [
    "s_learner",
    "t_learner",
    "x_learner",
    "cvt",
    "transformed_outcome",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uplift-model stability across multiple seeds and bootstrap samples."
    )
    parser.add_argument("--sample-path", default="data/processed/criteo_sample_500k.parquet")
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated challenger list; response and random are added automatically.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=730)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--business-results-path", default=None)
    parser.add_argument("--figure-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    policies = parse_policies(args.policies)
    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, seeds[0])

    metric_frames = []
    budget_frames = []
    timing_frames = []
    reference_result: CriteoExperimentResult | None = None

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        result = run_criteo_experiment(
            dataset,
            test_fraction=args.test_size,
            random_state=seed,
            progress=lambda name, current_seed=seed: print(
                f"Seed {current_seed}: training {name}..."
            ),
            policies=policies,
        )
        if reference_result is None:
            reference_result = result

        metric_frames.append(build_seed_metrics(result, seed))
        budget = budget_policy_table(
            result.y_test,
            result.treatment_test,
            result.scores,
        )
        budget.insert(0, "seed", seed)
        budget_frames.append(budget)

        timing = result.timing_table.copy()
        timing.insert(0, "seed", seed)
        timing_frames.append(timing)

    if reference_result is None:
        raise RuntimeError("No experiment results are available for evaluation.")

    metrics_by_seed = pd.concat(metric_frames, ignore_index=True)
    budgets_by_seed = pd.concat(budget_frames, ignore_index=True)
    timings_by_seed = pd.concat(timing_frames, ignore_index=True)
    metric_summary = summarize_metrics(metrics_by_seed)
    policy_budget_summary = summarize_policy_budgets(budgets_by_seed)
    champion = choose_stable_champion(metric_summary)
    seed_comparison = summarize_business_gain(budgets_by_seed, champion)
    timing_summary = summarize_timings(timings_by_seed)

    print(
        f"\nBootstrapping reference seed {seeds[0]} with "
        f"{args.bootstrap_iterations} iterations..."
    )
    bootstrap = bootstrap_uplift_uncertainty(
        reference_result.y_test,
        reference_result.treatment_test,
        reference_result.scores,
        champion=champion,
        n_bootstraps=args.bootstrap_iterations,
        random_state=args.bootstrap_seed,
    )

    figure_path = ROOT / (
        args.figure_path
        or (
            "reports/generated/figures/criteo_stability.png"
            if args.outcome == "visit"
            else f"reports/generated/figures/criteo_{args.outcome}_stability.png"
        )
    )
    plot_stability(metric_summary, budgets_by_seed, figure_path, args.outcome)

    results_path = ROOT / (
        args.results_path
        or (
            "reports/generated/criteo_stability_by_seed.csv"
            if args.outcome == "visit"
            else f"reports/generated/criteo_{args.outcome}_stability_by_seed.csv"
        )
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_seed.to_csv(results_path, index=False, encoding="utf-8-sig")

    business_results_path = ROOT / (
        args.business_results_path
        or (
            "reports/generated/criteo_business_by_policy.csv"
            if args.outcome == "visit"
            else f"reports/generated/criteo_{args.outcome}_business_by_policy.csv"
        )
    )
    business_results_path.parent.mkdir(parents=True, exist_ok=True)
    policy_budget_summary.to_csv(
        business_results_path, index=False, encoding="utf-8-sig"
    )

    report_path = ROOT / (
        args.report_path
        or (
            "reports/generated/criteo_stability.md"
            if args.outcome == "visit"
            else f"reports/generated/criteo_{args.outcome}_stability.md"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            seeds=seeds,
            n_rows=len(dataset.X),
            metrics_by_seed=metrics_by_seed,
            metric_summary=metric_summary,
            seed_comparison=seed_comparison,
            timing_summary=timing_summary,
            bootstrap=bootstrap,
            champion=champion,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )

    print("Stability analysis complete.")
    print(f"Stable uplift champion: {champion}")
    print(f"Report: {report_path}")
    print(f"Business summary: {business_results_path}")
    print(metric_summary.to_string(index=False))


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be a list of integers, for example: 42,123,2026") from exc
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required to evaluate stability.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("The seed list must not contain duplicates.")
    return seeds


def parse_policies(value: str | None) -> list[str] | None:
    if value is None:
        return None
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        raise ValueError("policies must not be empty.")
    if "response_model" not in policies:
        policies.insert(0, "response_model")
    return policies


def build_seed_metrics(result: CriteoExperimentResult, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "policy": policy,
                "benchmark_relative_auuc": separate_relative_auuc(
                    result.y_test,
                    result.treatment_test,
                    score,
                ),
            }
            for policy, score in result.scores.items()
        ]
    )


def summarize_metrics(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_by_seed.groupby("policy", as_index=False)
        .agg(
            mean_auuc=("benchmark_relative_auuc", "mean"),
            std_auuc=("benchmark_relative_auuc", "std"),
            min_auuc=("benchmark_relative_auuc", "min"),
            max_auuc=("benchmark_relative_auuc", "max"),
        )
        .sort_values("mean_auuc", ascending=False)
    )


def choose_stable_champion(metric_summary: pd.DataFrame) -> str:
    candidates = metric_summary[metric_summary["policy"].isin(UPLIFT_POLICIES)]
    if candidates.empty:
        raise RuntimeError("No uplift policy was found for champion selection.")
    return str(candidates.iloc[0]["policy"])


def summarize_business_gain(budgets_by_seed: pd.DataFrame, champion: str) -> pd.DataFrame:
    pivot = budgets_by_seed.pivot(
        index=["seed", "budget_pct"],
        columns="policy",
        values="incremental_outcome",
    ).reset_index()
    pivot["gain_vs_response"] = pivot[champion] - pivot["response_model"]
    pivot["gain_vs_random"] = pivot[champion] - pivot["random"]

    rows = []
    for budget_pct, group in pivot.groupby("budget_pct", observed=True):
        rows.append(
            {
                "budget_pct": budget_pct,
                "champion_mean": group[champion].mean(),
                "response_mean": group["response_model"].mean(),
                "gain_vs_response_mean": group["gain_vs_response"].mean(),
                "gain_vs_response_std": group["gain_vs_response"].std(),
                "gain_vs_response_min": group["gain_vs_response"].min(),
                "gain_vs_response_max": group["gain_vs_response"].max(),
                "positive_seed_rate": (group["gain_vs_response"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy_budgets(budgets_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy/budget results into reusable cost-benefit inputs."""
    return (
        budgets_by_seed.groupby(["policy", "budget_pct"], as_index=False)
        .agg(
            n_targeted=("n_targeted", "first"),
            incremental_outcome=("incremental_outcome", "mean"),
            incremental_outcome_std=("incremental_outcome", "std"),
            incremental_outcome_per_1k=("incremental_outcome_per_1k", "mean"),
        )
        .sort_values(["policy", "budget_pct"])
    )


def summarize_timings(timings_by_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        timings_by_seed.groupby("model", as_index=False)
        .agg(
            mean_fit_seconds=("fit_seconds", "mean"),
            std_fit_seconds=("fit_seconds", "std"),
        )
        .sort_values("mean_fit_seconds")
    )


def plot_stability(
    metric_summary: pd.DataFrame,
    budgets_by_seed: pd.DataFrame,
    output_path: Path,
    outcome: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policies = metric_summary["policy"].tolist()
    colors = [
        "#147d64",
        "#2563a6",
        "#d97706",
        "#b43c59",
        "#6b5ca5",
        "#8a6d1d",
        "#7a8793",
    ]
    color_map = {policy: colors[index % len(colors)] for index, policy in enumerate(policies)}

    budget_10 = budgets_by_seed[budgets_by_seed["budget_pct"] == 10.0]
    budget_summary = (
        budget_10.groupby("policy", as_index=False)["incremental_outcome"]
        .agg(["mean", "std"])
        .reset_index()
        .set_index("policy")
        .loc[policies]
        .reset_index()
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(
        policies,
        metric_summary["mean_auuc"],
        yerr=metric_summary["std_auuc"],
        color=[color_map[policy] for policy in policies],
        capsize=4,
    )
    axes[0].set_title("Benchmark AUUC across seeds")
    axes[0].set_ylabel("Mean AUUC ± 1 SD")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        budget_summary["policy"],
        budget_summary["mean"],
        yerr=budget_summary["std"],
        color=[color_map[policy] for policy in budget_summary["policy"]],
        capsize=4,
    )
    axes[1].set_title(f"Incremental {outcome} at a 10% budget")
    axes[1].set_ylabel(f"Mean incremental {outcome} ± 1 SD")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("Criteo uplift-policy stability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_report(
    args: argparse.Namespace,
    seeds: list[int],
    n_rows: int,
    metrics_by_seed: pd.DataFrame,
    metric_summary: pd.DataFrame,
    seed_comparison: pd.DataFrame,
    timing_summary: pd.DataFrame,
    bootstrap: BootstrapUpliftResult,
    champion: str,
    figure_path: Path,
) -> str:
    seed_pivot = metrics_by_seed.pivot(
        index="seed", columns="policy", values="benchmark_relative_auuc"
    ).reset_index()
    uplift_seed_metrics = metrics_by_seed[metrics_by_seed["policy"].isin(UPLIFT_POLICIES)].copy()
    uplift_seed_metrics["rank"] = uplift_seed_metrics.groupby("seed")[
        "benchmark_relative_auuc"
    ].rank(ascending=False, method="min")
    winner_counts = (
        uplift_seed_metrics[uplift_seed_metrics["rank"] == 1]
        .groupby("policy", as_index=False)
        .agg(seed_wins=("seed", "count"))
        .sort_values("seed_wins", ascending=False)
    )
    figure_relative = Path("figures") / figure_path.name
    overall_leader = str(metric_summary.iloc[0]["policy"])
    overall_leader_mean = float(metric_summary.iloc[0]["mean_auuc"])
    champion_mean = float(
        metric_summary.loc[metric_summary["policy"] == champion, "mean_auuc"].iloc[0]
    )
    top_10 = seed_comparison.loc[seed_comparison["budget_pct"] == 10.0].iloc[0]
    bootstrap_top_10 = bootstrap.business_gains.loc[
        bootstrap.business_gains["budget_pct"] == 10.0
    ].iloc[0]
    bootstrap_champion_metric = bootstrap.policy_metrics.loc[
        bootstrap.policy_metrics["policy"] == champion
    ].iloc[0]
    if top_10["gain_vs_response_mean"] >= 0:
        budget_comparison = (
            f"exceeds the response model by an average of "
            f"{top_10['gain_vs_response_mean']:.2f} {args.outcome}"
        )
    else:
        budget_comparison = (
            f"is below the response model by an average of "
            f"{abs(top_10['gain_vs_response_mean']):.2f} {args.outcome}"
        )

    return f"""# Criteo Uplift-Model Stability: {args.outcome}

## Setup

- Sample: `{args.sample_path}`
- Rows used: `{n_rows:,}`
- Outcome: `{args.outcome}`
- Train/test seeds: `{", ".join(str(seed) for seed in seeds)}`
- Test size: `{args.test_size}`
- Bootstrap: `{args.bootstrap_iterations}` iterations, 95% confidence level
- Bootstrap seed: `{args.bootstrap_seed}`

## AUUC by Seed

{dataframe_to_markdown(seed_pivot)}

## Ranking-Stability Summary

{dataframe_to_markdown(metric_summary)}

The stable champion among uplift models by mean benchmark relative AUUC is **{champion}**. The leading policy including baselines is **{overall_leader}** (`{overall_leader_mean:.6f}`), while `{champion}` reaches `{champion_mean:.6f}`.

### Wins among Uplift Models

{dataframe_to_markdown(winner_counts)}

![Stability across seeds]({figure_relative.as_posix()})

## Business Performance across Seeds

{dataframe_to_markdown(seed_comparison)}

At a 10% budget, `{champion}` generates an average of `{top_10['champion_mean']:.2f}` incremental {args.outcome}; it {budget_comparison} across {len(seeds)} seeds. The share of seeds with positive gain is `{top_10['positive_seed_rate']:.0%}`.

## Bootstrap 95% CI on the Reference Seed

### Benchmark relative AUUC

{dataframe_to_markdown(bootstrap.policy_metrics)}

The `difference_vs_response` columns and corresponding confidence intervals are paired comparisons on the same bootstrap sample. If an interval contains zero, the full-range AUUC difference from the response model is not statistically conclusive.

For `{champion}`, the AUUC difference from the response model is `{bootstrap_champion_metric['difference_vs_response']:.6f}`, with a bootstrap 95% CI from `{bootstrap_champion_metric['difference_ci_lower']:.6f}` to `{bootstrap_champion_metric['difference_ci_upper']:.6f}`.

### Incremental Outcome and Policy Gain

{dataframe_to_markdown(bootstrap.business_gains)}

At a 10% budget, `{champion}` gains `{bootstrap_top_10['gain_vs_response']:.2f}` over the response model, with a bootstrap 95% CI from `{bootstrap_top_10['gain_vs_response_ci_lower']:.2f}` to `{bootstrap_top_10['gain_vs_response_ci_upper']:.2f}`.

## Training Time

{dataframe_to_markdown(timing_summary)}

## Cautious Conclusions

- Multi-seed analysis reflects variation from train/test splits and model random states.
- Bootstrap CIs reflect test-set sampling uncertainty for models trained on the reference seed.
- If the CI for `gain_vs_response` contains zero, there is insufficient evidence that the uplift policy outperforms the response model at that budget.
- Results remain an offline evaluation; final validation requires a randomized online experiment or a new campaign holdout.
"""


if __name__ == "__main__":
    main()
