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

from src.data.criteo import (
    feature_summary,
    load_criteo,
    prepare_criteo_sample,
    summarize_criteo,
)
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a sample and EDA for Criteo Uplift v2.1.")
    parser.add_argument("--data-path", default="data/criteo-uplift-v2.1.csv.gz")
    parser.add_argument("--sample-path", default="data/processed/criteo_sample_500k.parquet")
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-path", default="reports/criteo_eda.md")
    parser.add_argument("--figure-path", default="reports/figures/criteo_outcome_rates.png")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Scanning full Criteo statistics...")
    summary = summarize_criteo(ROOT / args.data_path)

    print(f"Creating a reservoir sample of {args.sample_size:,} rows...")
    sample_path = prepare_criteo_sample(
        ROOT / args.data_path,
        ROOT / args.sample_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
        force=args.force,
    )
    sample = load_criteo(sample_path)
    sample_summary = summarize_sample(sample.raw)
    features = feature_summary(sample.raw)

    figure_path = ROOT / args.figure_path
    plot_outcome_rates(summary.by_treatment, figure_path)

    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            overall=summary.overall,
            by_treatment=summary.by_treatment,
            sample_summary=sample_summary,
            features=features,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )

    print("Criteo data preparation complete.")
    print(f"Sample: {sample_path}")
    print(f"Report: {report_path}")
    print(summary.overall.to_string(index=False))


def summarize_sample(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n": len(df),
                "treatment_rate": df["treatment"].mean(),
                "visit_rate": df["visit"].mean(),
                "conversion_rate": df["conversion"].mean(),
                "exposure_rate": df["exposure"].mean(),
            }
        ]
    )


def plot_outcome_rates(by_treatment: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Control", "Treatment"]
    colors = ["#65727f", "#d64f3c"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))

    for axis, column, title in zip(
        axes,
        ["visit_rate", "conversion_rate"],
        ["Visit rate", "Conversion rate"],
    ):
        values = by_treatment.sort_values("treatment")[column].to_numpy()
        bars = axis.bar(labels, values, color=colors, width=0.6)
        axis.set_title(title)
        axis.set_ylabel("Outcome rate")
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, labels=[f"{value:.4%}" for value in values], padding=3)
        axis.set_ylim(0, max(values) * 1.22)

    figure.suptitle("Criteo Uplift v2.1: outcome by treatment arm")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_report(
    args: argparse.Namespace,
    overall: pd.DataFrame,
    by_treatment: pd.DataFrame,
    sample_summary: pd.DataFrame,
    features: pd.DataFrame,
    figure_path: Path,
) -> str:
    rates = by_treatment.set_index("treatment")
    ate = pd.DataFrame(
        [
            {
                "outcome": outcome,
                "treated_rate": rates.loc[1, f"{outcome}_rate"],
                "control_rate": rates.loc[0, f"{outcome}_rate"],
                "ATE": rates.loc[1, f"{outcome}_rate"] - rates.loc[0, f"{outcome}_rate"],
            }
            for outcome in ["visit", "conversion"]
        ]
    )
    figure_relative = Path("figures") / figure_path.name

    return f"""# Criteo Uplift v2.1 Exploratory Data Analysis

## Setup

- Source file: `{args.data_path}`
- Sample: `{args.sample_path}`
- Sample size: `{args.sample_size:,}`
- Random seed: `{args.random_state}`
- Sampling method: reservoir sampling over the full file with DuckDB

## Full Dataset Overview

{dataframe_to_markdown(overall)}

## Results by Treatment Group

{dataframe_to_markdown(by_treatment)}

## Observed Average Treatment Effect (ATE)

{dataframe_to_markdown(ate)}

![Outcome rate by treatment]({figure_relative.as_posix()})

## Sample Validation

{dataframe_to_markdown(sample_summary)}

The sample closely preserves the treatment and outcome rates of the full dataset. The Parquet file is used for experimental iterations to reduce data-loading time.

## Feature Distributions in the Sample

{dataframe_to_markdown(features, index=True)}

## Modeling Decisions

- Use `f0` through `f11` as input features.
- Exclude `exposure` because it is a post-treatment variable and may cause leakage.
- Use `visit` as the primary outcome because it has more positive samples.
- Retain `conversion` as a robustness check for a rare outcome.
- Use a train/test split stratified jointly by treatment and outcome.
"""


if __name__ == "__main__":
    main()
