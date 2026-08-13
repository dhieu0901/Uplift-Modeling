# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import CriteoDataset, load_criteo, subsample_criteo
from src.evaluation.calibration import (
    summarize_uplift_calibration,
    uplift_calibration_table,
)
from src.evaluation.policy_value import (
    monetize_policy_table,
    target_by_expected_value,
    uplift_score_threshold,
)
from src.evaluation.uplift import incremental_outcome, separate_relative_auuc
from src.models.registry import (
    default_model_factories,
    rare_outcome_model_factories,
)
from src.models.uplift_calibration import UpliftIsotonicCalibrator
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate uplift scores on a calibration set and evaluate a holdout."
    )
    parser.add_argument(
        "--sample-path", default="data/processed/criteo_sample_2m.parquet"
    )
    parser.add_argument(
        "--outcome",
        default="conversion",
        choices=["visit", "conversion"],
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--models", default="undersampled_t_lr_k5")
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--undersampling-factors",
        default="5",
        help="Optional rare-outcome logistic T/CVT factors.",
    )
    parser.add_argument(
        "--undersampling-families",
        default="t",
        help="Comma-separated rare-outcome families: t,cvt.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--calibrator-bins", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outcome-value", type=float, default=100.0)
    parser.add_argument("--treatment-cost", type=float, default=5.0)
    parser.add_argument(
        "--report-path",
        default="outputs/conversion_uplift_calibration.md",
    )
    parser.add_argument(
        "--bins-path",
        default="outputs/tables/conversion_uplift_calibration_bins.csv",
    )
    parser.add_argument(
        "--figure-path",
        default="outputs/figures/conversion_uplift_calibration.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factories = default_model_factories(crossfit_folds=args.crossfit_folds)
    factors = parse_factors(args.undersampling_factors)
    families = parse_rare_families(args.undersampling_families)
    factories.update(
        rare_outcome_model_factories(factors, families=families)
    )
    model_names = parse_models(
        args.models,
        supported=set(factories) - {"response_model"},
    )
    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, args.random_state)
    splits = split_train_calibration_test(
        dataset,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
        random_state=args.random_state,
    )
    X_train, y_train, w_train = splits["train"]
    X_calibration, y_calibration, w_calibration = splits["calibration"]
    X_test, y_test, w_test = splits["test"]
    propensity = float(w_train.mean())

    calibration_frames = []
    summary_rows = []
    threshold_rows = []
    fit_rows = []
    for model_name in model_names:
        print(f"Training {model_name}...")
        model = factories[model_name]()
        started = perf_counter()
        model.fit(
            X_train,
            y_train,
            w_train,
            random_state=args.random_state,
        )
        model_fit_seconds = perf_counter() - started
        raw_calibration_score = np.asarray(
            model.predict_uplift(X_calibration), dtype=float
        )
        raw_test_score = np.asarray(model.predict_uplift(X_test), dtype=float)

        print(f"Calibrating {model_name}...")
        calibrator = UpliftIsotonicCalibrator(n_bins=args.calibrator_bins)
        started = perf_counter()
        calibrator.fit(
            raw_calibration_score,
            y_calibration,
            w_calibration,
            propensity=propensity,
        )
        calibration_fit_seconds = perf_counter() - started
        calibrated_test_score = calibrator.predict(raw_test_score)
        fit_rows.append(
            {
                "model": model_name,
                "model_fit_seconds": model_fit_seconds,
                "calibrator_fit_seconds": calibration_fit_seconds,
            }
        )

        for version, score in {
            "raw": raw_test_score,
            "calibrated": calibrated_test_score,
        }.items():
            table = uplift_calibration_table(
                y_test,
                w_test,
                score,
                n_bins=args.n_bins,
                binning_score=raw_test_score,
            )
            table.insert(0, "score_version", version)
            table.insert(0, "model", model_name)
            calibration_frames.append(table)

            summary = summarize_uplift_calibration(table)
            summary_rows.append(
                {
                    "model": model_name,
                    "score_version": version,
                    **summary,
                    "benchmark_relative_auuc": separate_relative_auuc(
                        y_test, w_test, score
                    ),
                    "score_mean": float(np.mean(score)),
                    "score_min": float(np.min(score)),
                    "score_max": float(np.max(score)),
                }
            )
            threshold_rows.append(
                evaluate_threshold_policy(
                    y_test,
                    w_test,
                    score,
                    model_name=model_name,
                    score_version=version,
                    outcome_value=args.outcome_value,
                    treatment_cost=args.treatment_cost,
                )
            )

    calibration_bins = pd.concat(calibration_frames, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    fit_table = pd.DataFrame(fit_rows)

    bins_path = ROOT / args.bins_path
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_bins.to_csv(bins_path, index=False, encoding="utf-8-sig")

    figure_path = ROOT / args.figure_path
    plot_calibration(calibration_bins, model_names, figure_path)

    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            dataset_size=len(dataset.X),
            split_sizes={name: len(values[0]) for name, values in splits.items()},
            propensity=propensity,
            summary_table=summary_table,
            threshold_table=threshold_table,
            fit_table=fit_table,
            calibration_bins=calibration_bins,
            figure_path=figure_path,
            bins_path=bins_path,
        ),
        encoding="utf-8",
    )
    print(f"Report written to: {report_path}")
    print(summary_table.to_string(index=False))


def parse_models(value: str, supported: set[str]) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("models must not be empty.")
    unknown = sorted(set(models) - supported)
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    return list(dict.fromkeys(models))


def parse_factors(value: str) -> tuple[float, ...]:
    try:
        return tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "undersampling-factors must be comma-separated numbers."
        ) from exc


def parse_rare_families(value: str) -> tuple[str, ...]:
    families = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not families:
        raise ValueError("undersampling-families must not be empty.")
    unknown = sorted(set(families) - {"t", "cvt"})
    if unknown:
        raise ValueError(
            "undersampling-families supports only: t,cvt; "
            f"received {unknown}"
        )
    return families


def split_train_calibration_test(
    dataset: CriteoDataset,
    train_fraction: float,
    calibration_fraction: float,
    random_state: int,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in the interval (0, 1).")
    if not 0.0 < calibration_fraction < 1.0 - train_fraction:
        raise ValueError(
            "calibration_fraction must be positive and smaller than the share "
            "remaining after training."
        )

    strata = dataset.treatment.astype(str) + "_" + dataset.y.astype(str)
    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        dataset.X,
        dataset.y,
        dataset.treatment,
        train_size=train_fraction,
        random_state=random_state,
        stratify=strata,
    )
    calibration_share_of_temp = calibration_fraction / (1.0 - train_fraction)
    temp_strata = w_temp.astype(str) + "_" + y_temp.astype(str)
    X_cal, X_test, y_cal, y_test, w_cal, w_test = train_test_split(
        X_temp,
        y_temp,
        w_temp,
        train_size=calibration_share_of_temp,
        random_state=random_state + 1,
        stratify=temp_strata,
    )
    return {
        "train": (X_train, y_train, w_train),
        "calibration": (X_cal, y_cal, w_cal),
        "test": (X_test, y_test, w_test),
    }


def evaluate_threshold_policy(
    y: pd.Series,
    treatment: pd.Series,
    score: np.ndarray,
    model_name: str,
    score_version: str,
    outcome_value: float,
    treatment_cost: float,
) -> dict[str, float | int | str | bool]:
    target = target_by_expected_value(score, outcome_value, treatment_cost)
    n_targeted = int(target.sum())
    if n_targeted == 0:
        estimated_incremental = 0.0
    else:
        estimated_incremental = incremental_outcome(
            y.to_numpy()[target], treatment.to_numpy()[target]
        )
    base = pd.DataFrame(
        [
            {
                "policy": f"{model_name}_{score_version}",
                "budget_pct": 100.0 * n_targeted / len(score),
                "n_targeted": n_targeted,
                "incremental_outcome": estimated_incremental,
            }
        ]
    )
    monetized = monetize_policy_table(
        base,
        outcome_value=outcome_value,
        treatment_cost=treatment_cost,
    ).iloc[0]
    return {
        "model": model_name,
        "score_version": score_version,
        "score_threshold": uplift_score_threshold(outcome_value, treatment_cost),
        "target_rate_pct": monetized["budget_pct"],
        "n_targeted": n_targeted,
        "incremental_outcome": estimated_incremental,
        "net_value": monetized["net_value"],
        "profitable": bool(monetized["profitable"]),
    }


def plot_calibration(
    calibration_bins: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(6 * len(model_names), 5),
        squeeze=False,
    )
    colors = {"raw": "#d97706", "calibrated": "#147d64"}
    markers = {"raw": "o", "calibrated": "s"}
    for axis, model_name in zip(axes[0], model_names, strict=True):
        model_data = calibration_bins[calibration_bins["model"] == model_name]
        low = min(
            float(model_data["predicted_uplift"].min()),
            float(model_data["ci_lower"].min()),
            0.0,
        )
        high = max(
            float(model_data["predicted_uplift"].max()),
            float(model_data["ci_upper"].max()),
            0.0,
        )
        padding = max((high - low) * 0.08, 0.002)
        limits = (low - padding, high + padding)
        axis.plot(limits, limits, color="#5b6470", linestyle="--", label="ideal")
        for version in ["raw", "calibrated"]:
            data = model_data[model_data["score_version"] == version]
            y_error = np.vstack(
                [
                    data["observed_uplift"] - data["ci_lower"],
                    data["ci_upper"] - data["observed_uplift"],
                ]
            )
            axis.errorbar(
                data["predicted_uplift"],
                data["observed_uplift"],
                yerr=y_error,
                marker=markers[version],
                color=colors[version],
                linewidth=1.5,
                capsize=3,
                label=version,
            )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(model_name)
        axis.set_xlabel("Predicted uplift")
        axis.set_ylabel("Observed uplift")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Uplift calibration on an independent holdout")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_report(
    args: argparse.Namespace,
    dataset_size: int,
    split_sizes: dict[str, int],
    propensity: float,
    summary_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    fit_table: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    figure_path: Path,
    bins_path: Path,
) -> str:
    calibrated_bins = calibration_bins[
        calibration_bins["score_version"] == "calibrated"
    ][
        [
            "model",
            "bin",
            "n",
            "predicted_uplift",
            "observed_uplift",
            "ci_lower",
            "ci_upper",
        ]
    ]
    # `pivot` is deliberate: one row per (model, score_version) already exists,
    # so `pivot_table` would silently average duplicates instead of failing.
    comparison = summary_table.pivot(  # noqa: PD010
        index="model", columns="score_version", values="weighted_mae"
    ).reset_index()
    comparison["mae_change"] = comparison["calibrated"] - comparison["raw"]
    improved_models = comparison.loc[comparison["mae_change"] < 0, "model"].tolist()
    improvement_text = (
        ", ".join(f"`{model}`" for model in improved_models)
        if improved_models
        else "no models"
    )
    figure_relative = Path("figures") / figure_path.name
    bins_relative = bins_path.relative_to(ROOT).as_posix()
    break_even_threshold = uplift_score_threshold(
        args.outcome_value,
        args.treatment_cost,
    )
    split_summary = " / ".join(
        f"{split_sizes[name]:,}" for name in ("train", "calibration", "test")
    )

    return f"""# Uplift-Score Calibration on an Independent Holdout

## Setup

- Data: `{args.sample_path}` ({dataset_size:,} rows), outcome `{args.outcome}`.
- Train/calibration/test rows: {split_summary}.
- Treatment propensity estimated from training data: `{propensity:.6f}`.
- Model: `{args.models}`; random seed `{args.random_state}`.
- The isotonic calibrator is fitted only on the calibration set using transformed outcomes.
- The calibrator is fitted on `{args.calibrator_bins}` weighted quantile groups
  to reduce pseudo-outcome noise.
- Calibration metrics and threshold policies are evaluated only on the untouched test holdout.

## Calibration Results Summary

{dataframe_to_markdown(summary_table)}

Ideal calibration has an intercept near `0`, a slope near `1`, and errors near
`0`. Based on weighted MAE on the holdout, calibration improves: {improvement_text}.
Isotonic mapping is monotonic and therefore preserves ordering in principle;
AUUC may change slightly because multiple scores can be mapped to the same value.

![Calibration plot]({figure_relative.as_posix()})

## Post-Calibration Groups

{dataframe_to_markdown(calibrated_bins)}

Bin 1 contains the group with the highest raw scores. The confidence interval is
a normal approximation for the difference between treatment and control rates
within each bin. All bins are stored at `{bins_relative}`.

## Break-Even Targeting at a Given Value Ratio

A calibrated score is on the outcome scale, so it can be compared against a
threshold instead of a rank. The threshold depends only on the ratio of what an
incremental {args.outcome} is worth to what a contact costs, which is why the
two enter as arguments: change the ratio and the section re-derives the rule.

- Value of one incremental {args.outcome}: `{args.outcome_value:.2f}`.
- Cost per targeting action: `{args.treatment_cost:.2f}`.
- Break-even uplift threshold: `{break_even_threshold:.6f}`.

{dataframe_to_markdown(threshold_table)}

Only `calibrated` rows can carry an absolute interpretation, because a raw score
is not on the probability scale. The fixed 5% budget remains the operating rule
for the online experiment, because a threshold rule moves the number of users
contacted with the score distribution while a budget rule does not.

## Runtime

{dataframe_to_markdown(fit_table)}

## Recommendations

- Use the calibration plot to validate score magnitude, not as a replacement
  for AUUC as a ranking metric.
- Do not select a threshold on the test holdout after reviewing its results.
- Lock the model, calibrator, and threshold before the randomized online experiment.
"""


if __name__ == "__main__":
    main()
