from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, subsample_criteo
from src.evaluation.uplift import (
    exact_qini_curve,
    exact_uplift_curve,
    separate_relative_auuc,
)
from src.models.base import make_classifier
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate S/T-learners and uplift curves against scikit-uplift."
    )
    parser.add_argument(
        "--sample-path", default="data/processed/criteo_sample_500k.parquet"
    )
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report-path", default="reports/scikit_uplift_validation.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import sklift
        from sklift.metrics import (
            qini_auc_score,
            qini_curve,
            uplift_auc_score,
            uplift_curve,
        )
        from sklift.models import SoloModel, TwoModels
    except ImportError as exc:
        raise RuntimeError(
            "scikit-uplift is missing. Run: python -m pip install -r requirements.txt"
        ) from exc

    dataset = load_criteo(ROOT / args.sample_path, outcome=args.outcome)
    dataset = subsample_criteo(dataset, args.max_rows, args.random_state)
    strata = dataset.treatment.astype(str) + "_" + dataset.y.astype(str)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        dataset.X,
        dataset.y,
        dataset.treatment,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=strata,
    )

    model_pairs = {
        "s_learner": (
            SLearner(model=make_classifier(args.random_state)),
            SoloModel(
                estimator=make_classifier(args.random_state),
                method="dummy",
            ),
        ),
        "t_learner": (
            TLearner(
                treated_model=make_classifier(args.random_state),
                control_model=make_classifier(args.random_state + 1),
            ),
            TwoModels(
                estimator_trmnt=make_classifier(args.random_state),
                estimator_ctrl=make_classifier(args.random_state + 1),
                method="vanilla",
            ),
        ),
    }

    predictions: dict[tuple[str, str], np.ndarray] = {}
    timings: dict[tuple[str, str], float] = {}
    for model_name, (local_model, reference_model) in model_pairs.items():
        print(f"Fitting the project's {model_name}...")
        started = perf_counter()
        local_model.fit(
            X_train,
            y_train,
            w_train,
            random_state=args.random_state,
        )
        timings[(model_name, "local")] = perf_counter() - started
        predictions[(model_name, "local")] = np.asarray(
            local_model.predict_uplift(X_test), dtype=float
        )

        print(f"Fitting scikit-uplift's {model_name}...")
        started = perf_counter()
        reference_model.fit(X_train, y_train, w_train)
        timings[(model_name, "scikit-uplift")] = perf_counter() - started
        predictions[(model_name, "scikit-uplift")] = np.asarray(
            reference_model.predict(X_test), dtype=float
        )

    agreement_rows = []
    metric_rows = []
    curve_rows = []
    for model_name in model_pairs:
        local_score = predictions[(model_name, "local")]
        reference_score = predictions[(model_name, "scikit-uplift")]
        agreement_rows.append(
            {
                "model": model_name,
                "pearson": _correlation(local_score, reference_score),
                "spearman": _rank_correlation(local_score, reference_score),
                "mean_abs_score_diff": float(
                    np.mean(np.abs(local_score - reference_score))
                ),
                "max_abs_score_diff": float(
                    np.max(np.abs(local_score - reference_score))
                ),
            }
        )

        for implementation in ["local", "scikit-uplift"]:
            score = predictions[(model_name, implementation)]
            metric_rows.append(
                {
                    "model": model_name,
                    "implementation": implementation,
                    "benchmark_relative_auuc": separate_relative_auuc(
                        y_test, w_test, score
                    ),
                    "sklift_uplift_auc": uplift_auc_score(y_test, score, w_test),
                    "sklift_qini_auc": qini_auc_score(y_test, score, w_test),
                    "fit_seconds": timings[(model_name, implementation)],
                }
            )

            local_uplift = exact_uplift_curve(y_test, w_test, score)
            local_qini = exact_qini_curve(y_test, w_test, score)
            reference_uplift_x, reference_uplift_y = uplift_curve(
                y_test, score, w_test
            )
            reference_qini_x, reference_qini_y = qini_curve(
                y_test, score, w_test
            )
            curve_rows.append(
                {
                    "model": model_name,
                    "score_source": implementation,
                    "same_uplift_x": np.array_equal(
                        local_uplift["n_targeted"].to_numpy(), reference_uplift_x
                    ),
                    "max_abs_uplift_diff": _max_abs_difference(
                        local_uplift["incremental_outcome"], reference_uplift_y
                    ),
                    "same_qini_x": np.array_equal(
                        local_qini["n_targeted"].to_numpy(), reference_qini_x
                    ),
                    "max_abs_qini_diff": _max_abs_difference(
                        local_qini["qini_value"], reference_qini_y
                    ),
                }
            )

    agreement_table = pd.DataFrame(agreement_rows)
    metric_table = pd.DataFrame(metric_rows)
    curve_table = pd.DataFrame(curve_rows)
    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            n_rows=len(dataset.X),
            train_size=len(X_train),
            test_size=len(X_test),
            sklift_version=sklift.__version__,
            agreement_table=agreement_table,
            metric_table=metric_table,
            curve_table=curve_table,
        ),
        encoding="utf-8",
    )

    print(f"Report written to: {report_path}")
    print(agreement_table.to_string(index=False))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return _correlation(left_rank, right_rank)


def _max_abs_difference(left, right) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    return float(np.max(np.abs(left_arr - right_arr)))


def build_report(
    args: argparse.Namespace,
    n_rows: int,
    train_size: int,
    test_size: int,
    sklift_version: str,
    agreement_table: pd.DataFrame,
    metric_table: pd.DataFrame,
    curve_table: pd.DataFrame,
) -> str:
    predictions_match = bool(
        (agreement_table["spearman"] > 0.999999).all()
        and (agreement_table["max_abs_score_diff"] < 1e-10).all()
    )
    curves_match = bool(
        curve_table["same_uplift_x"].all()
        and curve_table["same_qini_x"].all()
        and (curve_table["max_abs_uplift_diff"] < 1e-10).all()
        and (curve_table["max_abs_qini_diff"] < 1e-10).all()
    )
    prediction_conclusion = (
        "scores and rankings match" if predictions_match else "does not match completely"
    )
    curve_conclusion = (
        "match point by point" if curves_match else "have discrepancies that require investigation"
    )

    return f"""# Validation against scikit-uplift

## Setup

- Data: `{args.sample_path}` ({n_rows:,} rows after limiting).
- Outcome: `{args.outcome}`.
- Train/test: {train_size:,} / {test_size:,}.
- Random seed: {args.random_state}.
- Base learner: the same LightGBM configuration for both implementations.
- Versions: scikit-uplift `{sklift_version}`, scikit-learn `{sklearn_version}`.

`SLearner` is compared with `SoloModel(method="dummy")`; `TLearner` is compared
with `TwoModels(method="vanilla")`. All models use the same train/test split.

## Prediction Agreement

{dataframe_to_markdown(agreement_table)}

Prediction result: **{prediction_conclusion}**.

## Evaluation-Metric Agreement

{dataframe_to_markdown(metric_table)}

`benchmark_relative_auuc` follows the Criteo benchmark. The two `sklift_*`
columns are scores normalized according to scikit-uplift's own definitions, so
values should not be compared directly across columns. The relevant check is
that the local implementation and library return the same result for the same ranking.

## Curve Agreement

{dataframe_to_markdown(curve_table)}

The project's exact uplift/Qini curves **{curve_conclusion}** with scikit-uplift,
including the handling of tied scores.

## Conclusion

This comparison validates two independent layers: the S/T-learner implementations
and the evaluation-curve formulas. The result does not establish business value;
the champion decision still depends on stability across multiple seeds, bootstrap
analysis, and gain at the deployment budget.
"""


if __name__ == "__main__":
    main()
