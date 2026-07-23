from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

from src.data.hillstrom import load_hillstrom, make_binary_hillstrom, summarize_by_segment
from src.evaluation.uplift import auuc, budget_policy_table, cumulative_uplift_curve, uplift_by_quantile
from src.models.response_model import ResponseModel
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an uplift warm-up experiment on Hillstrom.")
    parser.add_argument("--data-path", default="data/hillstrom_email.csv")
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--treatment-segment", default="Mens E-Mail")
    parser.add_argument("--control-segment", default="No E-Mail")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-path", default="reports/hillstrom_warmup.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    raw = load_hillstrom(args.data_path)
    segment_summary = summarize_by_segment(raw)
    dataset = make_binary_hillstrom(
        raw,
        treatment_segment=args.treatment_segment,
        control_segment=args.control_segment,
        outcome=args.outcome,
    )

    strata = dataset.treatment.astype(str) + "_" + dataset.y.astype(str)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        dataset.X,
        dataset.y,
        dataset.treatment,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=strata,
    )

    response_model = ResponseModel().fit(X_train, y_train, w_train, random_state=args.random_state)
    s_learner = SLearner().fit(X_train, y_train, w_train, random_state=args.random_state)
    t_learner = TLearner().fit(X_train, y_train, w_train, random_state=args.random_state)
    x_learner = XLearner().fit(X_train, y_train, w_train, random_state=args.random_state)

    scores = {
        "random": rng.random(len(X_test)),
        "response_model": response_model.predict_score(X_test),
        "s_learner": s_learner.predict_uplift(X_test),
        "t_learner": t_learner.predict_uplift(X_test),
        "x_learner": x_learner.predict_uplift(X_test),
    }

    budget_table = budget_policy_table(y_test, w_test, scores)
    auuc_table = pd.DataFrame(
        [
            {
                "policy": name,
                "auuc": auuc(cumulative_uplift_curve(y_test, w_test, score)),
            }
            for name, score in scores.items()
        ]
    ).sort_values("auuc", ascending=False)
    decile_table = uplift_by_quantile(y_test, w_test, scores["t_learner"], n_bins=10)

    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            args=args,
            segment_summary=segment_summary,
            budget_table=budget_table,
            auuc_table=auuc_table,
            decile_table=decile_table,
        ),
        encoding="utf-8",
    )

    print("Hillstrom warm-up complete.")
    print(f"Report: {report_path}")
    print("\nPolicy comparison by budget:")
    print(budget_table.to_string(index=False))
    print("\nAUUC table:")
    print(auuc_table.to_string(index=False))


def build_report(
    args: argparse.Namespace,
    segment_summary: pd.DataFrame,
    budget_table: pd.DataFrame,
    auuc_table: pd.DataFrame,
    decile_table: pd.DataFrame,
) -> str:
    best_rows = (
        budget_table.sort_values(["budget_pct", "incremental_outcome"], ascending=[True, False])
        .groupby("budget_pct", as_index=False)
        .head(1)
    )

    return f"""# Pipeline Validation on Hillstrom

## Setup

- Dataset: `Hillstrom Email Dataset`
- Treatment: `{args.treatment_segment}`
- Control: `{args.control_segment}`
- Outcome: `{args.outcome}`
- Test size: `{args.test_size}`
- Random seed: `{args.random_state}`

This warm-up validates the end-to-end workflow before moving to Criteo: load data → train response/uplift models → rank users → evaluate incremental outcomes by budget.

## Overview by Experimental Group

{dataframe_to_markdown(segment_summary, index=True)}

## Policy Comparison by Budget

{dataframe_to_markdown(budget_table, index=False)}

## Best Policy at Each Budget

{dataframe_to_markdown(best_rows, index=False)}

## Approximate AUUC

{dataframe_to_markdown(auuc_table, index=False)}

## T-Learner Uplift by Decile

{dataframe_to_markdown(decile_table, index=False)}

## Methodology Notes

- `response_model` learns outcome probability in the treated group, approximating traditional targeting.
- `s_learner` learns one outcome model with the treatment indicator as a feature.
- `t_learner` learns `P(Y|X,T=1)` and `P(Y|X,T=0)` separately, then takes the difference between predictions.
- `x_learner` first estimates outcome models, imputes treatment effects, and learns treatment effects in stage two.
- `incremental_outcome` is calculated for the top-k group as `(mean(Y|treated) - mean(Y|control)) * n_top_k`.

"""
if __name__ == "__main__":
    main()
