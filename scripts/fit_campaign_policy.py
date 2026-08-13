# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import CRITEO_FEATURE_COLUMNS, load_criteo
from src.experiments.honest_uplift import _model_seed
from src.models.registry import select_model_factories
from src.serving.campaign_policy import (
    CampaignPolicy,
    measured_budgets_from_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the locked champion once and save it together with what the "
            "confirmatory test measured it to do, so a target list can be "
            "produced without refitting and can carry a number with it."
        )
    )
    parser.add_argument(
        "--fit-path",
        default="data/processed/criteo_audit_1m.parquet",
        help="The sample the confirmatory test refit the policy on.",
    )
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--model", default="s_learner")
    parser.add_argument(
        "--random-state",
        type=int,
        default=20260730,
        help=(
            "Must match the confirmatory run. The per-model seed is derived "
            "from it the same way, so this reproduces the evaluated model "
            "rather than a similar one."
        ),
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--policy-values-path",
        default="outputs/tables/confirmatory_visit_test.csv",
    )
    parser.add_argument(
        "--contrast-path",
        default="outputs/tables/confirmatory_visit_contrasts.csv",
    )
    parser.add_argument(
        "--measured-sample",
        default="data/processed/criteo_confirm_4m.parquet",
        help="Recorded only as provenance for the rates being attached.",
    )
    parser.add_argument("--output-path", default="artifacts/campaign_policy.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_values = pd.read_csv(ROOT / args.policy_values_path)
    contrasts = pd.read_csv(ROOT / args.contrast_path)
    budgets = measured_budgets_from_tables(
        policy_values,
        contrasts,
        model_name=args.model,
        confidence_level=args.confidence_level,
    )

    dataset = load_criteo(ROOT / args.fit_path, outcome=args.outcome)
    factory = select_model_factories([args.model])[args.model]
    seed = _model_seed(args.random_state, args.model)
    print(f"Fitting {args.model} on {len(dataset.X):,} rows with seed {seed}...")
    model = factory()
    model.fit(dataset.X, dataset.y, dataset.treatment, random_state=seed)

    measured_rows = int(policy_values["n_targeted"].max() / (
        policy_values["budget_pct"].max() / 100.0
    ))
    policy = CampaignPolicy(
        model=model,
        model_name=args.model,
        outcome=args.outcome,
        feature_columns=list(CRITEO_FEATURE_COLUMNS),
        propensity=float(dataset.treatment.mean()),
        confidence_level=args.confidence_level,
        fit_sample=Path(args.fit_path).as_posix(),
        fit_rows=len(dataset.X),
        measured_sample=Path(args.measured_sample).as_posix(),
        measured_rows=measured_rows,
        fitted_at=datetime.now(UTC).strftime("%Y-%m-%d"),
        model_seed=seed,
        budgets=budgets,
    )
    saved = policy.save(ROOT / args.output_path)

    print(f"\nSaved {saved.relative_to(ROOT).as_posix()}")
    print(f"  model      {policy.model_name} ({policy.outcome})")
    print(f"  fit on     {policy.fit_sample} ({policy.fit_rows:,} rows)")
    print(f"  measured   {policy.measured_sample} ({policy.measured_rows:,} rows)")
    print("  budgets    " + ", ".join(
        f"{value:g}%" for value in policy.measured_budgets()
    ))


if __name__ == "__main__":
    main()
