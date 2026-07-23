from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simulated aggregate results for a dry run of online analysis."
    )
    parser.add_argument(
        "--arms-path", default="reports/tables/online_experiment_arms.csv"
    )
    parser.add_argument(
        "--output-path",
        default="data/online_experiment_synthetic_results.csv",
    )
    parser.add_argument("--missing-rate", type=float, default=0.002)
    parser.add_argument("--delivery-rate", type=float, default=0.98)
    parser.add_argument("--treatment-cost", type=float, default=5.0)
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("missing_rate must be in the interval [0, 1).")
    if not 0.0 <= args.delivery_rate <= 1.0:
        raise ValueError("delivery_rate must be in the interval [0, 1].")
    if args.treatment_cost < 0 or args.sample_scale <= 0:
        raise ValueError("treatment_cost must be nonnegative and sample_scale must be positive.")

    arms = pd.read_csv(ROOT / args.arms_path)
    required = {
        "arm",
        "target_rate_pct",
        "sample_size",
        "expected_visit_rate_offline",
    }
    missing = sorted(required - set(arms.columns))
    if missing:
        raise ValueError(f"Arms file is missing columns: {missing}")

    rng = np.random.default_rng(args.random_state)
    rows = []
    for arm in arms.itertuples(index=False):
        assigned_n = max(1, int(round(arm.sample_size * args.sample_scale)))
        missing_n = int(rng.binomial(assigned_n, args.missing_rate))
        observed_n = assigned_n - missing_n
        outcomes = int(
            rng.binomial(observed_n, arm.expected_visit_rate_offline)
        )
        targeted_n = int(round(assigned_n * arm.target_rate_pct / 100.0))
        treatment_received_n = int(
            rng.binomial(targeted_n, args.delivery_rate)
        )
        rows.append(
            {
                "arm": arm.arm,
                "assigned_n": assigned_n,
                "outcome_observed_n": observed_n,
                "outcomes": outcomes,
                "targeted_n": targeted_n,
                "treatment_received_n": treatment_received_n,
                "total_campaign_cost": targeted_n * args.treatment_cost,
                "is_synthetic": True,
            }
        )

    output_path = ROOT / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Synthetic aggregate created: {output_path}")


if __name__ == "__main__":
    main()
