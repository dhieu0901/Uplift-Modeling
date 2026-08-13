# ruff: noqa: E402

from __future__ import annotations

import argparse
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

from src.data.criteo import CRITEO_ROW_ID
from src.serving.campaign_policy import CampaignPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a user file and a budget into the list to contact, together "
            "with what the locked test says that list is worth."
        )
    )
    parser.add_argument("--policy-path", default="artifacts/campaign_policy.joblib")
    parser.add_argument(
        "--users",
        default="data/processed/criteo_confirm_4m.parquet",
        help="Users to rank. Outcome columns are ignored if present.",
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Score only the first N rows, for a quick look.",
    )
    parser.add_argument("--out", default="outputs/target_list.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = CampaignPolicy.load(ROOT / args.policy_path)

    columns = list(policy.feature_columns)
    users = pd.read_parquet(ROOT / args.users)
    carried = [name for name in (CRITEO_ROW_ID,) if name in users.columns]
    if args.max_rows is not None:
        users = users.head(args.max_rows)
    frame = users[columns + carried].reset_index(drop=True)

    score = policy.rank(frame)
    selected = policy.select(frame, args.budget)
    projection = policy.expected_outcome(args.budget, n_users=len(frame))

    target = frame.loc[selected, carried].copy()
    target["uplift_score"] = score[selected]
    target = target.sort_values("uplift_score", ascending=False)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Model      {policy.model_name}  (fitted {policy.fitted_at})")
    print(f"Outcome    {policy.outcome}")
    print(f"Users      {projection['n_users']:,}")
    print(
        f"Budget     {projection['budget_pct']:g}%  ->  "
        f"{projection['n_targeted']:,} selected"
    )
    print()
    print("Projected from the locked test, not measured on these users:")
    print(
        f"  incremental {policy.outcome}s   "
        f"{projection['incremental_outcome']:+,.0f}   "
        f"[{projection['incremental_ci_lower']:+,.0f}, "
        f"{projection['incremental_ci_upper']:+,.0f}]"
    )
    print(
        f"  gain over the incumbent  "
        f"{projection['gain_vs_incumbent']:+,.0f}   "
        f"[{projection['gain_ci_lower']:+,.0f}, "
        f"{projection['gain_ci_upper']:+,.0f}]"
        f"   {'clears zero' if projection['gain_excludes_zero'] else 'contains zero'}"
    )
    print()
    print(f"Wrote {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
