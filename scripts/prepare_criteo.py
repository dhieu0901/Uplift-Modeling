# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.criteo import load_criteo, prepare_criteo_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible Criteo development sample."
    )
    parser.add_argument("--data-path", default="data/criteo-uplift-v2.1.csv.gz")
    parser.add_argument(
        "--sample-path",
        default="data/processed/criteo_sample_500k.parquet",
    )
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Creating a reservoir sample of {args.sample_size:,} rows...")
    sample_path = prepare_criteo_sample(
        ROOT / args.data_path,
        ROOT / args.sample_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
        force=args.force,
    )
    sample = load_criteo(sample_path)
    print(f"Sample: {sample_path}")
    print(
        "Rates: "
        f"treatment={sample.treatment.mean():.6f}, "
        f"visit={sample.raw['visit'].mean():.6f}, "
        f"conversion={sample.raw['conversion'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
