from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


HILLSTROM_FEATURE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

HILLSTROM_OUTCOME_COLUMNS = ["visit", "conversion", "spend"]


@dataclass(frozen=True)
class HillstromDataset:
    """Binary-treatment version of the Hillstrom email experiment."""

    X: pd.DataFrame
    y: pd.Series
    treatment: pd.Series
    raw: pd.DataFrame
    feature_columns: list[str]
    treatment_segment: str
    control_segment: str
    outcome: str


def load_hillstrom(path: str | Path = "data/hillstrom_email.csv") -> pd.DataFrame:
    """Load the raw Hillstrom email dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hillstrom dataset not found at: {path}")
    return pd.read_csv(path)


def summarize_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Return high-level outcome rates by randomized segment."""
    summary = (
        df.groupby("segment", observed=True)
        .agg(
            n=("segment", "size"),
            visit_rate=("visit", "mean"),
            conversion_rate=("conversion", "mean"),
            avg_spend=("spend", "mean"),
        )
        .sort_index()
    )
    return summary


def make_binary_hillstrom(
    df: pd.DataFrame,
    treatment_segment: str = "Mens E-Mail",
    control_segment: str = "No E-Mail",
    outcome: str = "visit",
) -> HillstromDataset:
    """Create a two-arm Hillstrom dataset.

    The original experiment has three arms: Mens E-Mail, Womens E-Mail and
    No E-Mail. For the first warm-up run we compare one campaign against the
    no-email control group and drop the other campaign arm.
    """
    if outcome not in HILLSTROM_OUTCOME_COLUMNS:
        raise ValueError(f"Invalid outcome: {outcome}")

    keep_segments = [treatment_segment, control_segment]
    subset = df.loc[df["segment"].isin(keep_segments)].copy()
    if subset.empty:
        raise ValueError("No rows remain after filtering treatment/control segments.")

    subset["treatment"] = (subset["segment"] == treatment_segment).astype(int)
    y = subset[outcome].copy()
    treatment = subset["treatment"].copy()

    X_raw = subset[HILLSTROM_FEATURE_COLUMNS].copy()
    X = pd.get_dummies(
        X_raw,
        columns=["history_segment", "zip_code", "channel"],
        drop_first=False,
        dtype=float,
    )
    X.columns = make_safe_feature_names(X.columns)

    return HillstromDataset(
        X=X,
        y=y,
        treatment=treatment,
        raw=subset,
        feature_columns=list(X.columns),
        treatment_segment=treatment_segment,
        control_segment=control_segment,
        outcome=outcome,
    )


def make_safe_feature_names(columns: pd.Index) -> list[str]:
    """Normalize feature names for LightGBM compatibility."""
    safe_columns: list[str] = []
    seen: dict[str, int] = {}

    for column in columns:
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(column)).strip("_").lower()
        if not safe:
            safe = "feature"
        count = seen.get(safe, 0)
        seen[safe] = count + 1
        if count:
            safe = f"{safe}_{count}"
        safe_columns.append(safe)

    return safe_columns
