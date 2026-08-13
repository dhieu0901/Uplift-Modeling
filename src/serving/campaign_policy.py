from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeasuredBudget:
    """What the locked test measured at one operating budget.

    Rates rather than totals, so the same measurement applies to a campaign of
    any size. Both are per evaluated user: ``incremental_rate`` against
    contacting nobody, ``gain_rate`` against the incumbent response model.
    """

    budget_pct: float
    incremental_rate: float
    incremental_ci_lower_rate: float
    incremental_ci_upper_rate: float
    gain_rate: float
    gain_ci_lower_rate: float
    gain_ci_upper_rate: float


@dataclass
class CampaignPolicy:
    """A locked targeting policy together with what it was measured to do.

    The model alone cannot say what a campaign will gain: a score ranks users
    but carries no units. Shipping the measured rates with the model is what
    lets a target list come with a number attached, and keeps that number
    traceable to the sample it came from.
    """

    model: object
    model_name: str
    outcome: str
    feature_columns: list[str]
    propensity: float
    confidence_level: float
    fit_sample: str
    fit_rows: int
    measured_sample: str
    measured_rows: int
    fitted_at: str
    model_seed: int
    budgets: list[MeasuredBudget] = field(default_factory=list)

    def rank(self, X: pd.DataFrame) -> np.ndarray:
        """Score users, highest expected change first."""
        missing = [name for name in self.feature_columns if name not in X.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")
        score = self.model.predict_uplift(X[self.feature_columns])
        return np.asarray(score, dtype=float)

    def select(self, X: pd.DataFrame, budget: float) -> np.ndarray:
        """Return a boolean mask of the users a budget can afford to contact.

        The cutoff matches the evaluation exactly, so the list this produces is
        the list the reported numbers describe.
        """
        budget_pct = self._validate_budget(budget)
        score = self.rank(X)
        n = len(score)
        n_targeted = max(1, int(round(budget_pct / 100.0 * n)))
        order = np.argsort(-score, kind="mergesort")
        mask = np.zeros(n, dtype=bool)
        mask[order[:n_targeted]] = True
        return mask

    def expected_outcome(self, budget: float, n_users: int) -> dict:
        """Scale the locked measurement to a campaign of ``n_users``.

        This is a projection, not a measurement of the users passed in. A list
        awaiting contact has no outcomes to measure, so the only honest number
        is the rate observed on the locked test carried across.
        """
        if n_users <= 0:
            raise ValueError("n_users must be positive.")
        measured = self._budget_at(self._validate_budget(budget))
        n_targeted = max(1, int(round(measured.budget_pct / 100.0 * n_users)))
        return {
            "budget_pct": measured.budget_pct,
            "n_users": int(n_users),
            "n_targeted": n_targeted,
            "incremental_outcome": measured.incremental_rate * n_users,
            "incremental_ci_lower": measured.incremental_ci_lower_rate * n_users,
            "incremental_ci_upper": measured.incremental_ci_upper_rate * n_users,
            "gain_vs_incumbent": measured.gain_rate * n_users,
            "gain_ci_lower": measured.gain_ci_lower_rate * n_users,
            "gain_ci_upper": measured.gain_ci_upper_rate * n_users,
            "gain_excludes_zero": bool(
                measured.gain_ci_lower_rate > 0.0
                or measured.gain_ci_upper_rate < 0.0
            ),
        }

    def measured_budgets(self) -> list[float]:
        return [item.budget_pct for item in self.budgets]

    def _validate_budget(self, budget: float) -> float:
        budget_pct = round(100.0 * float(budget), 4)
        available = self.measured_budgets()
        if not any(np.isclose(budget_pct, value) for value in available):
            raise ValueError(
                f"Budget {budget_pct}% was never evaluated. Available: "
                f"{available}. Interpolating between them would invent a "
                "confidence interval that no sample supports."
            )
        return budget_pct

    def _budget_at(self, budget_pct: float) -> MeasuredBudget:
        for item in self.budgets:
            if np.isclose(item.budget_pct, budget_pct):
                return item
        raise ValueError(f"No measurement at {budget_pct}%.")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> CampaignPolicy:
        policy = joblib.load(Path(path))
        if not isinstance(policy, CampaignPolicy):
            raise ValueError(f"{path} does not hold a CampaignPolicy.")
        return policy


def measured_budgets_from_tables(
    policy_values: pd.DataFrame,
    contrasts: pd.DataFrame,
    model_name: str,
    confidence_level: float,
) -> list[MeasuredBudget]:
    """Read the locked test's own output files into the served measurement.

    Taking the rates from the tracked tables rather than recomputing them keeps
    the number a campaign is quoted identical to the number in the report.
    """
    values = policy_values[policy_values["policy"] == model_name]
    gains = contrasts[contrasts["policy"] == model_name]
    if values.empty:
        raise ValueError(f"No policy values recorded for {model_name}.")
    z = _critical_value(confidence_level)

    budgets = []
    for _, row in values.sort_values("budget_pct").iterrows():
        matched = gains[np.isclose(gains["budget_pct"], row["budget_pct"])]
        if matched.empty:
            continue
        gain = matched.iloc[0]
        half_width = z * float(gain["standard_error_rate"])
        budgets.append(
            MeasuredBudget(
                budget_pct=float(row["budget_pct"]),
                incremental_rate=float(row["incremental_outcome_rate"]),
                incremental_ci_lower_rate=float(row["ci_lower_rate"]),
                incremental_ci_upper_rate=float(row["ci_upper_rate"]),
                gain_rate=float(gain["difference_rate"]),
                gain_ci_lower_rate=float(gain["difference_rate"]) - half_width,
                gain_ci_upper_rate=float(gain["difference_rate"]) + half_width,
            )
        )
    if not budgets:
        raise ValueError("No budget appears in both the value and contrast tables.")
    return budgets


def _critical_value(confidence_level: float) -> float:
    from statistics import NormalDist

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
