from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from statistics import NormalDist
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.criteo import CriteoDataset
from src.evaluation.policy_value import (
    aipw_policy_contrast_table,
    aipw_policy_value_table,
)
from src.evaluation.uplift import separate_relative_auuc
from src.experiments.splitting import HonestUpliftSplits, split_train_validation_test
from src.models.registry import (
    REFERENCE_POLICIES,
    ModelFactory,
    default_model_factories,
)
from src.models.t_learner import TLearner


@dataclass(frozen=True)
class LockedPolicyEvaluation:
    """Outputs from refitting locked policies and opening a test sample once."""

    policy_values: pd.DataFrame
    contrasts: pd.DataFrame
    metrics: pd.DataFrame
    timing_table: pd.DataFrame
    scores: dict[str, np.ndarray]
    propensity: float
    n_fit: int
    n_test: int


@dataclass(frozen=True)
class HonestUpliftExperimentResult:
    """Outputs from validation-only selection and locked-test evaluation."""

    champion: str
    primary_budget: float
    selection_folds: int
    selection_size: int
    splits: HonestUpliftSplits
    validation_policy_values: pd.DataFrame
    validation_contrasts: pd.DataFrame
    validation_metrics: pd.DataFrame
    test_policy_values: pd.DataFrame
    test_contrasts: pd.DataFrame
    test_metrics: pd.DataFrame
    timing_table: pd.DataFrame
    validation_scores: dict[str, np.ndarray]
    test_scores: dict[str, np.ndarray]


def run_honest_uplift_experiment(
    dataset: CriteoDataset,
    model_factories: dict[str, ModelFactory] | None = None,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    budgets: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    primary_budget: float = 0.05,
    confidence_level: float = 0.95,
    random_state: int = 42,
    selection_folds: int = 1,
    evaluate_all_test: bool = False,
    progress: Callable[[str], None] | None = None,
) -> HonestUpliftExperimentResult:
    """Select an uplift model on validation and evaluate it once on test.

    Candidate learners and their operating budget are never selected from test
    outcomes. After validation chooses the champion, only that learner and the
    response baseline are refit on the combined development data.
    """
    factories = model_factories or default_model_factories()
    if "response_model" not in factories:
        raise ValueError("model_factories must include response_model.")
    uplift_candidates = [
        name for name in factories if name not in REFERENCE_POLICIES
    ]
    if not uplift_candidates:
        raise ValueError("At least one uplift candidate is required.")
    normalized_budgets = tuple(float(value) for value in budgets)
    if primary_budget not in normalized_budgets:
        raise ValueError("primary_budget must be included in budgets.")
    if selection_folds < 1:
        raise ValueError("selection_folds must be at least 1.")

    splits = split_train_validation_test(
        dataset,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    if selection_folds == 1:
        selection_split = splits.validation
        validation_scores, validation_timing = _fit_and_score_models(
            factories,
            fit_split=splits.train,
            score_X=selection_split.X,
            random_state=random_state,
            stage="selection",
            progress=progress,
        )
        validation_mu0, validation_mu1, nuisance_seconds = (
            _fit_nuisance_and_predict(
                splits.train,
                selection_split.X,
                random_state=random_state + 10_000,
            )
        )
        validation_timing = pd.concat(
            [
                validation_timing,
                pd.DataFrame(
                    [
                        {
                            "stage": "selection",
                            "model": "aipw_nuisance_t_learner",
                            "fit_seconds": nuisance_seconds,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        propensity = float(splits.train.treatment.mean())
    else:
        selection_split = _combine_development_splits(splits)
        (
            validation_scores,
            validation_mu0,
            validation_mu1,
            validation_timing,
        ) = _cross_validated_selection_predictions(
            factories,
            selection_split,
            n_splits=selection_folds,
            random_state=random_state,
            progress=progress,
        )
        propensity = float(selection_split.treatment.mean())

    validation_policy_values = aipw_policy_value_table(
        selection_split.y,
        selection_split.treatment,
        validation_scores,
        validation_mu0,
        validation_mu1,
        propensity=propensity,
        fractions=normalized_budgets,
        confidence_level=confidence_level,
    )
    validation_contrasts = aipw_policy_contrast_table(
        selection_split.y,
        selection_split.treatment,
        validation_scores,
        reference_policy="response_model",
        mu0=validation_mu0,
        mu1=validation_mu1,
        propensity=propensity,
        fractions=normalized_budgets,
        confidence_level=confidence_level,
    )
    validation_metrics = _ranking_metrics(
        selection_split.y,
        selection_split.treatment,
        validation_scores,
    )
    champion = select_validation_champion(
        validation_contrasts,
        candidate_policies=uplift_candidates,
        primary_budget=primary_budget,
    )

    development = _combine_development_splits(splits)
    if evaluate_all_test:
        final_factories = factories
    else:
        final_factories = {
            name: factories[name]
            for name in (*REFERENCE_POLICIES, champion)
            if name in factories
        }
    locked = evaluate_locked_policies(
        development,
        splits.test,
        final_factories,
        budgets=normalized_budgets,
        confidence_level=confidence_level,
        random_state=random_state + 20_000,
        progress=progress,
    )
    timing_table = pd.concat(
        [validation_timing, locked.timing_table],
        ignore_index=True,
    )
    return HonestUpliftExperimentResult(
        champion=champion,
        primary_budget=primary_budget,
        selection_folds=selection_folds,
        selection_size=len(selection_split.X),
        splits=splits,
        validation_policy_values=validation_policy_values,
        validation_contrasts=validation_contrasts,
        validation_metrics=validation_metrics,
        test_policy_values=locked.policy_values,
        test_contrasts=locked.contrasts,
        test_metrics=locked.metrics,
        timing_table=timing_table,
        validation_scores=validation_scores,
        test_scores=locked.scores,
    )


def select_validation_champion(
    validation_contrasts: pd.DataFrame,
    candidate_policies: Sequence[str],
    primary_budget: float,
) -> str:
    """Select the largest lower confidence bound at the locked budget."""
    candidates = _candidates_at_budget(
        validation_contrasts,
        candidate_policies,
        primary_budget,
    )
    ordered = candidates.sort_values(
        ["ci_lower", "difference"],
        ascending=[False, False],
    )
    return str(ordered.iloc[0]["policy"])


def multiplicity_adjusted_bounds(
    validation_contrasts: pd.DataFrame,
    candidate_policies: Sequence[str],
    primary_budget: float,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Re-derive each candidate's lower bound at a Bonferroni-corrected level.

    The selection rule reads one interval per candidate and keeps the largest
    lower bound, so that bound is a maximum over several intervals and reads
    high. Spreading the same error rate across the candidates gives the level
    at which "only this candidate clears zero" is a statement about the whole
    table rather than about one row of it.

    The champion is still chosen by :func:`select_validation_champion`, whose
    rule was fixed before any data was seen. Adjusting monotonically penalizes
    wide intervals more than narrow ones, so it can reorder candidates; using
    it to pick would be choosing a rule after seeing the result. This is
    reported next to the rule, never in place of it.
    """
    candidates = _candidates_at_budget(
        validation_contrasts,
        candidate_policies,
        primary_budget,
        required=frozenset(
            {"policy", "budget_pct", "difference", "ci_lower", "ci_upper"}
        ),
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1).")
    n_candidates = len(candidates)
    alpha = 1.0 - confidence_level
    z_nominal = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    z_adjusted = NormalDist().inv_cdf(1.0 - alpha / (2.0 * n_candidates))
    # n * standard_error, recovered from the interval the table already carries.
    standard_error = (
        candidates["ci_upper"] - candidates["ci_lower"]
    ) / (2.0 * z_nominal)
    return pd.DataFrame(
        {
            "policy": candidates["policy"].to_numpy(),
            "n_candidates": n_candidates,
            "adjusted_confidence_level": 1.0 - alpha / n_candidates,
            "ci_lower_adjusted": (
                candidates["difference"] - z_adjusted * standard_error
            ).to_numpy(),
        }
    )


def summarize_selection(
    result: HonestUpliftExperimentResult,
    budget_pct: float,
    tracked_policy: str = "s_learner",
) -> dict:
    """Record how close the selection was, not just who won.

    A champion that changes across runs can mean two very different things:
    the rule is unstable, or several candidates are genuinely tied and the
    rule is picking arbitrarily among equals. Those call for different
    responses, and the champion name alone cannot tell them apart. Recording
    the runner-up and the margin does.

    ``tracked_policy`` names the candidate whose rank travels with every run,
    so that a report can say where one learner placed even when it did not
    win. Its rank is returned under a key named after it.
    """
    contrasts = result.validation_contrasts
    ranked = contrasts[
        np.isclose(contrasts["budget_pct"], budget_pct)
        & contrasts["policy"].isin(
            [name for name in result.validation_scores if name not in REFERENCE_POLICIES]
        )
    ].dropna(subset=["ci_lower"]).sort_values("ci_lower", ascending=False)

    if ranked.empty:
        return {}

    champion_row = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None
    order = list(ranked["policy"])

    return {
        "runner_up": None if runner_up is None else str(runner_up["policy"]),
        "selection_margin": (
            float("nan")
            if runner_up is None
            else float(champion_row["ci_lower"] - runner_up["ci_lower"])
        ),
        "champion_selection_ci_lower": float(champion_row["ci_lower"]),
        "runner_up_selection_ci_lower": (
            float("nan") if runner_up is None else float(runner_up["ci_lower"])
        ),
        # The margin only means something next to the uncertainty in the
        # quantity being ranked, so the champion's own interval travels with it.
        "champion_selection_halfwidth": float(
            (champion_row["ci_upper"] - champion_row["ci_lower"]) / 2.0
        ),
        "n_candidates_with_positive_bound": int((ranked["ci_lower"] > 0).sum()),
        f"{tracked_policy}_selection_rank": (
            order.index(tracked_policy) + 1 if tracked_policy in order else -1
        ),
    }


def _candidates_at_budget(
    validation_contrasts: pd.DataFrame,
    candidate_policies: Sequence[str],
    primary_budget: float,
    required: frozenset[str] = frozenset(
        {"policy", "budget_pct", "difference", "ci_lower"}
    ),
) -> pd.DataFrame:
    missing = sorted(required - set(validation_contrasts.columns))
    if missing:
        raise ValueError(f"Validation table is missing columns: {missing}")
    budget_pct = round(100.0 * float(primary_budget), 4)
    candidates = validation_contrasts[
        validation_contrasts["policy"].isin(candidate_policies)
        & np.isclose(validation_contrasts["budget_pct"], budget_pct)
    ].dropna(subset=["ci_lower"])
    if candidates.empty:
        raise ValueError("No candidate has a valid estimate at the primary budget.")
    return candidates


def evaluate_locked_policies(
    fit_data,
    test_data,
    model_factories: dict[str, ModelFactory],
    budgets: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    confidence_level: float = 0.95,
    random_state: int = 42,
    reference_policy: str = "response_model",
    progress: Callable[[str], None] | None = None,
) -> LockedPolicyEvaluation:
    """Refit already-locked policies on ``fit_data`` and score ``test_data`` once.

    No selection happens here. Every choice -- learner, operating budget,
    confidence level -- must already be locked before this function runs, which
    is what makes the resulting interval an honest read of the chosen policy.

    Both arguments only need ``X``, ``y``, and ``treatment`` attributes, so a
    :class:`~src.data.criteo.CriteoDataset` and an
    :class:`~src.experiments.splitting.UpliftSplit` are equally acceptable. That
    lets the same code path serve the internal locked test and a confirmatory
    test drawn from a separate file.
    """
    if reference_policy not in model_factories:
        raise ValueError(f"model_factories must include {reference_policy}.")
    normalized_budgets = tuple(float(value) for value in budgets)

    scores, timing = _fit_and_score_models(
        model_factories,
        fit_split=fit_data,
        score_X=test_data.X,
        random_state=random_state,
        stage="locked_test",
        progress=progress,
    )
    mu0, mu1, nuisance_seconds = _fit_nuisance_and_predict(
        fit_data,
        test_data.X,
        random_state=random_state + 10_000,
    )
    propensity = float(fit_data.treatment.mean())
    policy_values = aipw_policy_value_table(
        test_data.y,
        test_data.treatment,
        scores,
        mu0,
        mu1,
        propensity=propensity,
        fractions=normalized_budgets,
        confidence_level=confidence_level,
    )
    contrasts = aipw_policy_contrast_table(
        test_data.y,
        test_data.treatment,
        scores,
        reference_policy=reference_policy,
        mu0=mu0,
        mu1=mu1,
        propensity=propensity,
        fractions=normalized_budgets,
        confidence_level=confidence_level,
    )
    metrics = _ranking_metrics(test_data.y, test_data.treatment, scores)
    timing_table = pd.concat(
        [
            timing,
            pd.DataFrame(
                [
                    {
                        "stage": "locked_test",
                        "model": "aipw_nuisance_t_learner",
                        "fit_seconds": nuisance_seconds,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return LockedPolicyEvaluation(
        policy_values=policy_values,
        contrasts=contrasts,
        metrics=metrics,
        timing_table=timing_table,
        scores=scores,
        propensity=propensity,
        n_fit=len(fit_data.X),
        n_test=len(test_data.X),
    )


def _model_seed(random_state: int, name: str) -> int:
    """Derive a per-model seed that does not depend on the candidate set.

    Seeding by a model's position in the factory dict means that adding or
    removing one candidate silently reseeds every candidate after it, so two
    runs that share a learner are not comparable. Hashing the name keeps each
    seed a function of the base seed and the model name only.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
    return int((int(random_state) + int.from_bytes(digest, "big")) % (2**31 - 1))


def _fit_and_score_models(
    factories: dict[str, ModelFactory],
    fit_split,
    score_X: pd.DataFrame,
    random_state: int,
    stage: str,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    scores = {}
    timing_rows = []
    for name, factory in factories.items():
        if progress is not None:
            progress(f"{stage}: {name}")
        model = factory()
        started = perf_counter()
        model.fit(
            fit_split.X,
            fit_split.y,
            fit_split.treatment,
            random_state=_model_seed(random_state, name),
        )
        elapsed = perf_counter() - started
        if name == "response_model":
            score = model.predict_score(score_X)
        else:
            score = model.predict_uplift(score_X)
        scores[name] = np.asarray(score, dtype=float)
        timing_rows.append(
            {"stage": stage, "model": name, "fit_seconds": elapsed}
        )
    return scores, pd.DataFrame(timing_rows)


def _fit_nuisance_and_predict(
    fit_split,
    score_X: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = TLearner()
    started = perf_counter()
    model.fit(
        fit_split.X,
        fit_split.y,
        fit_split.treatment,
        random_state=random_state,
    )
    elapsed = perf_counter() - started
    return model.predict_mu0(score_X), model.predict_mu1(score_X), elapsed


def _cross_validated_selection_predictions(
    factories: dict[str, ModelFactory],
    development,
    n_splits: int,
    random_state: int,
    progress: Callable[[str], None] | None,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    strata = (
        development.treatment.astype(str)
        + "_"
        + development.y.astype(str)
    )
    smallest_stratum = int(strata.value_counts().min())
    actual_splits = min(n_splits, smallest_stratum)
    if actual_splits < 2:
        raise ValueError(
            "Cross-validated selection requires at least two observations "
            "in every treatment/outcome stratum."
        )
    splitter = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = {
        name: np.empty(len(development.X), dtype=float)
        for name in factories
    }
    mu0 = np.empty(len(development.X), dtype=float)
    mu1 = np.empty(len(development.X), dtype=float)
    timing_frames = []

    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(development.X, strata),
        start=1,
    ):
        fit_split = _subset_existing_split(development, fit_indices)
        holdout_X = development.X.iloc[holdout_indices].reset_index(drop=True)
        fold_scores, fold_timing = _fit_and_score_models(
            factories,
            fit_split=fit_split,
            score_X=holdout_X,
            random_state=random_state + 1000 * fold,
            stage=f"selection_fold_{fold}",
            progress=progress,
        )
        for name, score in fold_scores.items():
            # Each fold is scored by a separately fitted model, so raw scores
            # are only comparable within a fold. Targeting the top 5% of the
            # pooled column would otherwise favour whichever fold happened to
            # produce the largest numbers. Ranking within the fold first makes
            # the pooled column a ranking across folds, which is all the
            # budget-constrained policy needs.
            scores[name][holdout_indices] = _percentile_rank(score)
        fold_mu0, fold_mu1, nuisance_seconds = _fit_nuisance_and_predict(
            fit_split,
            holdout_X,
            random_state=random_state + 10_000 + 1000 * fold,
        )
        mu0[holdout_indices] = fold_mu0
        mu1[holdout_indices] = fold_mu1
        timing_frames.extend(
            [
                fold_timing,
                pd.DataFrame(
                    [
                        {
                            "stage": f"selection_fold_{fold}",
                            "model": "aipw_nuisance_t_learner",
                            "fit_seconds": nuisance_seconds,
                        }
                    ]
                ),
            ]
        )
    return scores, mu0, mu1, pd.concat(timing_frames, ignore_index=True)


def _ranking_metrics(
    y: pd.Series,
    treatment: pd.Series,
    scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy": name,
                "benchmark_relative_auuc": separate_relative_auuc(
                    y,
                    treatment,
                    score,
                ),
            }
            for name, score in scores.items()
        ]
    ).sort_values("benchmark_relative_auuc", ascending=False)


def _combine_development_splits(splits: HonestUpliftSplits):
    from src.experiments.splitting import UpliftSplit

    return UpliftSplit(
        X=pd.concat(
            [splits.train.X, splits.validation.X],
            ignore_index=True,
        ),
        y=pd.concat(
            [splits.train.y, splits.validation.y],
            ignore_index=True,
        ),
        treatment=pd.concat(
            [splits.train.treatment, splits.validation.treatment],
            ignore_index=True,
        ),
        indices=np.concatenate(
            [splits.train.indices, splits.validation.indices]
        ),
    )


def _subset_existing_split(split, indices: np.ndarray):
    from src.experiments.splitting import UpliftSplit

    return UpliftSplit(
        X=split.X.iloc[indices].reset_index(drop=True),
        y=split.y.iloc[indices].reset_index(drop=True),
        treatment=split.treatment.iloc[indices].reset_index(drop=True),
        indices=split.indices[indices],
    )


def _percentile_rank(score: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(score, dtype=float)).rank(
        method="average",
        pct=True,
    ).to_numpy(dtype=float)
