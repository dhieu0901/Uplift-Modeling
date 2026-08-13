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

from src.evaluation.exposure_iv import ExposureAnalysis, analyse_exposure
from src.reporting import dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify what the exposure column appears to show and what it "
            "actually identifies, so the decision to exclude it from the "
            "models rests on a number rather than on an argument."
        )
    )
    parser.add_argument(
        "--sample-path",
        default="data/processed/criteo_sample_500k.parquet",
        help=(
            "Development sample. This is a property of the data rather than a "
            "model result, so it does not spend the confirmatory sample."
        ),
    )
    parser.add_argument("--outcomes", default="visit,conversion")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--report-path", default="outputs/exposure_iv.md")
    parser.add_argument("--table-path", default="outputs/tables/exposure_iv.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = [name.strip() for name in args.outcomes.split(",") if name.strip()]
    if not outcomes:
        raise ValueError("At least one outcome is required.")

    frame = pd.read_parquet(
        ROOT / args.sample_path,
        columns=["treatment", "exposure", *outcomes],
    )
    analyses = [
        analyse_exposure(frame, outcome, confidence_level=args.confidence_level)
        for outcome in outcomes
    ]

    report_path = ROOT / args.report_path
    table_path = ROOT / args.table_path
    for path in (report_path, table_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([_row(analysis) for analysis in analyses])
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(args, analyses, table_path),
        encoding="utf-8",
    )

    print(f"Exposure report: {report_path}")
    print(table.to_string(index=False))


def _row(analysis: ExposureAnalysis) -> dict:
    return {
        "outcome": analysis.outcome,
        "rate_control_pct": 100.0 * analysis.rate_control,
        "rate_assigned_unexposed_pct": 100.0 * analysis.rate_assigned_unexposed,
        "rate_assigned_exposed_pct": 100.0 * analysis.rate_assigned_exposed,
        "naive_gap_pp": 100.0 * analysis.naive_gap.value,
        "itt_pp": 100.0 * analysis.intention_to_treat.value,
        "itt_ci_lower_pp": 100.0 * analysis.intention_to_treat.ci_lower,
        "itt_ci_upper_pp": 100.0 * analysis.intention_to_treat.ci_upper,
        "exposure_rate_pct": 100.0 * analysis.exposure_rate.value,
        "complier_effect_pp": 100.0 * analysis.complier_effect.value,
        "complier_effect_ci_lower_pp": 100.0 * analysis.complier_effect.ci_lower,
        "complier_effect_ci_upper_pp": 100.0 * analysis.complier_effect.ci_upper,
        "complier_baseline_pct": 100.0 * analysis.complier_baseline,
        "selection_pp": 100.0
        * (analysis.naive_gap.value - analysis.complier_effect.value),
    }


def _group_table(analysis: ExposureAnalysis) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": "not assigned",
                "users": analysis.n_control,
                f"{analysis.outcome}_pct": 100.0 * analysis.rate_control,
            },
            {
                "group": "assigned, ad never rendered",
                "users": analysis.n_assigned_unexposed,
                f"{analysis.outcome}_pct": 100.0
                * analysis.rate_assigned_unexposed,
            },
            {
                "group": "assigned, ad rendered",
                "users": analysis.n_assigned_exposed,
                f"{analysis.outcome}_pct": 100.0 * analysis.rate_assigned_exposed,
            },
        ]
    )


def _dilution_note(analysis: ExposureAnalysis) -> str:
    """Relate the two estimates, since only one of them drives a decision."""
    share = analysis.exposure_rate.value
    cace_positive = analysis.complier_effect.ci_lower > 0.0
    itt_positive = analysis.intention_to_treat.ci_lower > 0.0
    lead = (
        f"Only `{100.0 * share:.2f}%` of assigned users see the ad, so the "
        f"effect among them is diluted by roughly `{1.0 / share:.0f}x` before "
        "it reaches the number a campaign can act on."
    )
    if cace_positive and not itt_positive:
        return (
            f"{lead} That is why an effect can be clearly positive among users "
            "who see the ad while the population-level number stays "
            "inconclusive: the two are measuring different groups, not "
            "disagreeing."
        )
    return (
        f"{lead} Both estimates point the same way here; they differ in which "
        "group they describe, not in direction."
    )


def _interval(estimate) -> str:
    return (
        f"`[{100.0 * estimate.ci_lower:+.4f}, "
        f"{100.0 * estimate.ci_upper:+.4f}]`"
    )


def _section(analysis: ExposureAnalysis) -> str:
    naive = 100.0 * analysis.naive_gap.value
    cace = analysis.complier_effect
    itt = analysis.intention_to_treat
    selection = naive - 100.0 * cace.value
    broken = analysis.rate_assigned_unexposed < analysis.rate_control
    evidence = (
        (
            f"Users who were assigned but never saw the ad sit at "
            f"`{100.0 * analysis.rate_assigned_unexposed:.3f}%`, **below** the "
            f"`{100.0 * analysis.rate_control:.3f}%` of users never assigned at "
            "all. Sending an ad cannot make someone less likely to act, so that "
            "ordering is not an effect: it shows the split on a post-assignment "
            "column has selected on the outcome."
        )
        if broken
        else (
            "Users assigned but never exposed do not sit below the control arm "
            "here, so this particular symptom is absent - the column is still "
            "decided after assignment and still cannot be used for targeting."
        )
    )
    return f"""### {analysis.outcome}

{dataframe_to_markdown(_group_table(analysis))}

Reading the last two rows as a comparison gives **{naive:+.2f} pp**. {evidence}

The mechanism is that an ad renders only while someone is already browsing, so
`exposure = 1` marks users who were already on their way to the site.

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| Effect of being assigned, everyone | `{100.0 * itt.value:+.4f} pp` | {_interval(itt)} |
| Share of assigned users whose ad rendered | `{100.0 * analysis.exposure_rate.value:.4f}%` | - |
| **Effect among users whose ad renders** | **`{100.0 * cace.value:+.4f} pp`** | {_interval(cace)} |

Splitting the apparent gap:

| Component | Value |
|---|---:|
| Apparent gap, rendered minus not assigned | `{naive:+.2f} pp` |
| What those users would have done untreated | `{100.0 * analysis.complier_baseline:.3f}%` |
| Effect the ad actually caused among them | `{100.0 * cace.value:+.2f} pp` |
| Selection rather than campaign | `{selection:+.2f} pp` |

Those users were already \
`{analysis.complier_baseline / analysis.rate_control:.1f}x` more likely to act \
than the population before any ad was shown.

{_dilution_note(analysis)}"""


def build_report(
    args: argparse.Namespace,
    analyses: list[ExposureAnalysis],
    table_path: Path,
) -> str:
    sections = "\n\n".join(_section(analysis) for analysis in analyses)
    return f"""# What the Exposure Column Would Give

## Why This Exists

`exposure` records whether the ad actually rendered. The models exclude it, and
this report is the reason in numbers rather than in argument: it reports both
what the column appears to show and what it can legitimately identify.

The question the column invites - *does someone who sees the ad and then acts
show the campaign working?* - is a real one. It is answerable, just not by
comparing the exposed group against the control arm.

## Protocol

- Data: `{Path(args.sample_path).as_posix()}` ({analyses[0].n:,} rows). This is a
  development sample on purpose. What follows describes how the data was
  recorded rather than how a model performs, so it has no business spending the
  confirmatory sample, which is opened once.
- Confidence level: `{100.0 * args.confidence_level:.1f}%`.
- Assignment is randomized and **no** control user can be exposed, so
  assignment is a valid instrument for exposure under one-sided noncompliance
  and the Wald ratio identifies the effect among users whose ad renders.
- Standard errors for that ratio come from its influence function, because the
  numerator and denominator are computed on the same users.

## Results

{sections}

## What This Licenses

- The campaign chooses **whom to send to**, not whose ad renders. The
  decision-relevant number stays the effect of being assigned.
- Nothing here can be targeted on. At the moment a user is selected it is not
  known whether their ad will render, so `exposure` cannot enter a model that
  has to rank users before the campaign is sent.
- The exclusion restriction is assumed: assignment moves the outcome only
  through the ad rendering. For display advertising that is reasonable, since
  an ad that never rendered leaves nothing for the user to respond to. It is an
  assumption rather than something this data can verify.
- The effect among users whose ad renders describes a group that cannot be
  identified in advance, so it explains the mechanism rather than supporting a
  targeting decision.

## Reproducible Outputs

- Summary table: `{table_path.relative_to(ROOT).as_posix()}`
"""


if __name__ == "__main__":
    main()
