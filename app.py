# ruff: noqa: E402
"""Streamlit front end for the locked campaign policy.

Run with ``python -m streamlit run app.py`` so the page uses the same
interpreter as the rest of the project.

The page estimates nothing of its own. It loads the policy saved by
``scripts/fit_campaign_policy.py``, ranks whatever users it is given, and
reports the rates the confirmatory test measured, scaled to the campaign size.
It also reports how far the winner moved when the prediction model underneath
the learner was changed, read from the comparison table rather than recomputed.
Every number shown is traceable to a tracked file under ``outputs/``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.criteo import CRITEO_ROW_ID
from src.serving.campaign_policy import CampaignPolicy

POLICY_PATH = ROOT / "artifacts" / "campaign_policy.joblib"
DEFAULT_USERS = ROOT / "data" / "processed" / "criteo_confirm_4m.parquet"
BASE_LEARNER_SUMMARY = (
    ROOT / "outputs" / "tables" / "base_learner_selection_summary.csv"
)
BASE_LEARNER_ROWS = ROOT / "outputs" / "tables" / "base_learner_comparison.csv"
COMPARISON_FIGURE = ROOT / "outputs" / "figures" / "base_learner_comparison.png"

#: Plain names for the estimator families, because a page aimed at a campaign
#: decision should not require the reader to know what a gradient booster is.
BASE_FAMILY_LABELS = {
    "gradient_boosting": "Boosted trees",
    "linear": "Linear model",
    "forest": "Random forest",
}

#: One line on what each estimator does differently, and what that buys.
BASE_FAMILY_BLURB = {
    "gradient_boosting": (
        "Many shallow trees, each correcting the last. Flexible enough to find "
        "interactions nobody specified, and the configuration the campaign runs on."
    ),
    "linear": (
        "A penalised straight-line fit. It cannot represent an interaction it "
        "was not given, which makes it hard to overfit a noisy target."
    ),
    "forest": (
        "Many deep trees grown independently and averaged. Tree-based like "
        "boosting, but it reduces variance instead of bias."
    ),
}

INK = "#16202A"
MUTED = "#5C6975"
TEAL = "#0E6E64"
GREY = "#98A2AA"
RULE = "#D9D9D1"
RED = "#A83E36"

st.set_page_config(
    page_title="Campaign targeting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 2.4rem; max-width: 1180px; }

  .masthead { border-bottom: 1px solid #D9D9D1; padding-bottom: 1.1rem;
              margin-bottom: 1.8rem; }
  .masthead .eyebrow { font-size: 0.7rem; letter-spacing: 0.16em;
              text-transform: uppercase; color: #5C6975; font-weight: 600; }
  .masthead h1 { font-size: 2.1rem; font-weight: 650; letter-spacing: -0.02em;
              margin: 0.35rem 0 0.5rem; color: #16202A; }
  .masthead p { color: #5C6975; margin: 0; max-width: 68ch; font-size: 0.95rem;
              line-height: 1.6; }

  .label { font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
              color: #5C6975; font-weight: 600; margin-bottom: 0.7rem; }

  div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E2DA;
              border-radius: 4px; padding: 1rem 1.15rem; }
  div[data-testid="stMetric"] label p { font-size: 0.68rem !important;
              letter-spacing: 0.13em; text-transform: uppercase; color: #5C6975; }
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums;
              font-size: 1.9rem; letter-spacing: -0.02em; }

  .verdict { border-radius: 4px; padding: 0.85rem 1.1rem; font-size: 0.93rem;
              line-height: 1.55; border-left: 3px solid; }
  .verdict.clear { background: #E9F2F0; border-color: #0E6E64; color: #123F3A; }
  .verdict.unclear { background: #F5EEE6; border-color: #A8721E; color: #4A3616; }
  .verdict b { font-variant-numeric: tabular-nums; }

  .provenance { font-size: 0.82rem; line-height: 1.75; color: #16202A; }
  .provenance .k { color: #5C6975; }
  .provenance .v { font-variant-numeric: tabular-nums; }

  .footnote { color: #5C6975; font-size: 0.83rem; line-height: 1.6;
              max-width: 78ch; }

  section[data-testid="stSidebar"] { border-right: 1px solid #D9D9D1; }
  section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

  div[data-testid="stDataFrame"] { border: 1px solid #E2E2DA; border-radius: 4px; }
  hr { border-color: #D9D9D1; margin: 2.2rem 0 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


def fill_width(element) -> dict:
    """Ask a Streamlit element to fill its container, on either API.

    Streamlit renamed this: ``use_container_width=True`` became
    ``width="stretch"``. The old name warns on recent versions and is dropped
    after 2025-12-31, and the version pinned in ``requirements.lock.txt`` has
    only the old one, where ``width`` still means a pixel count. Detecting
    which one this install has keeps the page quiet on new Streamlit without
    breaking it on the pinned build.
    """
    parameters = inspect.signature(element).parameters
    width = parameters.get("width")
    if width is not None and "int" not in str(width.annotation):
        return {"width": "stretch"}
    for legacy in ("use_container_width", "use_column_width"):
        if legacy in parameters:
            return {legacy: True}
    return {}


def one_of(choice):
    """Narrow a select_slider result to the single value it returned.

    ``st.select_slider`` hands back a pair when it is given a pair as its
    starting value. Both sliders on this page are given one value, so both
    return one, and this states that rather than leaving it implied.
    """
    return choice[0] if isinstance(choice, tuple) else choice


def file_stamp(*paths: Path) -> tuple:
    """Identify a set of files by size and modification time.

    Streamlit caches on argument values, so a loader that takes no arguments
    keeps serving its first read forever. Passing this in means rerunning a
    script and refreshing the page shows the new evidence rather than the old,
    which matters more here than in most apps: the whole claim of the page is
    that its numbers come from those files.
    """
    return tuple(
        (path.stat().st_size, path.stat().st_mtime) if path.exists() else None
        for path in paths
    )


@st.cache_resource(show_spinner=False)
def load_policy(_stamp: tuple) -> CampaignPolicy:
    return CampaignPolicy.load(POLICY_PATH)


@st.cache_data(show_spinner=False)
def load_base_learner_comparison(_stamp: tuple) -> pd.DataFrame | None:
    """Read what each prediction model produced, not just which learner won.

    Returns ``None`` when the comparison has not been run, so a fresh clone
    still serves the policy instead of failing on a missing file.

    Effects are carried as per-user rates so the page can scale them to
    whatever campaign size is being asked about. The ratio is carried as a
    number and formatted at render time, because the page has to compare it
    against one and reading a number back out of its own label would be a way
    to get that wrong later.
    """
    if not (BASE_LEARNER_SUMMARY.exists() and BASE_LEARNER_ROWS.exists()):
        return None
    summary = pd.read_csv(BASE_LEARNER_SUMMARY, encoding="utf-8-sig")
    rows = pd.read_csv(BASE_LEARNER_ROWS, encoding="utf-8-sig")
    winners = rows.merge(
        summary[["base_family", "champion", "runner_up", "margin_over_halfwidth"]],
        left_on=["base_family", "policy"],
        right_on=["base_family", "champion"],
        how="inner",
    )
    # The tables count outcomes over the sample they were measured on. Dividing
    # the count by its own rate recovers that sample size, which is what turns
    # every column into something a campaign of any size can be quoted in.
    evaluated = winners["difference"] / winners["difference_rate"]
    return pd.DataFrame(
        {
            "base_family": winners["base_family"],
            "Prediction model": winners["base_family"]
            .map(BASE_FAMILY_LABELS)
            .fillna(winners["base_family"]),
            "Winner": winners["champion"],
            "Runner-up": winners["runner_up"],
            "effect_rate": winners["difference_rate"].astype(float),
            "lower_rate": (winners["ci_lower"] / evaluated).astype(float),
            "upper_rate": (winners["ci_upper"] / evaluated).astype(float),
            "Lead over runner-up": winners["margin_over_halfwidth"].astype(float),
            "budget_pct": winners["budget_pct"].astype(float),
        }
    )


@st.cache_data(show_spinner=False)
def load_all_candidates(_stamp: tuple) -> pd.DataFrame | None:
    """Every candidate in every family, as per-user rates.

    The winners table answers "which one won". This answers "by how much, and
    over whom", which is what makes a per-family tab worth opening.
    """
    if not BASE_LEARNER_ROWS.exists():
        return None
    rows = pd.read_csv(BASE_LEARNER_ROWS, encoding="utf-8-sig")
    evaluated = rows["difference"] / rows["difference_rate"]
    return pd.DataFrame(
        {
            "base_family": rows["base_family"],
            "Rank": rows["selection_rank"].astype(int),
            "Candidate": rows["policy"],
            "effect_rate": rows["difference_rate"].astype(float),
            "lower_rate": (rows["ci_lower"] / evaluated).astype(float),
            "upper_rate": (rows["ci_upper"] / evaluated).astype(float),
        }
    )


@st.cache_data(show_spinner=False)
def rank_users(path: str, n_rows: int, _policy: CampaignPolicy) -> pd.DataFrame:
    columns = [*_policy.feature_columns, CRITEO_ROW_ID]
    frame = pd.read_parquet(path)
    keep = [name for name in columns if name in frame.columns]
    frame = frame[keep].head(n_rows).reset_index(drop=True)
    return frame.assign(uplift_score=_policy.rank(frame))


if not POLICY_PATH.exists():
    st.error(
        "No saved policy found. Build one first with "
        "`python scripts/fit_campaign_policy.py`"
    )
    st.stop()

policy = load_policy(file_stamp(POLICY_PATH))

st.markdown(
    f"""
<div class="masthead">
  <div class="eyebrow">Locked policy &nbsp;/&nbsp; {policy.model_name}</div>
  <h1>Who should the campaign contact?</h1>
  <p>The budget reaches only part of the list, so users are ranked by how much
  contact is expected to <em>change</em> them rather than by how likely they are
  to act anyway.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="label">Where the numbers come from</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="provenance">
  <span class="k">Model</span><br><b>{policy.model_name}</b>, outcome
  <b>{policy.outcome}</b><br><br>
  <span class="k">Fitted on</span><br>
  <span class="v">{policy.fit_rows:,}</span> users<br>
  <code>{policy.fit_sample}</code><br><br>
  <span class="k">Measured on</span><br>
  <span class="v">{policy.measured_rows:,}</span> users, opened once<br>
  <code>{policy.measured_sample}</code><br><br>
  <span class="k">Fitted</span> <span class="v">{policy.fitted_at}</span><br>
  <span class="k">Seed</span> <span class="v">{policy.model_seed}</span><br>
  <span class="k">Assignment rate</span>
  <span class="v">{policy.propensity:.4f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="footnote" style="margin-top:1.4rem">The two samples share no '
        "user, and the model was locked before the second was opened. The "
        "interval therefore describes the policy, not the search for it.</p>",
        unsafe_allow_html=True,
    )

controls, results = st.columns([1, 2.15], gap="large")

with controls:
    st.markdown('<div class="label">Campaign</div>', unsafe_allow_html=True)
    budget_options = policy.measured_budgets()
    budget_pct = float(
        one_of(
            st.select_slider(
                "Contact budget",
                options=budget_options,
                value=budget_options[0],
                format_func=lambda value: f"{value:g}%",
            )
        )
    )
    campaign_size = st.number_input(
        "Campaign size (users)",
        min_value=1_000,
        max_value=20_000_000,
        value=1_000_000,
        step=100_000,
    )
    st.markdown(
        '<p class="footnote">Only measured budgets are offered. A value between '
        "them would need an interval no sample supports.</p>",
        unsafe_allow_html=True,
    )

campaign_size = int(campaign_size)
projection = policy.expected_outcome(budget_pct / 100.0, n_users=campaign_size)

with results:
    st.markdown('<div class="label">Projected result</div>',
                unsafe_allow_html=True)
    one, two, three = st.columns(3)
    one.metric("Contacted", f"{projection['n_targeted']:,}")
    two.metric(
        f"Incremental {policy.outcome}s",
        f"{projection['incremental_outcome']:+,.0f}",
        help="Against contacting nobody at all.",
    )
    three.metric(
        "Gain over incumbent",
        f"{projection['gain_vs_incumbent']:+,.0f}",
        help="Against ranking users by how likely they are to act.",
    )

    bounds = (
        f"<b>[{projection['gain_ci_lower']:+,.0f}, "
        f"{projection['gain_ci_upper']:+,.0f}]</b>"
    )
    if projection["gain_excludes_zero"]:
        st.markdown(
            f'<div class="verdict clear" style="margin-top:1rem">95% interval '
            f"{bounds} clears zero: the gain over the incumbent is measurable "
            "at this budget.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict unclear" style="margin-top:1rem">95% interval '
            f"{bounds} contains zero: at this budget the data cannot separate "
            "uplift targeting from the incumbent.</div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="label">Why a tight budget</div>', unsafe_allow_html=True)

frame = pd.DataFrame(
    [
        {
            "budget_pct": item.budget_pct,
            "gain": item.gain_rate * campaign_size,
            "lower": item.gain_ci_lower_rate * campaign_size,
            "upper": item.gain_ci_upper_rate * campaign_size,
        }
        for item in policy.budgets
    ]
)

figure, axis = plt.subplots(figsize=(10, 3.5))
figure.patch.set_facecolor("#F7F7F3")
axis.set_facecolor("#F7F7F3")
for _, row in frame.iterrows():
    chosen = abs(row["budget_pct"] - budget_pct) < 1e-9
    clears = row["lower"] > 0 or row["upper"] < 0
    axis.bar(
        row["budget_pct"],
        row["gain"],
        width=2.6,
        color=TEAL if clears else GREY,
        alpha=1.0 if chosen else 0.3,
        zorder=2,
    )
    axis.errorbar(
        row["budget_pct"],
        row["gain"],
        yerr=[[row["gain"] - row["lower"]], [row["upper"] - row["gain"]]],
        fmt="none",
        ecolor=INK if chosen else GREY,
        capsize=5,
        linewidth=1.5,
        zorder=3,
    )
    axis.annotate(
        f"{row['gain']:+,.0f}",
        (row["budget_pct"], row["upper"]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=9.5,
        color=INK if chosen else MUTED,
        fontweight="bold" if chosen else "normal",
    )
axis.axhline(0.0, color=RED, linewidth=1.1, zorder=1)
axis.set_xticks(frame["budget_pct"].to_numpy(dtype=float))
axis.set_xticklabels([f"{value:g}%" for value in frame["budget_pct"]], color=INK)
axis.set_xlabel("Contact budget", color=MUTED, fontsize=10)
axis.set_ylabel(f"Extra {policy.outcome}s vs incumbent", color=MUTED, fontsize=10)
axis.tick_params(colors=MUTED, labelsize=9)
axis.margins(y=0.22)
axis.grid(axis="y", alpha=0.15, color=MUTED)
for spine in ("top", "right"):
    axis.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    axis.spines[spine].set_color(RULE)
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)

st.markdown(
    '<p class="footnote">Teal clears zero, grey does not. Past a fifth of the '
    "list, contacting more is close enough to contacting everyone that ranking "
    "well stops paying.</p>",
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="label">How much rests on one modelling choice</div>',
            unsafe_allow_html=True)

families = load_base_learner_comparison(
    file_stamp(BASE_LEARNER_SUMMARY, BASE_LEARNER_ROWS)
)
candidates = load_all_candidates(file_stamp(BASE_LEARNER_ROWS))

if families is None or candidates is None:
    st.markdown(
        '<p class="footnote">Not measured yet. Run '
        "<code>python scripts/run_base_learner_comparison.py</code>.</p>",
        unsafe_allow_html=True,
    )
else:
    comparison_budget = float(families["budget_pct"].iloc[0])
    st.markdown(
        f'<p class="footnote">A <b>{policy.model_name}</b> is a recipe, and the '
        "prediction model it runs on is a separate choice. The same rule was run "
        "on three of them. Each tab is one run; the last puts them together.</p>",
        unsafe_allow_html=True,
    )

    order = list(families["base_family"])
    tabs = st.tabs(
        [BASE_FAMILY_LABELS.get(name, name) for name in order] + ["Side by side"]
    )

    for tab, family_name in zip(tabs, order, strict=False):
        with tab:
            row = families[families["base_family"] == family_name].iloc[0]
            table = candidates[candidates["base_family"] == family_name]
            table = table.sort_values("Rank")

            st.markdown(
                f'<p class="footnote">{BASE_FAMILY_BLURB.get(family_name, "")}</p>',
                unsafe_allow_html=True,
            )
            picked, lead = st.columns(2)
            picked.metric("This run picked", str(row["Winner"]))
            lead.metric(
                "Lead over second",
                f"{row['Lead over runner-up']:.2f}x",
                help=(
                    "Gap to second place as a share of the winner's own margin "
                    "of error. Under 1.00x is a ranking without a separation."
                ),
            )
            if family_name == "gradient_boosting":
                st.markdown(
                    '<div class="verdict clear">The configuration the campaign '
                    "runs on, and the only one carried through to a confirmatory "
                    "test. The projection at the top of this page comes from "
                    "that test.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="verdict unclear">A development run only. It was '
                    "never carried to the confirmatory sample, so there is no "
                    "projection for it and these figures are not what it would "
                    "deliver.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="label" style="margin-top:1.4rem">All seven '
                "candidates, best first</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Rank": table["Rank"],
                        "Candidate": table["Candidate"],
                        "Extra visits": table["effect_rate"] * campaign_size,
                        "Low": table["lower_rate"] * campaign_size,
                        "High": table["upper_rate"] * campaign_size,
                    }
                ),
                hide_index=True,
                column_config={
                    "Extra visits": st.column_config.NumberColumn(format="%+,.0f"),
                    "Low": st.column_config.NumberColumn(format="%+,.0f"),
                    "High": st.column_config.NumberColumn(format="%+,.0f"),
                },
                **fill_width(st.dataframe),
            )
            clearing = int((table["lower_rate"] > 0).sum())
            st.markdown(
                f'<p class="footnote"><b>{clearing}</b> of <b>{len(table)}</b> '
                "beat the incumbent by more than their own margin of error. "
                f"Scaled to {campaign_size:,} users at the "
                f"{comparison_budget:g}% budget.</p>",
                unsafe_allow_html=True,
            )

    with tabs[-1]:
        shown = pd.DataFrame(
            {
                "Prediction model": families["Prediction model"],
                "Winner": families["Winner"],
                "Extra visits": families["effect_rate"] * campaign_size,
                "Low": families["lower_rate"] * campaign_size,
                "High": families["upper_rate"] * campaign_size,
                "Lead": families["Lead over runner-up"],
            }
        )
        st.dataframe(
            shown,
            hide_index=True,
            column_config={
                "Extra visits": st.column_config.NumberColumn(format="%+,.0f"),
                "Low": st.column_config.NumberColumn(format="%+,.0f"),
                "High": st.column_config.NumberColumn(format="%+,.0f"),
                "Lead": st.column_config.NumberColumn(format="%.2fx"),
            },
            **fill_width(st.dataframe),
        )

        winners = sorted(set(families["Winner"]))
        separated = bool((families["Lead over runner-up"] >= 1.0).any())
        all_positive = bool((shown["Low"] > 0).all())
        if separated:
            message = (
                "At least one prediction model separates its winner from second "
                "place, so the choice of learner is doing measurable work."
            )
            tone = "clear"
        elif all_positive:
            message = (
                f"<b>{len(winners)} models, {len(winners)} different winners</b>, "
                "none separated from second place. The learner is not settled, "
                "but every version still beats the incumbent, so the case does "
                "not rest on having picked the right one."
            )
            tone = "unclear"
        else:
            message = (
                f"<b>{len(winners)} models, {len(winners)} different winners</b>, "
                "and not all of them beat the incumbent. Here the choice matters."
            )
            tone = "unclear"
        st.markdown(
            f'<div class="verdict {tone}" style="margin-top:0.9rem">{message}</div>',
            unsafe_allow_html=True,
        )
        if COMPARISON_FIGURE.exists():
            st.image(str(COMPARISON_FIGURE), **fill_width(st.image))
        st.markdown(
            '<p class="footnote">From the stage that <em>chose</em> the model, on '
            "the audit sample, fixed at the "
            f"{comparison_budget:g}% budget. Compare these against each other, not "
            "against the projection above: a winner flatters itself on the sample "
            "that crowned it, which is why measuring used a different one. "
            "<code>outputs/tables/base_learner_comparison.csv</code></p>",
            unsafe_allow_html=True,
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="label">Build the contact list</div>',
            unsafe_allow_html=True)

pick, size, action = st.columns([2.4, 1.5, 1])
with pick:
    users_path = st.text_input("User file", value=str(DEFAULT_USERS))
with size:
    n_rows = int(
        one_of(
            st.select_slider(
                "Users to rank",
                options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
                value=200_000,
                format_func=lambda value: f"{value:,}",
            )
        )
    )
with action:
    st.markdown('<div style="height:1.85rem"></div>', unsafe_allow_html=True)
    run = st.button("Rank and select", type="primary", **fill_width(st.button))

if run:
    if not Path(users_path).exists():
        st.error(f"No such file: {users_path}")
    else:
        with st.spinner(f"Ranking {n_rows:,} users"):
            scored = rank_users(users_path, n_rows, policy)
        n_targeted = max(1, round(budget_pct / 100.0 * len(scored)))
        target = scored.nlargest(n_targeted, "uplift_score")
        carried = [
            name
            for name in (CRITEO_ROW_ID, "uplift_score")
            if name in target.columns
        ]
        st.markdown(
            f'<p class="footnote">Top <b>{n_targeted:,}</b> of '
            f"<b>{len(scored):,}</b>. The score orders the list; it is not a "
            "promise about any one person.</p>",
            unsafe_allow_html=True,
        )
        preview, download = st.columns([2.6, 1])
        with preview:
            st.dataframe(
                target[carried].head(12),
                **fill_width(st.dataframe),
                hide_index=True,
            )
        with download:
            st.download_button(
                "Download full list (CSV)",
                data=target[carried].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"target_list_{budget_pct:g}pct.csv",
                mime="text/csv",
                **fill_width(st.download_button),
            )
