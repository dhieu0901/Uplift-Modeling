# ruff: noqa: E402
"""Streamlit front end for the locked campaign policy.

Run with ``python -m streamlit run app.py`` so the page uses the same
interpreter as the rest of the project.

The page estimates nothing of its own. It loads the policy saved by
``scripts/fit_campaign_policy.py``, ranks whatever users it is given, and
reports the rates the confirmatory test measured, scaled to the campaign size.
Every number shown is traceable to a tracked file under ``outputs/``.
"""

from __future__ import annotations

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


@st.cache_resource(show_spinner=False)
def load_policy() -> CampaignPolicy:
    return CampaignPolicy.load(POLICY_PATH)


@st.cache_data(show_spinner=False)
def rank_users(path: str, n_rows: int, _policy: CampaignPolicy) -> pd.DataFrame:
    columns = [*_policy.feature_columns, CRITEO_ROW_ID]
    frame = pd.read_parquet(path)
    keep = [name for name in columns if name in frame.columns]
    frame = frame[keep].head(n_rows).reset_index(drop=True)
    scored = frame.copy()
    scored["uplift_score"] = _policy.rank(frame)
    return scored


if not POLICY_PATH.exists():
    st.error(
        "No saved policy found. Build one first with "
        "`python scripts/fit_campaign_policy.py`"
    )
    st.stop()

policy = load_policy()

st.markdown(
    f"""
<div class="masthead">
  <div class="eyebrow">Locked policy &nbsp;/&nbsp; {policy.model_name}</div>
  <h1>Who should the campaign contact?</h1>
  <p>The budget covers only part of the list, so users are ranked by how much
  contact is expected to <em>change</em> them, not by how likely they are to act
  anyway. Every figure below comes from a test run once on users this model
  never saw.</p>
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
        "user. The model was chosen before the measurement sample was opened, so "
        "the interval describes the policy rather than the search that found it."
        "</p>",
        unsafe_allow_html=True,
    )

controls, results = st.columns([1, 2.15], gap="large")

with controls:
    st.markdown('<div class="label">Campaign</div>', unsafe_allow_html=True)
    budget_options = policy.measured_budgets()
    budget_pct = st.select_slider(
        "Contact budget",
        options=budget_options,
        value=budget_options[0],
        format_func=lambda value: f"{value:g}%",
    )
    campaign_size = st.number_input(
        "Campaign size (users)",
        min_value=1_000,
        max_value=20_000_000,
        value=1_000_000,
        step=100_000,
    )
    st.markdown(
        '<p class="footnote">Only budgets the locked test measured are offered. '
        "A value between them would need a confidence interval that no sample "
        "supports.</p>",
        unsafe_allow_html=True,
    )

projection = policy.expected_outcome(budget_pct / 100.0, n_users=int(campaign_size))

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
            f"{bounds} clears zero. The gain over the incumbent is measurable "
            "at this budget.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict unclear" style="margin-top:1rem">95% interval '
            f"{bounds} contains zero. At this budget the data cannot separate "
            "uplift targeting from the incumbent, so there is no case for "
            "switching either way.</div>",
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
axis.set_xticks(frame["budget_pct"])
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
    '<p class="footnote">Teal bars are budgets whose interval clears zero; grey '
    "bars are budgets where it does not. Contacting a fifth of everyone is close "
    "enough to contacting everyone that choosing well stops paying, which is why "
    "the recommendation is a tight budget rather than a large one.</p>",
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="label">Build the contact list</div>',
            unsafe_allow_html=True)

pick, size, action = st.columns([2.4, 1.5, 1])
with pick:
    users_path = st.text_input("User file", value=str(DEFAULT_USERS))
with size:
    n_rows = st.select_slider(
        "Users to rank",
        options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
        value=200_000,
        format_func=lambda value: f"{value:,}",
    )
with action:
    st.markdown('<div style="height:1.85rem"></div>', unsafe_allow_html=True)
    run = st.button("Rank and select", type="primary", use_container_width=True)

if run:
    if not Path(users_path).exists():
        st.error(f"No such file: {users_path}")
    else:
        with st.spinner(f"Ranking {n_rows:,} users"):
            scored = rank_users(users_path, int(n_rows), policy)
        n_targeted = max(1, int(round(budget_pct / 100.0 * len(scored))))
        target = scored.nlargest(n_targeted, "uplift_score")
        carried = [
            name
            for name in (CRITEO_ROW_ID, "uplift_score")
            if name in target.columns
        ]
        st.markdown(
            f'<p class="footnote">Selected the top <b>{n_targeted:,}</b> of '
            f"<b>{len(scored):,}</b> ranked users. The score orders the list; it "
            "is not a promise about any one person, and the numbers above "
            "describe the group as a whole.</p>",
            unsafe_allow_html=True,
        )
        preview, download = st.columns([2.6, 1])
        with preview:
            st.dataframe(
                target[carried].head(12),
                use_container_width=True,
                hide_index=True,
            )
        with download:
            st.download_button(
                "Download full list (CSV)",
                data=target[carried].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"target_list_{budget_pct:g}pct.csv",
                mime="text/csv",
                use_container_width=True,
            )
