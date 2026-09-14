"""
Lodige Industries - Talent Brand Dashboard (Demo)

Illustrative dashboard built on synthetic data (see generate_demo_data.py).
Purpose: show HR stakeholders and Meta (for API access review) what the
production dashboard will look like once Website, LinkedIn, Facebook and
Instagram accounts are connected. No live accounts are read by this app.

Run: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ---------------------------------------------------------------- constants

PLATFORM_COLOR = {
    "Website": "#2a78d6",
    "LinkedIn": "#eb6834",
    "Facebook": "#1baf7a",
    "Instagram": "#eda100",
}
PLATFORMS = list(PLATFORM_COLOR.keys())

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DATA_PATH = Path(__file__).parent / "data" / "demo_kpis.csv"

st.set_page_config(
    page_title="Lodige Talent Brand Dashboard (Demo)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- minimal styling

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1100px;}
    div[data-testid="stMetric"] {
        background: #fcfcfb;
        border: 1px solid rgba(11,11,11,0.10);
        border-radius: 8px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] {color: #52514e;}
    .demo-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 600;
        color: #52514e;
        background: #f4f3f0;
        border: 1px solid rgba(11,11,11,0.10);
        border-radius: 999px;
        padding: 3px 10px;
        margin-bottom: 6px;
    }
    .roadmap-card {
        border: 1px dashed rgba(11,11,11,0.18);
        border-radius: 8px;
        padding: 12px 14px;
        color: #898781;
        font-size: 13px;
        height: 100%;
    }
    .roadmap-card b {color: #52514e;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- data loading + normalizing


@st.cache_data
def load_raw():
    if not DATA_PATH.exists():
        st.error("Demo data not found. Run: python generate_demo_data.py")
        st.stop()
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def pivot(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    sub = df[df["platform"] == platform]
    return sub.pivot_table(index="date", columns="metric", values="value", aggfunc="sum")


@st.cache_data
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps each platform's own metric names onto one shared shape, the same
    normalization step the real ETL performs: awareness, engagement_actions,
    engagement_rate, outcome. Ratios are recomputed from summed components,
    never averaged, so the shape holds at any aggregation level.
    """
    frames = []

    web = pivot(df, "Website")
    web_out = pd.DataFrame({
        "awareness": web["screen_page_views"],
        "engagement_actions": web["engaged_sessions"],
        "engagement_rate": web["engaged_sessions"] / web["sessions"],
        "outcome": web["apply_page_views"],
    })
    web_out["platform"] = "Website"
    frames.append(web_out)

    li = pivot(df, "LinkedIn")
    li_actions = li[["reactions", "comments", "reposts"]].sum(axis=1)
    li_out = pd.DataFrame({
        "awareness": li["impressions"],
        "engagement_actions": li_actions,
        "engagement_rate": li_actions / li["impressions"],
        "outcome": pd.NA,
    })
    li_out["platform"] = "LinkedIn"
    frames.append(li_out)

    for name in ["Facebook", "Instagram"]:
        p = pivot(df, name)
        actions = p[["likes_reactions", "comments", "shares", "link_clicks"]].sum(axis=1)
        p_out = pd.DataFrame({
            "awareness": p["views"],
            "engagement_actions": actions,
            "engagement_rate": actions / p["views"],
            "outcome": p["leads"],
        })
        p_out["platform"] = name
        frames.append(p_out)

    out = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return out


raw = load_raw()
data = normalize(raw)

# ---------------------------------------------------------------- sidebar

st.sidebar.markdown("**Period**")
period_label = st.sidebar.radio(
    "Period", ["Last 30 days", "Last 60 days", "Last 90 days", "All (120 days)"],
    index=0, label_visibility="collapsed",
)
period_days = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90, "All (120 days)": 120}[period_label]

st.sidebar.markdown("**Platforms**")
selected_platforms = [
    p for p in PLATFORMS
    if st.sidebar.checkbox(p, value=True, key=f"chk_{p}")
]

with st.sidebar.expander("About this data"):
    st.write(
        "Every number on this page is synthetic. It follows the same "
        "structure the live dashboard will use once Website, LinkedIn, "
        "Facebook and Instagram access is granted: Awareness, Engagement "
        "and Outcome per platform, normalized into one shared schema."
    )
    st.caption(
        "Live data would be aggregated and anonymized only. No individual "
        "identities are read, stored, or shown at any point."
    )

if not selected_platforms:
    st.warning("Select at least one platform in the sidebar.")
    st.stop()

end_date = data["date"].max()
start_date = end_date - pd.Timedelta(days=period_days - 1)
prev_start = start_date - pd.Timedelta(days=period_days)
prev_end = start_date - pd.Timedelta(days=1)

cur = data[(data["date"] >= start_date) & (data["platform"].isin(selected_platforms))]
prev = data[(data["date"] >= prev_start) & (data["date"] <= prev_end) & (data["platform"].isin(selected_platforms))]

# ---------------------------------------------------------------- header

st.markdown('<span class="demo-badge">DEMO DATA - NOT LIVE</span>', unsafe_allow_html=True)
st.title("Talent Brand Dashboard")
st.caption(
    "Awareness, engagement and outcome across the talent brand's channels. "
    f"Showing {period_label.lower()}, {start_date.date()} to {end_date.date()}."
)

# ---------------------------------------------------------------- top-line KPIs


def pct_delta(cur_val, prev_val):
    if prev_val in (0, None) or pd.isna(prev_val) or prev_val == 0:
        return None
    return (cur_val - prev_val) / prev_val * 100


awareness_cur, awareness_prev = cur["awareness"].sum(), prev["awareness"].sum()
actions_cur, actions_prev = cur["engagement_actions"].sum(), prev["engagement_actions"].sum()
outcome_cur, outcome_prev = cur["outcome"].sum(), prev["outcome"].sum()

c1, c2, c3 = st.columns(3)
with c1:
    d = pct_delta(awareness_cur, awareness_prev)
    st.metric("Awareness", f"{awareness_cur:,.0f}", f"{d:+.1f}%" if d is not None else None)
with c2:
    d = pct_delta(actions_cur, actions_prev)
    st.metric("Engagement", f"{actions_cur:,.0f}", f"{d:+.1f}%" if d is not None else None)
with c3:
    d = pct_delta(outcome_cur, outcome_prev)
    st.metric("Outcome", f"{outcome_cur:,.0f}", f"{d:+.1f}%" if d is not None else None)

st.caption(
    "Awareness sums page views and impressions. Engagement sums reactions, "
    "comments, shares and clicks. Outcome sums leads and careers apply-page "
    "views. Each tier combines counts of the same kind across platforms; "
    "rates below are kept separate since they are not additive."
)

st.divider()

# ---------------------------------------------------------------- trend chart

st.subheader("Trend")
tier = st.radio("Tier", ["Awareness", "Engagement", "Outcome"], horizontal=True, label_visibility="collapsed")
tier_col = {"Awareness": "awareness", "Engagement": "engagement_actions", "Outcome": "outcome"}[tier]

fig = go.Figure()
skipped = []
for p in selected_platforms:
    series = cur[cur["platform"] == p].sort_values("date")
    if series[tier_col].isna().all():
        skipped.append(p)
        continue
    fig.add_trace(go.Scatter(
        x=series["date"], y=series[tier_col],
        mode="lines", name=p,
        line=dict(color=PLATFORM_COLOR[p], width=2, shape="spline", smoothing=0.3),
        hovertemplate="%{y:,.0f}<extra>" + p + "</extra>",
    ))

fig.update_layout(
    height=340,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(color=INK_SECONDARY, size=13),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    xaxis=dict(showgrid=False, linecolor=GRID),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if skipped:
    st.caption(f"Not tracked at this tier: {', '.join(skipped)}.")

st.divider()

# ---------------------------------------------------------------- per-platform cards

st.subheader("By platform")
cols = st.columns(len(selected_platforms))
for col, p in zip(cols, selected_platforms):
    with col:
        pdata = cur[cur["platform"] == p]
        awareness = pdata["awareness"].sum()
        actions = pdata["engagement_actions"].sum()
        rate = actions / awareness if awareness else 0
        outcome = pdata["outcome"].sum()
        has_outcome = pdata["outcome"].notna().any()

        st.markdown(f"**{p}**")
        spark = go.Figure(go.Scatter(
            x=pdata["date"], y=pdata["awareness"],
            mode="lines", line=dict(color=PLATFORM_COLOR[p], width=2),
            fill="tozeroy", fillcolor=hex_to_rgba(PLATFORM_COLOR[p], 0.1),
            hoverinfo="skip",
        ))
        spark.update_layout(
            height=60, margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False})

        st.metric("Awareness", f"{awareness:,.0f}")
        st.metric("Engagement rate", f"{rate*100:.1f}%")
        st.metric(
            "Outcome", f"{outcome:,.0f}" if has_outcome else "—",
            help=None if has_outcome else "Not available at this tier for LinkedIn organic pages.",
        )

st.divider()

# ---------------------------------------------------------------- audience

st.subheader("Audience")
st.caption(
    "Who is being reached, not just how many. This is the demographic "
    "breakdown the requested API scopes unlock."
)

audience_platforms = [p for p in selected_platforms if p in ("LinkedIn", "Facebook", "Instagram")]

if audience_platforms:
    a_cols = st.columns(len(audience_platforms))
    for a_col, p in zip(a_cols, audience_platforms):
        with a_col:
            st.markdown(f"**{p}**")
            if p == "LinkedIn":
                seniority = pd.read_csv(Path(__file__).parent / "data" / "demo_linkedin_seniority.csv")
                order = ["Entry", "Senior", "Manager", "Director", "VP+"]
                seniority["segment"] = pd.Categorical(seniority["segment"], categories=order, ordered=True)
                seniority = seniority.sort_values("segment")
                ramp = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
                fig_a = go.Figure(go.Bar(
                    x=seniority["value"], y=seniority["segment"], orientation="h",
                    marker_color=ramp, text=[f"{v}%" for v in seniority["value"]],
                    textposition="outside", hovertemplate="%{y}: %{x}%<extra></extra>",
                ))
                fig_a.update_layout(
                    height=220, margin=dict(l=10, r=30, t=10, b=10),
                    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                    font=dict(color=INK_SECONDARY, size=12),
                    xaxis=dict(visible=False, range=[0, seniority["value"].max() * 1.25]),
                    yaxis=dict(showgrid=False, autorange="reversed"),
                )
                st.caption("Share of members reached, by seniority")
            else:
                ag = pd.read_csv(Path(__file__).parent / "data" / "demo_age_gender.csv")
                ag = ag[ag["platform"] == p]
                fig_a = go.Figure()
                for gender, color in [("Female", "#2a78d6"), ("Male", "#eb6834")]:
                    sub = ag[ag["gender"] == gender]
                    fig_a.add_trace(go.Bar(
                        x=sub["age_bracket"], y=sub["value"], name=gender, marker_color=color,
                        hovertemplate="%{x} " + gender + ": %{y}%<extra></extra>",
                    ))
                fig_a.update_layout(
                    height=220, barmode="group",
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                    font=dict(color=INK_SECONDARY, size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor=GRID, ticksuffix="%"),
                )
                st.caption("Reach by age and gender")
            st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------- roadmap

st.subheader("Roadmap: not in this demo")
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(
        '<div class="roadmap-card"><b>JOIN</b><br>Applications per role, source '
        'and stage. Needs an Advanced-plan API token and a data protection '
        'sign-off before go-live.</div>', unsafe_allow_html=True,
    )
with r2:
    st.markdown(
        '<div class="roadmap-card"><b>HubSpot</b><br>Sessions from tracking '
        'URLs and campaign leads. Has a usable API, not yet wired in.</div>',
        unsafe_allow_html=True,
    )
with r3:
    st.markdown(
        '<div class="roadmap-card"><b>Quality of applicant</b><br>A judgement '
        'metric entered by HR, not available from any API.</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------- raw data

with st.expander("Underlying data"):
    st.dataframe(
        cur.sort_values(["platform", "date"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download CSV",
        cur.to_csv(index=False).encode("utf-8"),
        file_name="talent_brand_demo_data.csv",
        mime="text/csv",
    )
