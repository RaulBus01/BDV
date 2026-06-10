from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "app" / "combined_penalty_with_players_enriched.csv"
IMAGE_PATH = ROOT / "app" / "image.png"

plotly_config = {
    'displayModeBar': False,
    'editable': False
}   

OUTCOME_COLORS = {
    "goal": "#16a34a",
    "save": "#2563eb",
    "miss": "#dc2626",
    "post": "#f59e0b",
}

ZONE_TO_GRID = {
    1: (3, 1), 2: (3, 2), 3: (3, 3),
    4: (2, 1), 5: (2, 3),
    6: (1, 1), 7: (1, 2), 8: (1, 3),
    9: (3, 0), 10: (0, 0),
    11: [(0, 2), (0, 1)],
    12: (0, 4), 13: (3, 4),
}

TEXT_ZONE_TO_NUMERIC = {
    "low-left": 1, "low-centre": 2, "low-center": 2, "low-right": 3,
    "left": 4, "right": 5,
    "high-left": 6, "high-centre": 7, "high-center": 7, "high-right": 8,
    "close-left": 9, "close-high-left": 10,
    "high": 11, "close-high": 11,
    "close-high-right": 12, "close-right": 13,
}

POSITION_LABELS = {"D": "Defender", "F": "Forward", "G": "Goalkeeper", "M": "Midfielder"}

st.set_page_config(
    page_title="WC 2026 · Penalty Story",
    page_icon="wc2026.svg",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    # derived
    df["is_goal"] = df["outcome"].str.lower().eq("goal")
    df["zone_id"] = df["zone"].str.lower().map(TEXT_ZONE_TO_NUMERIC)
    df["position_label"] = df["position"].map(POSITION_LABELS).fillna(df["position"])
    df["conversion_rate"] = df["total_scored"] / df["total_attempts"].replace(0, np.nan)
    df["penalty_type"] = df["type"].fillna("unknown")
    return df


def inline_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=True):
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns([1, 1, 1, 1])
        with row1_c1:
            all_groups = sorted(df["Group"].dropna().unique())
            sel_groups = st.multiselect("WC Group", all_groups)

        with row1_c2:
            if sel_groups:
                nation_pool = sorted(
                    df[df["Group"].isin(sel_groups)]["Nationality"].dropna().unique()
                )
            else:
                nation_pool = sorted(df["Nationality"].dropna().unique())
            sel_nations = st.multiselect("Nationality", nation_pool)

        with row1_c3:
            all_pos = sorted(df["position"].dropna().unique())
            sel_pos = st.multiselect(
                "Position", all_pos,
                format_func=lambda p: POSITION_LABELS.get(p, p),
            )

        with row1_c4:
            player_pool_df = df.copy()
            if sel_groups:
                player_pool_df = player_pool_df[player_pool_df["Group"].isin(sel_groups)]
            if sel_nations:
                player_pool_df = player_pool_df[player_pool_df["Nationality"].isin(sel_nations)]
            if sel_pos:
                player_pool_df = player_pool_df[player_pool_df["position"].isin(sel_pos)]
            player_options = ["— All players —"] + sorted(
                player_pool_df["player_name"].dropna().unique().tolist()
            )
            sel_player = st.selectbox(
                "Player",
                player_options,
                index=0,
                help="Type to search — suggestions update based on group / nationality / position filters above.",
            )

        row2_c1, row2_c2, row2_c3 = st.columns([1, 1, 2])

        with row2_c1:
            type_choice = st.radio(
                "Penalty Type",
                ["All", "match_penalty", "shootout"],
                horizontal=True,
            )

        with row2_c2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            captain_only = st.checkbox("Captains only")

        with row2_c3:
            min_a = int(df["total_attempts"].min())
            max_a = int(df["total_attempts"].max())
            att_range = st.slider("Total attempts", min_a, max_a, (min_a, max_a))

    flt = df.copy()
    if sel_groups:
        flt = flt[flt["Group"].isin(sel_groups)]
    if sel_nations:
        flt = flt[flt["Nationality"].isin(sel_nations)]
    if sel_pos:
        flt = flt[flt["position"].isin(sel_pos)]
    if type_choice != "All":
        flt = flt[flt["penalty_type"] == type_choice]
    if captain_only:
        flt = flt[flt["Is_Captain"] == 1]
    if sel_player and sel_player != "— All players —":
        flt = flt[flt["player_name"] == sel_player]
    flt = flt[flt["total_attempts"].between(*att_range)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Penalties in view", f"{len(flt):,}")
    m2.metric("Players in view", f"{flt['player_name'].nunique():,}")
    m3.metric("Nations in view", f"{flt['Nationality'].nunique():,}")

    return flt


_UNCOLORED_CELLS = {(0, 1), (0, 3), (1, 0), (2, 0)}


def draw_goal_plot(zones: dict, title: str, colors: bool = True):
    image = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
    grid_values = {(i, j): 0 for i in range(4) for j in range(5)}

    for zone, value in zones.items():
        if pd.isna(zone) or pd.isna(value):
            continue
        zone = int(zone)
        if zone not in ZONE_TO_GRID:
            continue
        grid_ref = ZONE_TO_GRID[zone]
        if zone == 11:
            for gp in grid_ref:
                grid_values[gp] = value
        else:
            grid_values[grid_ref] = value

    # fixups
    if grid_values[(1, 0)]:
        grid_values[(0, 0)] += grid_values[(1, 0)]; grid_values[(1, 0)] = 0
    if grid_values[(2, 0)]:
        grid_values[(3, 0)] += grid_values[(2, 0)]; grid_values[(2, 0)] = 0
    if grid_values[(0, 1)]:
        if colors:
            grid_values[(0, 2)] = grid_values[(0, 1)]; grid_values[(0, 1)] = 0
        else:
            grid_values[(0, 1)] = 0
    if grid_values[(0, 3)]:
        grid_values[(0, 1)] += grid_values[(0, 3)]; grid_values[(0, 3)] = 0

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.imshow(image)
    rows, cols = 4, 5

    for i in range(1, rows):
        if i not in {1, 3}:
            ax.axhline(y=i * image.shape[0] // rows, color="#1d4ed8", linewidth=2)
    for j in range(1, cols):
        if j not in {2, 3}:
            ax.axvline(x=j * image.shape[1] // cols, color="#1d4ed8", linewidth=2)

    ax.axhline(y=image.shape[0] // rows, xmin=1/cols, xmax=4/cols, color="#1d4ed8", linewidth=2)
    ax.axhline(y=3*image.shape[0] // rows, xmin=1/cols, xmax=4/cols, color="#1d4ed8", linewidth=2)
    ax.axvline(x=2*image.shape[1] // cols, ymin=0/rows, ymax=3/rows, color="#1d4ed8", linewidth=2)
    ax.axvline(x=3*image.shape[1] // cols, ymin=0/rows, ymax=3/rows, color="#1d4ed8", linewidth=2)

    values = [v for v in grid_values.values() if not pd.isna(v)]
    vmax = max(values) + 1 if values else 1
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.colormaps["plasma"]

    for (row, col), value in grid_values.items():
        color = cmap(norm(value))
        x = col * image.shape[1] // cols + image.shape[1] // (2 * cols)
        y_px = row * image.shape[0] // rows + image.shape[0] // (2 * rows)
        if (row, col) in {(0, 0), (2, 0)}:
            y_px = (row + 0.5) * image.shape[0] // rows
        elif (row, col) == (0, 1):
            x = col * image.shape[1] // cols + image.shape[1] // cols

        facecolor = color if colors and value > 0 else "none"
        rect = plt.Rectangle(
            (col * image.shape[1] // cols, row * image.shape[0] // rows),
            image.shape[1] // cols, image.shape[0] // rows,
            linewidth=1, edgecolor="#111827",
            facecolor=facecolor, alpha=0.76 if colors and value > 0 else 1,
        )
        ax.add_patch(rect)
        if value > 0:
            lbl = f"{value:.1f}%" if isinstance(value, float) and value % 1 else str(int(value))
            ax.text(x, y_px, lbl, color="white", fontsize=14, fontweight="bold",
                    ha="center", va="center")

    ax.set_title(title, fontsize=15, pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def tab_overview(df: pd.DataFrame):
    total = len(df)
    goals = int(df["is_goal"].sum())
    conv = goals / total * 100 if total else 0
    players = df["player_name"].nunique()
    nations = df["Nationality"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("⚽ Penalties", f"{total:,}")
    k2.metric("✅ Goals", f"{goals:,}")
    k3.metric("📈 Conversion", f"{conv:.1f}%")
    k4.metric("👤 Players", f"{players:,}")
    k5.metric("🌍 Nations", f"{nations:,}")

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Penalty Outcomes")
        oc = df["outcome"].value_counts().reset_index()
        oc.columns = ["outcome", "count"]
        oc["pct"] = (oc["count"] / oc["count"].sum() * 100).round(1)
        fig = px.bar(
            oc, x="outcome", y="count", color="outcome",
            color_discrete_map=OUTCOME_COLORS,
            text=oc["pct"].map(lambda v: f"{v}%"),
            template="plotly_dark",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    with c2:
        st.subheader("Match vs Shootout")
        tc = df["penalty_type"].value_counts().reset_index()
        tc.columns = ["type", "count"]
        fig = px.pie(tc, values="count", names="type",
                     color_discrete_sequence=["#38bdf8", "#818cf8"],
                     template="plotly_dark", hole=0.42)
        fig.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    insight(
        f"Overall conversion is <b>{conv:.1f}%</b>. Saves account for "
        f"<b>{df['outcome'].eq('save').mean()*100:.1f}%</b> of all attempts, "
        f"while missed/post penalties together represent "
        f"<b>{df['outcome'].isin(['miss','post']).mean()*100:.1f}%</b>."
    )

    st.subheader("Conversion Rate: Match Penalties vs Shootout")
    type_conv = (
        df.groupby("penalty_type")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    type_conv["conversion"] = (type_conv["goals"] / type_conv["attempts"] * 100).round(2)
    fig = px.bar(
        type_conv, x="penalty_type", y="conversion", color="penalty_type",
        color_discrete_sequence=["#38bdf8", "#818cf8"],
        text=type_conv["conversion"].map(lambda v: f"{v:.1f}%"),
        template="plotly_dark",
        labels={"conversion": "Conversion %", "penalty_type": "Type"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=0), yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    st.subheader("Conversion Rate by WC Group")
    grp = (
        df.groupby("Group")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    grp["conversion"] = (grp["goals"] / grp["attempts"] * 100).round(1)
    fig = px.bar(
        grp.sort_values("Group"), x="Group", y="conversion",
        color="conversion", color_continuous_scale="Blues",
        text=grp.sort_values("Group")["conversion"].map(lambda v: f"{v:.1f}%"),
        template="plotly_dark",
        labels={"conversion": "Conversion %"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


    st.subheader("Conversion by Match Minute")
    td = df.dropna(subset=["time"]).copy()
    if not td.empty:
        td["minute_bucket"] = (td["time"].astype(float) // 15 * 15).astype(int)
        bk = (
            td.groupby("minute_bucket")["is_goal"]
            .agg(attempts="count", goals="sum")
            .reset_index()
        )
        bk["conversion"] = (bk["goals"] / bk["attempts"] * 100).round(1)
        fig = px.line(
            bk, x="minute_bucket", y="conversion", markers=True,
            template="plotly_dark",
            labels={"minute_bucket": "Minute (15-min bucket)", "conversion": "Conversion %"},
        )
        fig.update_traces(line_color="#38bdf8", marker_color="#818cf8", marker_size=8)
        fig.add_hline(y=conv, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Overall avg {conv:.1f}%")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)


    st.subheader("Top 15 Nationalities — Volume vs Conversion")
    nat = (
        df.groupby("Nationality")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
        .sort_values("attempts", ascending=False)
        .head(15)
    )
    nat["conversion"] = (nat["goals"] / nat["attempts"] * 100).round(1)
    fig = px.scatter(
        nat, x="attempts", y="conversion", size="attempts",
        color="Nationality", text="Nationality",
        template="plotly_dark",
        labels={"conversion": "Conversion %", "attempts": "Total Attempts"},
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=conv, line_dash="dash", line_color="#f59e0b")
    fig.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    insight(
        "Bubble size = total attempts. Nations above the dashed line convert above average. "
        "High-volume nations don't always lead in efficiency — volume and quality diverge."
    )

def tab_goal_map(df: pd.DataFrame):
    st.subheader("🎯 Goal Zone Heatmap")

    c1, c2 = st.columns([3, 1])
    with c2:
        mode = st.radio("Metric", ["Attempts", "Goals", "Conversion %"], index=0)
        ptype_filter = st.radio("Penalty type", ["All", "match_penalty", "shootout"], index=0)

    if ptype_filter != "All":
        zone_df = df[df["penalty_type"] == ptype_filter].dropna(subset=["zone_id"])
    else:
        zone_df = df.dropna(subset=["zone_id"])

    if zone_df.empty:
        st.warning("No zone data for current filters.")
        return

    attempts_s = zone_df["zone_id"].value_counts()
    goals_s = zone_df[zone_df["is_goal"]]["zone_id"].value_counts()

    if mode == "Attempts":
        zones_dict = attempts_s.to_dict()
    elif mode == "Goals":
        zones_dict = goals_s.to_dict()
    else:
        zones_dict = ((goals_s / attempts_s) * 100).round(1).dropna().to_dict()

    with c1:
        fig = draw_goal_plot(zones_dict, f"Zone — {mode}", colors=True)
        st.pyplot(fig, width='stretch')

    # zone table
    st.markdown("#### Zone Breakdown Table")
    zone_tbl = (
        zone_df.groupby("zone")
        .agg(attempts=("outcome", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    zone_tbl["conversion"] = (zone_tbl["goals"] / zone_tbl["attempts"] * 100).round(1)
    zone_tbl = zone_tbl.sort_values("attempts", ascending=False)
    st.dataframe(zone_tbl, width='stretch', hide_index=True)

    best = zone_tbl.iloc[0]
    insight(
        f"Most popular zone: <b>{best['zone']}</b> ({int(best['attempts'])} attempts, "
        f"{best['conversion']}% conversion). "
        "Low-left and low-right corners tend to have the highest conversion — "
        "harder for goalkeepers to reach."
    )

def tab_players(df: pd.DataFrame):
    st.subheader("👤 Player Analytics")

    ps = (
        df.groupby(["player_name", "Nationality", "position", "club"])
        .agg(
            attempts=("outcome", "size"),
            goals=("is_goal", "sum"),
            Caps=("Caps", "first"),
            Intl_Goals=("Goals", "first"),
            Is_Captain=("Is_Captain", "first"),
            most_used_zone=("zone", lambda s: s.mode().iloc[0] if not s.mode().empty else "—"),
            body_part=("body_part", lambda s: s.mode().iloc[0] if not s.mode().dropna().empty else "—"),
        )
        .reset_index()
    )
    ps["conversion"] = (ps["goals"] / ps["attempts"] * 100).round(1)

    min_att = st.slider("Minimum attempts", 1, int(ps["attempts"].max()), 3)
    visible = ps[ps["attempts"] >= min_att].sort_values(
        ["conversion", "attempts"], ascending=[False, False]
    )

    st.dataframe(
        visible[["player_name", "Nationality", "position", "club", "attempts", "goals",
                 "conversion", "Caps", "Intl_Goals", "most_used_zone", "body_part", "Is_Captain"]],
        width='stretch', hide_index=True
    )
    st.markdown("#### Top 20 Players by Conversion Rate")
    top = visible.head(20).sort_values("conversion")
    fig = px.bar(
        top, x="conversion", y="player_name", orientation="h",
        color="attempts", color_continuous_scale="Blues",
        template="plotly_dark",
        labels={"conversion": "Conversion %", "player_name": ""},
        text=top["conversion"].map(lambda v: f"{v}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(t=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    st.markdown("#### Body Part Used vs Outcome")
    c1, c2 = st.columns(2)
    with c1:
        bp_out = (
            df[df["body_part"].notna()]
            .groupby(["body_part", "outcome"])["is_goal"]
            .count()
            .reset_index(name="count")
        )
        fig = px.bar(
            bp_out, x="body_part", y="count", color="outcome",
            color_discrete_map=OUTCOME_COLORS, template="plotly_dark",
            barmode="group", labels={"count": "Penalties"},
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    st.markdown("#### International Caps vs Conversion Rate (scatter)")
    sub = visible[visible["attempts"] >= 3]
    fig = px.scatter(
        sub, x="Caps", y="conversion", size="attempts",
        color="position", template="plotly_dark",
        hover_data=["player_name", "club", "attempts"],
        labels={"conversion": "Conversion %"},
        color_discrete_map={"D": "#38bdf8", "F": "#16a34a", "G": "#f59e0b", "M": "#818cf8"},
    )
    if len(sub) > 5:
        m, b, *_ = np.polyfit(sub["Caps"].fillna(0), sub["conversion"].fillna(0), 1, full=False)
        x_line = np.linspace(sub["Caps"].min(), sub["Caps"].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b, mode="lines",
            name="Trend", line=dict(color="#f59e0b", dash="dash")
        ))
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    insight(
        "Each bubble represents a player (size = attempts). "
        "The trend line shows whether international experience correlates with conversion efficiency."
    )



def tab_advanced(df: pd.DataFrame):
    st.markdown("### 🎖️ Captain Effect on Conversion")

    cap_grp = (
        df.groupby("Is_Captain")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    cap_grp["conversion"] = (cap_grp["goals"] / cap_grp["attempts"] * 100).round(2)
    cap_grp["label"] = cap_grp["Is_Captain"].map({0: "Non-Captain", 1: "Captain"})

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.bar(
            cap_grp, x="label", y="conversion",
            color="label", text=cap_grp["conversion"].map(lambda v: f"{v:.1f}%"),
            color_discrete_map={"Captain": "#f59e0b", "Non-Captain": "#38bdf8"},
            template="plotly_dark",
            labels={"conversion": "Conversion %", "label": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10), yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    with c2:
        cap_goals = int(df[df["Is_Captain"] == 1]["is_goal"].sum())
        cap_miss  = int((df["Is_Captain"] == 1).sum()) - cap_goals
        non_goals = int(df[df["Is_Captain"] == 0]["is_goal"].sum())
        non_miss  = int((df["Is_Captain"] == 0).sum()) - non_goals

        contingency = [[cap_goals, cap_miss], [non_goals, non_miss]]
        has_both_groups = (cap_goals + cap_miss) > 0 and (non_goals + non_miss) > 0
        no_zero_cells   = all(v > 0 for row in contingency for v in row)

        cap_rows  = cap_grp[cap_grp["Is_Captain"] == 1]
        non_rows  = cap_grp[cap_grp["Is_Captain"] == 0]
        cap_conv  = cap_rows["conversion"].values[0] if not cap_rows.empty else 0.0
        non_conv  = non_rows["conversion"].values[0] if not non_rows.empty else 0.0

        if has_both_groups and no_zero_cells:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            result_md = (
                f"- χ² = **{chi2:.3f}**, p-value = **{p_val:.4f}**, df = {dof}\n"
                + (
                    "- ✅ **Statistically significant** at α=0.05 — captains and "
                    "non-captains differ in conversion."
                    if p_val < 0.05 else
                    "- ❌ **Not significant** at α=0.05 — the difference may be by chance."
                )
            )
        else:
            result_md = (
                "_χ² test cannot be computed for the current filter selection "
                "(one group has no attempts, or all penalties have the same outcome). "
                "Try widening the filters._"
            )

        st.markdown(f"""
**χ² hypothesis test** (H₀: Captaincy has no effect on conversion):

| | Captain | Non-Captain |
|---|---|---|
| **Attempts** | {cap_goals + cap_miss:,} | {non_goals + non_miss:,} |
| **Goals** | {cap_goals:,} | {non_goals:,} |
| **Conversion** | {cap_conv:.1f}% | {non_conv:.1f}% |

{result_md}
    """)
    st.markdown("### 🔢 Shootout Sequence Analysis")
    seq_df = df[(df["penalty_type"] == "shootout") & df["sequence"].notna()].copy()
    if not seq_df.empty:
        seq = (
            seq_df.groupby("sequence")["is_goal"]
            .agg(attempts="count", goals="sum")
            .reset_index()
        )
        seq["conversion"] = (seq["goals"] / seq["attempts"] * 100).round(1)
        seq = seq[seq["sequence"] <= 10]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=seq["sequence"], y=seq["attempts"], name="Attempts",
                   marker_color="#334155", opacity=0.7), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=seq["sequence"], y=seq["conversion"], name="Conversion %",
                       mode="lines+markers", line=dict(color="#38bdf8", width=2),
                       marker=dict(size=8, color="#818cf8")), secondary_y=True
        )
        fig.update_layout(template="plotly_dark", margin=dict(t=10),
                          xaxis_title="Shootout Kick Number",
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Attempts", secondary_y=False)
        fig.update_yaxes(title_text="Conversion %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        insight(
            "In shootouts, the first 5 kickers have generally higher conversion rates than those after them. " 
            "Pressure, fatigue and even injuries may impact later kickers."
        )

    st.markdown("### 📈 Experience vs Penalty Volume")
    ps = (
        df.groupby("player_name")
        .agg(attempts=("outcome", "size"), Caps=("Caps", "first"),
             goals=("is_goal", "sum"), position=("position", "first"))
        .reset_index()
    )
    ps["conversion"] = (ps["goals"] / ps["attempts"] * 100).round(1)

    fig = px.scatter(
        ps, x="Caps", y="attempts", color="position",
        size="goals", opacity=0.7, template="plotly_dark",
        color_discrete_map={"D": "#38bdf8", "F": "#16a34a", "G": "#f59e0b", "M": "#818cf8"},
        hover_data=["player_name", "conversion"],
        labels={"Caps": "International Caps", "attempts": "Penalty Attempts"},
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    if len(ps) > 5:
        r, p = stats.pearsonr(ps["Caps"].fillna(0), ps["attempts"].fillna(0))
        insight(
            f"Pearson r (Caps vs Penalty Attempts) = <b>{r:.3f}</b>, p = {p:.4f}. "
            + ("Significant positive correlation — experienced players take more penalties."
               if p < 0.05 else "Weak or no significant correlation between experience and attempts.")
        )
def main():
    st.image("wc2026.svg", width=200)
    st.markdown(
        "### Penalty Story: WC 2026 Big Data Visualization\n"
    )

    df = load_data()
    filtered = inline_filters(df)

    if filtered.empty:
        st.warning("No penalties match the current filters. Adjust the filters above.")
        return

    tabs = st.tabs([
        "📊 Overview",
        "🎯 Goal Map",
        "👤 Players",
        "🔬 Advanced Insights",
    ])

    with tabs[0]:
        tab_overview(filtered)
    with tabs[1]:
        tab_goal_map(filtered)
    with tabs[2]:
        tab_players(filtered)
    with tabs[3]:
        tab_advanced(filtered)



if __name__ == "__main__":
    main()
