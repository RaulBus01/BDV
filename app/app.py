from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rcParams
from sklearn.cluster import KMeans
import networkx as nx
from scipy import stats

# ── Paths (portable: CSV sits beside app.py) ────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
CSV_PATH = _APP_DIR / "combined_penalty_with_players_enriched.csv"
IMAGE_PATH = _APP_DIR / "image.png"
GOAL_PNG_PATH = _APP_DIR.parent / "Goal.png"
SQUADS_CSV_PATH = _PROJECT_ROOT / "FinalData" / "world_cup_squadsNew.csv"

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

POSITION_COLORS = {
    "Defender": "#38bdf8",
    "Forward": "#16a34a",
    "Goalkeeper": "#f59e0b",
    "Midfielder": "#818cf8",
}

st.set_page_config(
    page_title="WC 2026 · Penalty Story",
    page_icon="wc2026.svg",
    layout="wide",
)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    # derived
    df["is_goal"] = df["outcome"].str.lower().eq("goal")
    df["zone_id"] = df["zone"].str.lower().map(TEXT_ZONE_TO_NUMERIC)
    df["position_label"] = df["position"].map(POSITION_LABELS).fillna(df["position"])
    df["conversion_rate"] = df["total_scored"] / df["total_attempts"].replace(0, np.nan)
    df["penalty_type"] = df["type"]
    return df


@st.cache_data(show_spinner=False)
def load_squads() -> pd.DataFrame:
    """Load WC 2026 squad list — used to filter GKs in the network tab."""
    return pd.read_csv(SQUADS_CSV_PATH)


def get_wc_gk_names() -> set:
    """Return the set of goalkeeper names from the WC 2026 squads."""
    sq = load_squads()
    return set(sq[sq["Pos."] == "GK"]["Player"].str.strip())


# ── Helpers ──────────────────────────────────────────────────────────────────
def data_coverage_note(df: pd.DataFrame, column: str, label: str = None):
    """Emit an st.caption disclosing how many rows have data for *column*."""
    label = label or column
    total = len(df)
    available = int(df[column].notna().sum())
    pct = available / total * 100 if total else 0
    st.caption(f"Data coverage for <b>{label}</b>: {available:,} / {total:,} "f"({pct:.1f}%)", unsafe_allow_html=True)
   


def insight(text: str):
    st.markdown(f'<div class="insight-box"> {text}</div>', unsafe_allow_html=True)


# ── Inline filters ───────────────────────────────────────────────────────────
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


# ── Goal-zone heatmap drawing ────────────────────────────────────────────────
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


def draw_goal_shot_scatter(df: pd.DataFrame):
    """Scatter-plot actual shot coordinates on the goal image, coloured by outcome
    (or by k-means cluster when *cluster* = True)."""
    if df.empty:
        st.warning("No shot placement data available for current filters.")
        return None
    
    goal_width = 306
    goal_height = 172
    # Scale the points for right displaying
    x_scale = 2.95
    y_scale = 1.34375

    # Convert coordinates to numeric, converting string penalties/NaNs into NaNs
    clean_df = df.copy()
    clean_df['x'] = pd.to_numeric(clean_df['x'], errors='coerce')
    clean_df['y'] = pd.to_numeric(clean_df['y'], errors='coerce')
    
    clean_df = clean_df.dropna(subset=['x', 'y'])
    
    if clean_df.empty:
        st.warning("No valid numerical coordinates available to plot.")
        return None

    clean_df['scaled_x'] = clean_df['x'] * x_scale
    clean_df['scaled_y'] = clean_df['y'] * y_scale

    fig, ax = plt.subplots()
    
    
    image = cv2.imread('image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)


    ax.scatter(clean_df['scaled_x'], clean_df['scaled_y'], c=clean_df['outcome'].map(OUTCOME_COLORS), edgecolors='black', s=100, alpha=0.8)
    plt.title('K-means Clustering of Penalty Shoot Locations')
    plt.xlabel('X coordinate (left to right)')
    plt.ylabel('Y coordinate (top to bottom)')


    plt.xlim(0, goal_width)
    plt.ylim(0, goal_height)

    ax.invert_yaxis()

    # Set plot size
    fig.set_size_inches(10.7, 8.27)

    
    return fig

def tab_overview(df: pd.DataFrame):
    total = len(df)
    goals = int(df["is_goal"].sum())
    conv = goals / total * 100 if total else 0
    players = df["player_name"].nunique()
    nations = df["Nationality"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Penalties", f"{total:,}")
    k2.metric("Goals", f"{goals:,}")
    k3.metric("Conversion", f"{conv:.1f}%")
    k4.metric("Players", f"{players:,}")
    k5.metric("Nations", f"{nations:,}")

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
        type_stats = (
            df.groupby("penalty_type")["is_goal"]
            .agg(attempts="count", goals="sum")
            .reset_index()
        )
        type_stats["misses"] = type_stats["attempts"] - type_stats["goals"]
        type_stats["conversion"] = (type_stats["goals"] / type_stats["attempts"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=type_stats["penalty_type"], y=type_stats["goals"],
            name="Goals", marker_color="#16a34a",
            text=type_stats.apply(
                lambda r: f"{int(r['goals'])} ({r['conversion']:.1f}%)", axis=1
            ),
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            x=type_stats["penalty_type"], y=type_stats["misses"],
            name="Missed / Saved", marker_color="#64748b",
            text=type_stats["misses"].astype(int),
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack", template="plotly_dark",
            margin=dict(t=10, b=0),
            legend=dict(orientation="h", y=1.12),
            yaxis_title="Count",
        )
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
    data_coverage_note(df, "time", "Match minute")
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


    st.subheader("Nationalities — Volume vs Conversion")
    nat = (
        df.groupby("Nationality")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
        .sort_values("attempts", ascending=False)
        
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
        # Removed duplicate penalty type radio — global filter already applied

    data_coverage_note(df, "zone_id", "Zone")
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

    with st.expander("Zone Breakdown Table"):
            zone_tbl = (
                zone_df.groupby("zone")
                .agg(attempts=("outcome", "size"), goals=("is_goal", "sum"))
                .reset_index()
            )
            zone_tbl["conversion"] = (zone_tbl["goals"] / zone_tbl["attempts"] * 100).round(1)
            zone_tbl = zone_tbl.sort_values("attempts", ascending=False)
            st.dataframe(zone_tbl, width='stretch', hide_index=True)

    best = zone_df.groupby("zone").agg(
        attempts=("outcome", "size"), goals=("is_goal", "sum")
    ).reset_index()
    best["conversion"] = (best["goals"] / best["attempts"] * 100).round(1)
    best = best.sort_values("attempts", ascending=False).iloc[0]
    insight(
        f"Most popular zone: <b>{best['zone']}</b> ({int(best['attempts'])} attempts, "
        f"{best['conversion']}% conversion). "
        "Low-left and low-right corners tend to have the highest conversion — "
        "harder for goalkeepers to reach."
    )


    st.markdown("---")
    st.subheader("Shot Placement on Goal Face")
    scatter_df = df.dropna(subset=["x", "y"]).copy()

    if scatter_df.empty:
        st.warning("No shot placement data available for current filters.")
    else:
        fig = draw_goal_shot_scatter(
            scatter_df,
        )
        if fig is not None:
            st.pyplot(fig, width='stretch')
    

def tab_players(df: pd.DataFrame):
    st.subheader("Player Analytics")

    ps = (
        df.groupby(["player_name", "Nationality", "position_label", "club"])
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
        visible[["player_name", "Nationality", "position_label", "club", "attempts", "goals",
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
    data_coverage_note(df, "body_part", "Body part")
    bp_out = (
        df[df["body_part"].notna()]
        .groupby(["body_part", "outcome"])["is_goal"]
        .count()
        .reset_index(name="count")
    )
    if not bp_out.empty:
        fig = px.bar(
            bp_out, x="body_part", y="count", color="outcome",
            color_discrete_map=OUTCOME_COLORS, template="plotly_dark",
            barmode="group", labels={"count": "Penalties"},
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    
def tab_distributions(df: pd.DataFrame):
    st.subheader("Statistical Distributions")

    st.markdown("#### Distribution of Per-Player Conversion Rates")
    ps = (
        df.groupby("player_name")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    ps["conversion"] = (ps["goals"] / ps["attempts"] * 100).round(1)
    min_att = st.slider(
        "Minimum attempts for distribution analysis", 1, 15, 3,
        key="dist_min_att",
    )
    ps_filt = ps[ps["attempts"] >= min_att]

    if ps_filt.empty:
        st.warning("No players match the minimum-attempts filter.")
        return

    mean_conv = ps_filt["conversion"].mean()
    median_conv = ps_filt["conversion"].median()
    std_conv = ps_filt["conversion"].std()
    skewness = ps_filt["conversion"].skew()

    fig = px.histogram(
        ps_filt, x="conversion", nbins=30,
        template="plotly_dark",
        labels={"conversion": "Conversion Rate (%)", "count": "Players"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig.add_vline(x=mean_conv, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Mean {mean_conv:.1f}%")
    fig.add_vline(x=median_conv, line_dash="dot", line_color="#16a34a",
                  annotation_text=f"Median {median_conv:.1f}%")

    # ── Box plot of conversion by position ────────────────────────────────
    st.markdown("---")
    st.markdown("#### Conversion Rate by Position (Box Plot)")
    ps_pos = (
        df.groupby(["player_name", "position_label"])["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    ps_pos["conversion"] = (ps_pos["goals"] / ps_pos["attempts"] * 100).round(1)
    ps_pos = ps_pos[ps_pos["attempts"] >= min_att]

    if not ps_pos.empty:
        fig = px.box(
            ps_pos, x="position_label", y="conversion",
            color="position_label",
            color_discrete_map=POSITION_COLORS,
            template="plotly_dark",
            labels={"conversion": "Conversion Rate (%)", "position_label": "Position"},
            points="outliers",
        )
        overall_mean = ps_pos["conversion"].mean()
        fig.add_hline(y=overall_mean, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Overall mean {overall_mean:.1f}%")
        fig.update_layout(margin=dict(t=30), showlegend=False,
                          title_text="Spread of Conversion Rates by Position")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        # Per-position stats table
        pos_stats = (
            ps_pos.groupby("position_label")["conversion"]
            .agg(["mean", "median", "std", "count"])
            .round(1)
            .reset_index()
        )
        pos_stats.columns = ["Position", "Mean %", "Median %", "Std Dev", "Players"]
        st.dataframe(pos_stats, hide_index=True, width='stretch')

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Correlation Heatmap (Numerical Features)")
    corr_ps = (
        df.groupby("player_name")
        .agg(
            Caps=("Caps", "first"),
            total_attempts=("total_attempts", "first"),
            Intl_Goals=("Goals", "first"),
            attempts_in_view=("outcome", "size"),
            goals_in_view=("is_goal", "sum"),
        )
        .reset_index()
    )
    corr_ps["conversion_rate"] = (
        corr_ps["goals_in_view"] / corr_ps["attempts_in_view"] * 100
    ).round(1)

    corr_cols = ["Caps", "total_attempts", "Intl_Goals", "conversion_rate"]
    corr_matrix = corr_ps[corr_cols].corr().round(3)

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="plotly_dark"
    )
    fig.update_layout(margin=dict(t=30), title_text="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    insight(
        "Strong positive correlations between Caps, attempts, and international goals suggest "
        "experienced players are also prolific — but conversion rate may not follow the same pattern."
    )


def tab_advanced(df: pd.DataFrame):
    st.markdown("### Captain Effect on Conversion")

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

    # ── Shootout sequence analysis (data-driven insight) ──────────────────
    st.markdown("### Shootout Sequence Analysis")
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
            f"Conversion tends to drop in later shootout rounds, possibly due to increased pressure or fatigue. "
        )


def tab_network(df: pd.DataFrame):
    st.subheader("Penalty Taker ↔ Goalkeeper Network")

    wc_gk_names = get_wc_gk_names()

    net_df = df.dropna(subset=["gk_name", "player_name"]).copy()
    net_df = net_df[net_df["gk_name"].str.strip().isin(wc_gk_names)]

    total_with_gk = int(df["gk_name"].notna().sum())
    wc_count = len(net_df)
    gk_matched = net_df["gk_name"].nunique()
    st.caption(
        f" Showing only goalkeepers from the **WC 2026 squads** "
        f"({gk_matched} GKs matched). "
        f"**{wc_count:,}** rows out of {total_with_gk:,} with GK data "
        f"({len(df):,} total in view)."
    )

    if net_df.empty:
        st.warning("No WC 2026 squad goalkeeper data available for current filters.")
        return
    top_n = st.slider(
        "Show top N most-connected takers & GKs",
        5, 50, 15, key="net_top_n",
    )

    edges = (
        net_df.groupby(["player_name", "gk_name"])
        .agg(
            encounters=("outcome", "size"),
            goals=("is_goal", "sum"),
        )
        .reset_index()
    )
    edges["conversion"] = (edges["goals"] / edges["encounters"] * 100).round(1)

    taker_counts = edges.groupby("player_name")["encounters"].sum().nlargest(top_n)
    gk_counts = edges.groupby("gk_name")["encounters"].sum().nlargest(top_n)
    edges_filt = edges[
        edges["player_name"].isin(taker_counts.index)
        | edges["gk_name"].isin(gk_counts.index)
    ]

    if edges_filt.empty:
        st.warning("Not enough data to build a network with current filters.")
        return


    G = nx.Graph()

    taker_stats = (
        net_df[net_df["player_name"].isin(edges_filt["player_name"].unique())]
        .groupby("player_name")["is_goal"]
        .agg(attempts="count", goals="sum")
        .reset_index()
    )
    taker_stats["conversion"] = (taker_stats["goals"] / taker_stats["attempts"] * 100).round(1)
    for _, r in taker_stats.iterrows():
        G.add_node(
            r["player_name"],
            bipartite=0,
            node_type="taker",
            attempts=r["attempts"],
            conversion=r["conversion"],
        )

   
    gk_stats = (
        net_df[net_df["gk_name"].isin(edges_filt["gk_name"].unique())]
        .groupby("gk_name")["is_goal"]
        .agg(faced="count", conceded="sum")
        .reset_index()
    )
    gk_stats["save_rate"] = ((1 - gk_stats["conceded"] / gk_stats["faced"]) * 100).round(1)
    for _, r in gk_stats.iterrows():
        G.add_node(
            r["gk_name"],
            bipartite=1,
            node_type="gk",
            faced=r["faced"],
            save_rate=r["save_rate"],
        )


    for _, r in edges_filt.iterrows():
        if G.has_node(r["player_name"]) and G.has_node(r["gk_name"]):
            G.add_edge(
                r["player_name"], r["gk_name"],
                weight=r["encounters"],
                goals=r["goals"],
                conversion=r["conversion"],
            )

    if G.number_of_edges() == 0:
        st.warning("No edges in the network for current selection.")
        return

    pos = nx.spring_layout(G, k=1.8, seed=42, iterations=50)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="none", mode="lines",
        showlegend=False,
    )

    def make_node_trace(node_type, color_field, cmap_name, symbol, name):
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        nx_vals = [pos[n][0] for n in nodes]
        ny_vals = [pos[n][1] for n in nodes]
        sizes = []
        colors = []
        hover = []
        for n in nodes:
            d = G.nodes[n]
            if node_type == "taker":
                sizes.append(max(8, min(d.get("attempts", 1) * 3, 40)))
                colors.append(d.get("conversion", 50))
                hover.append(
                    f"<b>{n}</b><br>"
                    f"Attempts: {d.get('attempts', '?')}<br>"
                    f"Conversion: {d.get('conversion', '?')}%<br>"
                    f"Connections: {G.degree(n)}"
                )
            else:
                sizes.append(max(8, min(d.get("faced", 1) * 3, 40)))
                colors.append(d.get("save_rate", 50))
                hover.append(
                    f"<b>{n}</b> (GK)<br>"
                    f"Faced: {d.get('faced', '?')}<br>"
                    f"Save rate: {d.get('save_rate', '?')}%<br>"
                    f"Connections: {G.degree(n)}"
                )
        return go.Scatter(
            x=nx_vals, y=ny_vals, mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale=cmap_name,
                cmin=0, cmax=100,
                symbol=symbol,
                line=dict(width=1, color="black"),
                colorbar=dict(
                    title=color_field,
                    x=1.05,
                    len=0.4,
                    y=0.75 if node_type == "taker" else 0.25,
                    yanchor="middle",
                ),
            ),
            text=[n.split()[-1] for n in nodes],  # surname only for readability
            textposition="top center",
            textfont=dict(size=8, color="black"),
            hovertext=hover,
            hoverinfo="text",
            name=name,
        )

    taker_trace = make_node_trace("taker", "Conv %", "Greens", "circle", "Takers")
    gk_trace = make_node_trace("gk", "Save %", "Blues", "diamond", "Goalkeepers")

    fig = go.Figure(data=[edge_trace, taker_trace, gk_trace])
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=30, b=40, l=20, r=100),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Bipartite Network: Penalty Takers (●) ↔ Goalkeepers (◆)",
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() else 0
    c3.metric("Avg Degree", f"{avg_deg:.1f}")
    components = nx.number_connected_components(G)
    c4.metric("Components", components)


    with st.expander("Degree Distribution Analysis"):
        degrees = [d for _, d in G.degree()]
        fig_deg = px.histogram(
            x=degrees, nbins=max(5, max(degrees) - min(degrees)),
            template="plotly_dark",
            labels={"x": "Degree (connections)", "y": "Count"},
            color_discrete_sequence=["#818cf8"],
        )
        fig_deg.update_layout(
            margin=dict(t=30),
            title_text="Degree Distribution",
            bargap=0.1,
        )
        st.plotly_chart(fig_deg, use_container_width=True, config=plotly_config)

    with st.expander(" Top Taker ↔ GK Matchups"):
        top_edges = edges_filt.sort_values("encounters", ascending=False).head(20)
        st.dataframe(
            top_edges[["player_name", "gk_name", "encounters", "goals", "conversion"]],
            hide_index=True, width='stretch',
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
        "Overview",
        "Goal Map",
        "Players",
        "Distributions",
        "Advanced Insights",
        "Network",
    ])

    with tabs[0]:
        tab_overview(filtered)
    with tabs[1]:
        tab_goal_map(filtered)
    with tabs[2]:
        tab_players(filtered)
    with tabs[3]:
        tab_distributions(filtered)
    with tabs[4]:
        tab_advanced(filtered)
    with tabs[5]:
        tab_network(filtered)



if __name__ == "__main__":
    main()
