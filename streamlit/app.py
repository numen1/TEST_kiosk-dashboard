import pandas as pd
import streamlit as st
import plotly.express as px
from haversine import haversine
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

import os
from PIL import Image

# ---------------- Landing Page ----------------
    st.title("🧠 Numen Kiosk Intelligence")
    st.caption("Real-time intelligence for Bitcoin ATM network")

st.markdown(
    """
Welcome to the **Numen Dashboard** — a strategic interface for visualizing and optimizing NBA Kiosk performance.

**Key Features**:
- 📊 Filter by state, performance, and clustering  
- 💰 See live profitability metrics  
- 🔄 Flag underperforming units for redeployment  
- 📦 Export data for operations and board reviews  
"""
)
st.divider()
st.markdown(
    """
    <div style="position: absolute; top: 10px; right: 20px;">
        <img src="https://raw.githubusercontent.com/numen1/TEST_kiosk-dashboard/main/streamlit/assets/mascot.png" 
             title="Numen watches all 👁‍🗨"
             width="60"
             style="opacity: 0.92; border-radius: 8px;"
        />
    </div>
    """,
    unsafe_allow_html=True
)

BREAK_EVEN = 4900
df = pd.read_csv("data/Numen_Kiosk_Dataset.csv")
df["avg_volume"] = df["avg_volume"].clip(lower=0)

# Cluster Detection
def detect_clusters(data, radius=5):
    clustered_flags = []
    coords = list(zip(data["latitude"], data["longitude"]))
    for i, point1 in enumerate(coords):
        count = sum(
            haversine(point1, point2) <= radius
            for j, point2 in enumerate(coords)
            if i != j
        )
        clustered_flags.append(count > 0)
    return clustered_flags

df["is_clustered"] = detect_clusters(df)
df["cluster_label"] = df["is_clustered"].map({True: "C", False: ""})
df["tier"] = df["avg_volume"].apply(lambda v: "High" if v >= 6500 else "Mid" if v >= 4000 else "Low")
df["plot_volume"] = df["avg_volume"]
df["redeploy_flag"] = df["avg_volume"] < 3000

# Sidebar Filters
st.sidebar.header("🎛 Smart Filters")
state_filter = st.sidebar.selectbox("📍 State", ["All"] + sorted(df["state"].unique()))
clustered_only = st.sidebar.checkbox("Only Clustered")
profitable_only = st.sidebar.checkbox("Only Profitable")
unprofitable_only = st.sidebar.checkbox("Only Unprofitable")

# Apply Filters
filtered = df.copy()
if state_filter != "All":
    filtered = filtered[filtered["state"] == state_filter]
if clustered_only:
    filtered = filtered[filtered["is_clustered"]]
if profitable_only:
    filtered = filtered[filtered["avg_volume"] >= BREAK_EVEN]
if unprofitable_only:
    filtered = filtered[filtered["avg_volume"] < BREAK_EVEN]

# Filter label string
active_filters = []
if state_filter != "All":
    active_filters.append(f"State: {state_filter}")
if clustered_only:
    active_filters.append("Only Clustered")
if profitable_only:
    active_filters.append("Only Profitable")
if unprofitable_only:
    active_filters.append("Only Unprofitable")
filter_context = " | ".join(active_filters) if active_filters else "All Kiosks"

# Zoom & Map Center
zoom = 4 if state_filter == "All" else 6
map_center = {
    "lat": filtered["latitude"].mean(),
    "lon": filtered["longitude"].mean()
}

# KPI Calculations
avg_vol = int(filtered["avg_volume"].mean())
avg_pl = avg_vol - BREAK_EVEN
high_perf = filtered[filtered["tier"] == "High"]
low_perf = filtered[filtered["tier"] == "Low"]
total_kiosks = len(filtered)
clustered_count = len(filtered[filtered["is_clustered"]])
redeploy_count = len(filtered[filtered["redeploy_flag"]])
low_perf_pct = (len(low_perf) / total_kiosks * 100) if total_kiosks else 0
loss_total = int((BREAK_EVEN - filtered["avg_volume"]).clip(lower=0).sum())

# 📦 Deployment Overview
st.markdown(f"### 📦 Kiosk Deployment Overview ({filter_context})")
col1, col2, col3 = st.columns(3)
col1.metric("Total Kiosks", total_kiosks)
col2.metric("Clustered Sites", clustered_count)
col3.metric("Redeploy Candidates", redeploy_count)
col4, _ = st.columns(2)
col4.metric("% Low Performers", f"{low_perf_pct:.1f}%")

# 💰 Financial Insights
st.markdown(f"### 💰 Financial Insights ({filter_context})")
col5, col6, col7 = st.columns(3)
col5.metric("Avg Volume", f"${avg_vol:,}", delta_color=("normal" if avg_vol >= BREAK_EVEN else "inverse"))
col6.metric("P/L per Kiosk", f"${avg_pl:+,}", delta_color="normal")
col7.metric("Loss from Unprofitable", f"${loss_total:,}")

# 📊 Network Status
net_result = (
    "✅ Profitable – Look to optimize and grow"
    if avg_vol >= BREAK_EVEN
    else "🚨 Not Profitable – Analyze and redeploy"
)
st.markdown(f"### 📊 Network Status ({filter_context})")
if avg_vol >= BREAK_EVEN:
    st.success(net_result)
else:
    st.error(net_result)

# 🗺️ Kiosk Map
st.subheader(f"🗺️ Kiosk Map ({filter_context})")
fig_map = px.scatter_mapbox(
    filtered,
    lat="latitude",
    lon="longitude",
    hover_name="kiosk_id",
    hover_data=["tier", "avg_volume", "transactions", "host", "location_type", "is_clustered"],
    color="tier",
    size="plot_volume",
    text="cluster_label",
    color_discrete_map={"High": "green", "Mid": "orange", "Low": "darkred"},
    mapbox_style="open-street-map",
    zoom=zoom,
    center=map_center,
    height=600
)
fig_map.update_traces(textposition="top center")
st.plotly_chart(fig_map, use_container_width=True)

# 📍 Regional Density
st.subheader(f"📍 Regional Density ({filter_context})")
st.markdown("""🧠 **Numen Insight**
This heatmap shows kiosk volume concentration across the selected region.
Use this view to detect over-saturation, identify high-potential low-density zones,
and plan future deployments or redeployments with precision.""")

fig_density = px.density_map(
    filtered,
    lat="latitude",
    lon="longitude",
    z="avg_volume",
    radius=20,
    center=map_center,
    zoom=zoom,
    map_style="open-street-map"
)
st.plotly_chart(fig_density, use_container_width=True)

# ---------------- Volume by Performance Tier ----------------
st.subheader(f"📊 Volume Share by Performance Tier ({filter_context})")
st.markdown("🧠 **Numen Insight**")

tier_summary = (
    filtered.groupby("tier")
    .agg(
        total_volume=("avg_volume", "sum"),
        kiosk_count=("kiosk_id", "count"),
        avg_volume=("avg_volume", "mean"),
    )
    .reindex(["High", "Mid", "Low"])
    .fillna(0)
    .astype(int)
)
tier_summary.columns = ["Total Volume ($)", "Kiosk Count", "Avg Volume ($)"]

top_tier = tier_summary["Total Volume ($)"].idxmax()
top_share = (
    tier_summary["Total Volume ($)"].max()
    / tier_summary["Total Volume ($)"].sum()
    * 100
)

total_volume_all = tier_summary["Total Volume ($)"].sum()
colA, colB, colC = st.columns(3)
colA.metric(
    "High Tier Volume",
    f"{(tier_summary.loc['High', 'Total Volume ($)'] / total_volume_all * 100):.1f}%",
)
colB.metric(
    "Mid Tier Volume",
    f"{(tier_summary.loc['Mid', 'Total Volume ($)'] / total_volume_all * 100):.1f}%",
)
colC.metric(
    "Low Tier Volume",
    f"{(tier_summary.loc['Low', 'Total Volume ($)'] / total_volume_all * 100):.1f}%",
)

st.markdown(
    f"""
- Majority of volume comes from **{top_tier}** tier (**{top_share:.1f}%**)
- Use this view to prioritize expansion, renegotiate drag, and rebalance allocation.
"""
)

st.dataframe(tier_summary, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_bar = px.bar(
        tier_summary.reset_index(),
        x="tier",
        y="Total Volume ($)",
        color="tier",
        text="Total Volume ($)",
        color_discrete_map={"High": "green", "Mid": "orange", "Low": "darkred"},
        title="Total Volume by Tier",
        height=400,
    )
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(t=60, b=40),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig_bar.update_traces(
        texttemplate="$%{text:,}", textposition="outside", cliponaxis=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_pie = px.pie(
        names=tier_summary.index,
        values=tier_summary["Total Volume ($)"],
        title="Performance Tier Volume Share",
        hole=0.5,
        color=tier_summary.index,
        color_discrete_map={"High": "green", "Mid": "orange", "Low": "darkred"},
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- Avg Volume by Kiosk ----------------
st.subheader(f"📈 Avg Volume by Kiosk ({filter_context})")
top_kiosk = filtered.loc[filtered["avg_volume"].idxmax()]
avg_bar_vol = int(filtered["avg_volume"].mean())

st.markdown("🧠 **Numen Insight**")
st.markdown(
    f"""
- Average volume per kiosk: **${avg_bar_vol:,}**
- Top performer: `{top_kiosk['kiosk_id']}` – **${int(top_kiosk['avg_volume']):,}/mo**
- Use this to track outliers, gaps, and targeting for upgrades.
"""
)

fig_kiosk_bar = px.bar(
    filtered.sort_values("avg_volume", ascending=False),
    x="kiosk_id",
    y="avg_volume",
    color="tier",
    text="avg_volume",
    color_discrete_map={"High": "green", "Mid": "orange", "Low": "darkred"},
    labels={"avg_volume": "Avg Volume ($)", "kiosk_id": "Kiosk ID"},
)
fig_kiosk_bar.update_traces(texttemplate="$%{text:,}", textposition="outside")
fig_kiosk_bar.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_kiosk_bar, use_container_width=True)

# ---------------- Top 5 Kiosks ----------------
st.markdown(f"### 🏆 Top 5 Kiosks by Volume ({filter_context})")
top5 = filtered.sort_values("avg_volume", ascending=False).head(5)
st.dataframe(top5[["kiosk_id", "avg_volume", "state", "host", "location_type"]])

# ---------------- Redeployment Table ----------------
st.subheader(f"🔄 Kiosks Flagged for Redeployment ({filter_context})")

if "nearest_km" not in filtered.columns and len(filtered) >= 2:
    coords = filtered[["latitude", "longitude"]].to_numpy()
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(coords)
    distances, _ = neigh.kneighbors(coords)
    filtered["nearest_km"] = distances[:, 1]

redeploy_df = filtered[filtered["avg_volume"] < 3000]
redeploy_count = len(redeploy_df)
redeploy_loss = int((BREAK_EVEN - redeploy_df["avg_volume"]).clip(lower=0).sum())
estimated_holding_cost = redeploy_count * 100

st.markdown("🧠 **Numen Insight**")
st.markdown(
    f"""
- **{redeploy_count} kiosks** flagged as underperforming (volume < $3,000/mo)  
- Monthly loss vs breakeven: **${redeploy_loss:,}**  
- Estimated holding cost: **${estimated_holding_cost:,}/mo**  
- Recommend removal, relocation, or renegotiation with host
"""
)

redeploy_cols = [
    col
    for col in [
        "kiosk_id",
        "state",
        "avg_volume",
        "transactions",
        "host",
        "location_type",
        "nearest_km",
    ]
    if col in redeploy_df.columns
]

st.dataframe(redeploy_df[redeploy_cols])

st.download_button(
    label="⬇️ Export Redeploy Table as CSV",
    data=redeploy_df[redeploy_cols].to_csv(index=False),
    file_name="redeploy_kiosks.csv",
    mime="text/csv",
)
