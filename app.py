import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="SPOTInsight AI | KnightHacks 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polish
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a5f; }
    .sub-header { font-size: 1.1rem; color: #4a6fa5; margin-bottom: 1rem; }
    .metric-card { background: #f0f4f8; padding: 1rem; border-radius: 8px; }
    .critical { color: #d32f2f; font-weight: bold; }
    .high { color: #f57c00; font-weight: bold; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    missions = pd.read_csv(DATA_DIR / "missions.csv")
    actions = pd.read_csv(DATA_DIR / "actions.csv")
    readings = pd.read_csv(DATA_DIR / "readings.csv")
    anomalies = pd.read_csv(DATA_DIR / "anomalies.csv")
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    
    # Enrich
    anomalies = anomalies.merge(assets[["asset_id", "asset_name", "asset_type", "criticality"]], on="asset_id", how="left")
    anomalies = anomalies.merge(locations[["location_id", "location_name", "x_coord", "y_coord", "zone_type"]], on="location_id", how="left")
    readings = readings.merge(assets[["asset_id", "asset_name", "criticality"]], on="asset_id", how="left")
    
    return missions, actions, readings, anomalies, locations, assets

missions, actions, readings, anomalies, locations, assets = load_data()

# Sidebar
st.sidebar.title("SPOTInsight AI")
st.sidebar.markdown("**Gemini-Powered SPOT Dashboard**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🗺️ Map & Locations",
    "⚠️ Anomaly Triage",
    "📊 Performance Comparison",
    "📈 Readings Explorer",
    "🤖 Gemini Assistant",
    "🌍 Community Impact"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Local Context**")
st.sidebar.info("Central Florida (Lake County / Eustis) environmental protection via reliable energy asset inspection.")

# Header
st.markdown('<p class="main-header">SPOTInsight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Turning Boston Dynamics SPOT mission data into clear, actionable intelligence for operators — with Gemini explanations and measurable community environmental impact.</p>', unsafe_allow_html=True)

if page == "🏠 Overview":
    st.header("Operational Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Missions", len(missions))
    with col2:
        st.metric("Total Actions", len(actions))
    with col3:
        st.metric("Anomalies Detected", len(anomalies))
    with col4:
        critical_count = len(anomalies[anomalies["severity"].isin(["HIGH", "CRITICAL"])])
        st.metric("High/Critical Anomalies", critical_count, delta="- needs review" if critical_count > 0 else None)
    
    st.subheader("Mission Status by Scenario")
    status_df = missions.groupby(["scenario", "mission_status"]).size().reset_index(name="count")
    fig = px.bar(status_df, x="scenario", y="count", color="mission_status", barmode="group",
                 color_discrete_map={"COMPLETED": "#2e7d32", "FAILED": "#c62828"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Missions")
    display_missions = missions[["mission_id", "mission_name", "scenario", "configuration_version", 
                                 "start_time", "duration_min", "mission_status", "end_battery_pct"]].copy()
    display_missions["start_time"] = pd.to_datetime(display_missions["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_missions.sort_values("start_time", ascending=False), use_container_width=True, height=300)

elif page == "🗺️ Map & Locations":
    st.header("Facility Map — Waypoints & Anomalies")
    st.caption("Synthetic coordinates for visualization. Color = highest anomaly severity at location.")
    
    # Aggregate severity per location
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    anomalies["sev_num"] = anomalies["severity"].map(sev_map).fillna(0)
    loc_agg = anomalies.groupby("location_id").agg(
        max_severity=("sev_num", "max"),
        anomaly_count=("anomaly_id", "count"),
        x=("x_coord", "first"),
        y=("y_coord", "first"),
        location_name=("location_name", "first")
    ).reset_index()
    
    # Merge all locations
    loc_full = locations.merge(loc_agg[["location_id", "max_severity", "anomaly_count"]], on="location_id", how="left")
    loc_full["max_severity"] = loc_full["max_severity"].fillna(0)
    loc_full["anomaly_count"] = loc_full["anomaly_count"].fillna(0)
    
    color_map = {0: "#90a4ae", 1: "#66bb6a", 2: "#ffa726", 3: "#ef5350", 4: "#b71c1c"}
    loc_full["color"] = loc_full["max_severity"].map(color_map)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loc_full["x_coord"], y=loc_full["y_coord"],
        mode="markers+text",
        marker=dict(size=loc_full["anomaly_count"] * 4 + 12, color=loc_full["color"], line=dict(width=1, color="white")),
        text=loc_full["location_id"],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Anomalies: %{customdata[1]}<extra></extra>",
        customdata=loc_full[["location_name", "anomaly_count"]].values
    ))
    fig.update_layout(
        title="SPOT Inspection Waypoints (size = anomaly count, color = severity)",
        xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
        height=600, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(loc_full[["location_id", "location_name", "zone_type", "anomaly_count", "max_severity"]], use_container_width=True)

elif page == "⚠️ Anomaly Triage":
    st.header("Anomaly Triage Queue")
    st.caption("Prioritized by severity and asset criticality. Gemini can explain any item.")
    
    # Priority score
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    crit_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    anomalies["priority"] = anomalies["severity"].map(sev_map).fillna(1) * anomalies["criticality"].map(crit_map).fillna(1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sev_filter = st.multiselect("Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
    with col2:
        review_only = st.checkbox("Operator Review Required Only", value=True)
    with col3:
        scenario_filter = st.selectbox("Scenario", ["All", "BASELINE", "IMPROVED"])
    
    filtered = anomalies[anomalies["severity"].isin(sev_filter)]
    if review_only:
        filtered = filtered[filtered["operator_review_required"] == True]
    if scenario_filter != "All":
        # Need to join scenario from missions
        mission_scenario = missions[["mission_id", "scenario"]]
        filtered = filtered.merge(mission_scenario, on="mission_id", how="left")
        filtered = filtered[filtered["scenario"] == scenario_filter]
    
    filtered = filtered.sort_values("priority", ascending=False)
    
    st.metric("Items in Queue", len(filtered))
    
    for _, row in filtered.head(15).iterrows():
        with st.expander(f"{row['severity']} | {row['asset_name']} ({row['asset_id']}) @ {row['location_name']} — {row['anomaly_type']}"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Timestamp:** {row['timestamp']}")
                st.write(f"**Mission:** {row['mission_id']} | **Action:** {row['action_id']}")
                st.write(f"**Asset Criticality:** {row['criticality']} | **Priority Score:** {row['priority']}")
            with col_b:
                if st.button("Explain with Gemini", key=f"exp_{row['anomaly_id']}"):
                    st.info("Gemini prompt ready — see Gemini Assistant tab for full integration.")
                    prompt = f"""You are an expert plant operator assistant. Explain this SPOT anomaly in plain language for a non-technical operator.

Anomaly: {row['anomaly_type']}
Severity: {row['severity']}
Asset: {row['asset_name']} ({row['asset_type']}, criticality: {row['criticality']})
Location: {row['location_name']}
Description: {row['description']}
Timestamp: {row['timestamp']}

Provide:
1. What this likely means in simple terms
2. Why it matters for operations and local environmental safety
3. Recommended next action (1-2 sentences)
4. Confidence note"""
                    st.code(prompt, language="text")

elif page == "📊 Performance Comparison":
    st.header("BASELINE (V1) vs IMPROVED (V2) Performance")
    
    # Mission completion
    mission_comp = missions.groupby("scenario")["mission_status"].apply(lambda x: (x == "COMPLETED").mean() * 100).reset_index()
    mission_comp.columns = ["scenario", "completion_pct"]
    
    # Action success
    action_success = actions.groupby("scenario")["status"].apply(lambda x: (x == "SUCCESS").mean() * 100).reset_index()
    action_success.columns = ["scenario", "success_pct"]
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(mission_comp, x="scenario", y="completion_pct", title="Mission Completion Rate (%)",
                      color="scenario", color_discrete_map={"BASELINE": "#78909c", "IMPROVED": "#2e7d32"})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(action_success, x="scenario", y="success_pct", title="Action Success Rate (%)",
                      color="scenario", color_discrete_map={"BASELINE": "#78909c", "IMPROVED": "#2e7d32"})
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Anomaly Severity Distribution")
    # Join scenario
    anom_scen = anomalies.merge(missions[["mission_id", "scenario"]], on="mission_id", how="left")
    sev_dist = anom_scen.groupby(["scenario", "severity"]).size().reset_index(name="count")
    fig3 = px.bar(sev_dist, x="severity", y="count", color="scenario", barmode="group",
                  category_orders={"severity": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]})
    st.plotly_chart(fig3, use_container_width=True)
    
    st.success("**Insight:** IMPROVED configuration shows higher mission completion and action success rates. High-severity anomalies are reduced, supporting greater operator confidence and lower environmental risk from equipment issues.")

elif page == "📈 Readings Explorer":
    st.header("Sensor Readings Explorer")
    
    measurement = st.selectbox("Measurement Type", readings["measurement_type"].unique())
    filtered_r = readings[readings["measurement_type"] == measurement]
    
    fig = px.scatter(filtered_r, x="timestamp", y="value", color="quality",
                     hover_data=["asset_name", "location_id", "mission_id"],
                     title=f"{measurement} over Time")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(filtered_r[["timestamp", "asset_name", "value", "unit", "quality", "mission_id"]].sort_values("timestamp", ascending=False).head(20),
                 use_container_width=True)

elif page == "🤖 Gemini Assistant":
    st.header("Gemini-Powered Operator Assistant")
    st.markdown("""
    **Best Use of Gemini API track**: This assistant turns complex SPOT data into plain-language insights that build operator confidence.
    
    Features (demo prompts ready for live API):
    - Anomaly explanations
    - Mission summaries
    - Natural language data queries
    - Shift handoff reports
    - Community impact narratives
    """)
    
    st.subheader("Quick Actions")
    if st.button("Generate Mission Comparison Summary"):
        prompt = """Analyze the SPOT mission data comparing BASELINE (V1) vs IMPROVED (V2).

From validation:
- Baseline mission completion: ~94.4%, action success: ~77.2%
- Improved mission completion: 100%, action success: ~91.1%

Provide a concise operator briefing (under 150 words) covering:
1. What improved
2. Remaining risks
3. Recommended focus areas
4. Confidence statement for non-technical operators"""
        st.code(prompt)
        st.info("Connect Gemini API key to generate live response. Prompt is optimized for structured, actionable output.")
    
    if st.button("Generate High-Priority Anomaly Briefing"):
        high_anoms = anomalies[anomalies["severity"].isin(["HIGH", "CRITICAL"])].head(5)
        summary = high_anoms[["anomaly_type", "severity", "asset_name", "location_name", "description"]].to_string()
        prompt = f"""Create a prioritized operator briefing for these high-severity SPOT anomalies:

{summary}

For each, give: plain-language meaning, operational impact, environmental safety note (Central Florida context), and recommended action."""
        st.code(prompt)
    
    st.subheader("Ask a Question")
    user_q = st.text_input("Ask about the SPOT data (e.g., 'Which locations have the most critical anomalies?')")
    if user_q:
        st.write("**Sample Gemini system prompt for RAG-style answer:**")
        st.code(f"""You are SPOTInsight, an AI assistant for energy plant operators. Use the provided SPOT mission, action, reading, and anomaly data to answer accurately and in plain language.

User question: {user_q}

Answer with:
- Direct answer
- Supporting data points
- Recommended next step if applicable
- Note any limitations""")

elif page == "🌍 Community Impact":
    st.header("OneEthos: Innovation for Good — Central Florida Impact")
    st.markdown("""
    **Local Problem**: Communities in Lake County / Eustis, Florida face environmental challenges including lake water quality degradation, stormwater runoff, and the risk that energy infrastructure failures (overheating, vibration, leaks) can affect local ecosystems and reliability.
    
    **Solution**: SPOTInsight makes robotic inspection data actionable so operators can prevent equipment issues before they become environmental incidents. Improved configuration performance directly reduces risk.
    """)
    
    # Metrics
    base_success = actions[actions["scenario"] == "BASELINE"]["status"].eq("SUCCESS").mean() * 100
    imp_success = actions[actions["scenario"] == "IMPROVED"]["status"].eq("SUCCESS").mean() * 100
    improvement = imp_success - base_success
    
    high_base = len(anomalies.merge(missions[["mission_id", "scenario"]], on="mission_id").query("scenario == 'BASELINE' and severity in ['HIGH', 'CRITICAL']"))
    high_imp = len(anomalies.merge(missions[["mission_id", "scenario"]], on="mission_id").query("scenario == 'IMPROVED' and severity in ['HIGH', 'CRITICAL']"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Action Success Improvement", f"+{improvement:.1f} pp", f"{base_success:.1f}% → {imp_success:.1f}%")
    with col2:
        st.metric("High/Critical Anomalies (BASELINE)", high_base)
    with col3:
        st.metric("High/Critical Anomalies (IMPROVED)", high_imp, delta=f"{high_imp - high_base}")
    
    st.subheader("Measurable Impact Proxies")
    st.markdown("""
    - **Reliability gain**: Higher action success rate means fewer missed inspections and faster issue detection.
    - **Risk reduction**: Fewer high-severity anomalies (temperature, vibration, visual defects) lower the chance of equipment failure that could lead to leaks or emissions affecting nearby lakes.
    - **Operator confidence**: Gemini plain-language explanations reduce the expertise barrier, enabling more consistent action.
    - **Community benefit**: Proactive maintenance supported by this dashboard helps protect Central Florida water quality and energy reliability.
    """)
    
    if st.button("Generate Community Impact Narrative (Gemini)"):
        prompt = """Write a short (100-120 word) community impact statement for a Central Florida audience (Lake County / Eustis area).

Context: A SPOT robot inspection dashboard improved action success from ~77% to ~91% and reduced high-severity anomalies. This helps prevent equipment failures at energy facilities that could affect local lakes and ecosystems.

Emphasize measurable reliability gains and environmental co-benefits. Keep language accessible and positive."""
        st.code(prompt)
        st.success("Ready for Gemini API. This narrative can be included in submissions for the OneEthos Innovation for Good track.")

st.sidebar.markdown("---")
st.sidebar.caption("KnightHacks 2026 | SPOTInsight AI")
st.sidebar.caption("Tracks: NextEra SPOT Dashboard • MLH Gemini • OneEthos Innovation for Good")
