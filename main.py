import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads .env into os.environ

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================
# 0) CONFIG
# =========================
DEFAULT_YEAR = 2024
DEFAULT_STATE_FIPS = "37"  # North Carolina
ACS5_BASE = "https://api.census.gov/data/{year}/acs/acs5"
ACS5_SUBJECT_BASE = "https://api.census.gov/data/{year}/acs/acs5/subject"

STATES = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19",
    "Kansas": "20", "Kentucky": "21", "Louisiana": "22", "Maine": "23",
    "Maryland": "24", "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
    "Mississippi": "28", "Missouri": "29", "Montana": "30", "Nebraska": "31",
    "Nevada": "32", "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35",
    "New York": "36", "North Carolina": "37", "North Dakota": "38", "Ohio": "39",
    "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44",
    "South Carolina": "45", "South Dakota": "46", "Tennessee": "47", "Texas": "48",
    "Utah": "49", "Vermont": "50", "Virginia": "51", "Washington": "53",
    "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56",
}
ALL_STATE_FIPS = tuple(sorted(STATES.values()))

st.set_page_config(
    page_title="Homework Gap Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 1) HELPERS
# =========================
def census_get(
    base_url: str,
    get_vars: list[str],
    geo_for: str,
    geo_in: Optional[str],
    api_key: str,
    include_name: bool = True,
) -> pd.DataFrame:
    """Generic Census API fetch -> pandas DataFrame."""
    get_list = (["NAME"] if include_name else []) + get_vars
    params = {"get": ",".join(get_list), "for": geo_for, "key": api_key}
    if geo_in:
        params["in"] = geo_in

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data[1:], columns=data[0])


@st.cache_data(show_spinner=False)
def group_vars(base_url: str, group: str) -> pd.DataFrame:
    url = f"{base_url}/groups/{group}.json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for varname, meta in payload["variables"].items():
        rows.append({"name": varname, "label": meta.get("label", "")})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def find_var_by_label(base_url: str, group: str, contains_text: str) -> str:
    meta = group_vars(base_url, group)
    matches = meta[meta["label"].str.contains(contains_text, case=False, na=False)]
    matches = matches[matches["name"].str.endswith("E")]
    if matches.empty:
        raise ValueError(f"Couldn't find a var in {group} containing: {contains_text}")
    return matches.iloc[0]["name"]


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assign_region(longitude: float) -> str:
    """Broad US macro-regions from centroid longitude."""
    if pd.isna(longitude):
        return "Unknown"
    if longitude <= -104:
        return "West"
    if longitude <= -90:
        return "Central"
    return "East"


def categorize_access(max_gap: float) -> str:
    if pd.isna(max_gap):
        return "Unknown"
    if max_gap >= 0.25:
        return "Critical"
    if max_gap >= 0.15:
        return "High"
    if max_gap >= 0.08:
        return "Moderate"
    return "Lower"


@st.cache_data(show_spinner=False)
def load_homework_gap_data(api_key: str, year: int = DEFAULT_YEAR, state_fips_tuple: tuple = (DEFAULT_STATE_FIPS,)):
    acs5_base = ACS5_BASE.format(year=year)
    acs5_subject_base = ACS5_SUBJECT_BASE.format(year=year)
    geo_for = "school district (unified):*"
    geo_in = f"state:{','.join(state_fips_tuple)}"

    b09001_total_under18 = "B09001_001E"
    b28002_total_hh = "B28002_001E"
    b28002_no_internet = find_var_by_label(acs5_base, "B28002", "No Internet access")
    b28001_total_hh = "B28001_001E"
    b28001_no_computer = find_var_by_label(acs5_base, "B28001", "No computer")
    s1901_median_hh_income = "S1901_C01_012E"

    df_main = census_get(
        acs5_base,
        get_vars=[
            b09001_total_under18,
            b28002_total_hh,
            b28002_no_internet,
            b28001_total_hh,
            b28001_no_computer,
        ],
        geo_for=geo_for,
        geo_in=geo_in,
        api_key=api_key,
        include_name=True,
    )

    df_income = census_get(
        acs5_subject_base,
        get_vars=[s1901_median_hh_income],
        geo_for=geo_for,
        geo_in=geo_in,
        api_key=api_key,
        include_name=False,
    )

    numeric_cols = [
        b09001_total_under18,
        b28002_total_hh,
        b28002_no_internet,
        b28001_total_hh,
        b28001_no_computer,
    ]
    df_main = to_num(df_main, numeric_cols)
    df_income = to_num(df_income, [s1901_median_hh_income])

    key_cols = ["state", "school district (unified)"]
    df = df_main.merge(df_income[key_cols + [s1901_median_hh_income]], on=key_cols, how="left")

    # Enforce zero-padding so Census GEOID always matches TIGER's 7-char format
    df["GEOID"] = df["state"].str.zfill(2) + df["school district (unified)"].str.zfill(5)
    df["district_name"] = df["NAME"]
    df["students_under_18"] = df[b09001_total_under18]
    df["households_total_internet"] = df[b28002_total_hh]
    df["households_no_internet"] = df[b28002_no_internet]
    df["households_total_computer"] = df[b28001_total_hh]
    df["households_no_computer"] = df[b28001_no_computer]
    df["median_household_income"] = df[s1901_median_hh_income]

    df["pct_no_internet"] = (
        df["households_no_internet"] / df["households_total_internet"].replace(0, pd.NA)
    ).fillna(0).clip(lower=0, upper=1)
    df["pct_no_computer"] = (
        df["households_no_computer"] / df["households_total_computer"].replace(0, pd.NA)
    ).fillna(0).clip(lower=0, upper=1)
    df["connectivity_gap"] = df[["pct_no_internet", "pct_no_computer"]].max(axis=1)

    max_students = df["students_under_18"].max()
    max_income = df["median_household_income"].max()

    child_component = (
        (df["students_under_18"].fillna(0) / max_students).fillna(0)
        if pd.notna(max_students) and max_students != 0
        else pd.Series(0.0, index=df.index)
    )
    internet_component = df["pct_no_internet"].fillna(0)
    device_component = df["pct_no_computer"].fillna(0)
    income_median = df["median_household_income"].median()
    income_component = (
        (1 - df["median_household_income"].fillna(income_median if pd.notna(income_median) else 0) / max_income).fillna(0)
        if pd.notna(max_income) and max_income != 0
        else pd.Series(0.0, index=df.index)
    )

    df["urgency_score"] = (
        child_component * 0.40
        + internet_component * 0.25
        + device_component * 0.25
        + income_component.clip(lower=0) * 0.10
    ).fillna(0)

    # Geometry for regional filter + map — fetch per state and concatenate
    layer_id = 10
    gdfs = []
    for fips in state_fips_tuple:
        tiger_query_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/School/MapServer/"
            f"{layer_id}/query?where=STATE='{fips}'&outFields=GEOID,NAME,STATE&outSR=4326&f=geojson&returnGeometry=true"
        )
        try:
            gdf_state = gpd.read_file(tiger_query_url)
            gdfs.append(gdf_state)
        except Exception:
            pass
    if not gdfs:
        raise ValueError("Failed to load district geometry from TIGER for any selected state.")
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs="EPSG:4326")

    # Simplify geometries to reduce browser payload — tolerance scales with scope
    n_states = len(state_fips_tuple)
    simplify_tol = 0.001 if n_states == 1 else 0.003 if n_states <= 5 else 0.008 if n_states <= 20 else 0.015
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=simplify_tol, preserve_topology=True)

    # Inner join: drop TIGER polygons with no matching Census data (they'd show blank)
    gdf = gdf.merge(df, on="GEOID", how="inner", suffixes=("_shape", "_acs"))

    # Standardize district name after merge
    if "district_name" not in gdf.columns:
        if "NAME_acs" in gdf.columns:
            gdf["district_name"] = gdf["NAME_acs"]
        elif "NAME_shape" in gdf.columns:
            gdf["district_name"] = gdf["NAME_shape"]
        elif "NAME" in gdf.columns:
            gdf["district_name"] = gdf["NAME"]

    # Compute centroids in projected CRS, then convert back to lat/lon
    gdf = gdf.to_crs(epsg=4326)
    gdf_proj = gdf.to_crs(epsg=3857)
    centroids_proj = gdf_proj.geometry.centroid
    centroids = gpd.GeoSeries(centroids_proj, crs=gdf_proj.crs).to_crs(epsg=4326)
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    gdf["region"] = gdf["longitude"].apply(assign_region)
    gdf["access_level"] = gdf["connectivity_gap"].apply(categorize_access)

    df_out = pd.DataFrame(gdf.drop(columns="geometry"))
    df_out["rank"] = df_out["urgency_score"].fillna(0).rank(method="dense", ascending=False).fillna(0).astype(int)

    return df_out, gdf


def allocate_laptops(df: pd.DataFrame, total_laptops: int) -> pd.DataFrame:
    sim = df.copy()
    unmet_need = sim["households_no_computer"].fillna(0).clip(lower=0)
    weight = sim["urgency_score"].fillna(0).clip(lower=0)

    if total_laptops <= 0 or weight.sum() == 0:
        sim["allocated_laptops"] = 0
        sim["students_connected_est"] = 0
        sim["pct_device_gap_closed"] = 0.0
        return sim

    raw_allocation = (weight / weight.sum()) * total_laptops
    base_allocation = np.floor(raw_allocation).astype(int)
    base_allocation = pd.concat([base_allocation, unmet_need.astype(int)], axis=1).min(axis=1)

    remaining = total_laptops - int(base_allocation.sum())
    fractional_order = (raw_allocation - np.floor(raw_allocation)).sort_values(ascending=False).index.tolist()

    final_alloc = base_allocation.copy()
    for idx in fractional_order:
        if remaining <= 0:
            break
        if final_alloc.loc[idx] < int(unmet_need.loc[idx]):
            final_alloc.loc[idx] += 1
            remaining -= 1

    sim["allocated_laptops"] = final_alloc.astype(int)
    sim["students_connected_est"] = sim[["allocated_laptops", "students_under_18"]].min(axis=1)
    sim["pct_device_gap_closed"] = (
        sim["allocated_laptops"] / sim["households_no_computer"].replace(0, pd.NA)
    ).fillna(0).clip(upper=1)
    return sim


# =========================
# 2) APP UI
# =========================
st.title("Homework Gap Dashboard")
st.caption("Interactive decision dashboard for targeting laptop grants to high-need US school districts.")

with st.sidebar:
    st.header("Data and Filters")
    api_key = st.text_input(
        "Census API key",
        value=os.getenv("CENSUS_API_KEY", ""),
        type="password",
        help="Use an environment variable or paste a key here for local testing.",
    )
    year = st.number_input("ACS year", min_value=2019, max_value=2024, value=DEFAULT_YEAR, step=1)
    selected_states = st.multiselect(
        "State(s)",
        options=["All States"] + sorted(STATES.keys()),
        default=["North Carolina"],
        help="Choose one or more states, or 'All States' for a national view. Loading many states may take several minutes on first run (results are cached).",
    )
    if not selected_states or "All States" in selected_states:
        selected_fips = ALL_STATE_FIPS
    else:
        selected_fips = tuple(STATES[s] for s in selected_states if s in STATES)

if not api_key:
    st.warning("Enter a Census API key in the sidebar to load the dashboard.")
    st.stop()

if len(selected_fips) > 5:
    st.info(
        f"⏳ Loading data for **{len(selected_fips)} state(s)** — this may take a few minutes on first load. "
        "Results are cached for the rest of your session."
    )

try:
    df, gdf = load_homework_gap_data(api_key=api_key, year=int(year), state_fips_tuple=selected_fips)
except Exception as exc:
    st.error(f"Could not load Census data: {exc}")
    st.stop()

income_series = df["median_household_income"].dropna()
if income_series.empty:
    income_min, income_max = 0, 100000
else:
    income_min = 0  # always start from $0
    income_max = int(income_series.max())

max_gap_pct = float(df["connectivity_gap"].fillna(0).max() * 100)

with st.sidebar:
    income_range = st.slider(
        "Median household income range",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, int(df["median_household_income"].quantile(0.80)) if not income_series.empty else income_max),
        step=1000,
        help="Use this to exclude wealthier districts.",
    )
    region_options = sorted(df["region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Region", options=region_options, default=region_options)
    selected_access_levels = st.multiselect(
        "Access level",
        options=["Critical", "High", "Moderate", "Lower", "Unknown"],
        default=["Critical", "High", "Moderate", "Lower"],
    )
    access_gap_range = st.slider(
        "Connectivity gap (%)",
        min_value=0.0,
        max_value=max(5.0, round(max_gap_pct, 1)),
        value=(8.0, max(5.0, round(max_gap_pct, 1))),
        step=0.5,
        help="Highest of no-internet or no-computer share.",
    )
    total_laptops = st.slider("Laptops available for distribution", 0, 10000, 1000, 50)
    top_n = st.slider("Show top districts", 5, 50, 15, 5)

filtered = df.copy()
filtered = filtered[
    filtered["median_household_income"].isna()
    | filtered["median_household_income"].between(income_range[0], income_range[1])
]
if selected_regions:
    filtered = filtered[filtered["region"].isin(selected_regions)]
if selected_access_levels:
    filtered = filtered[filtered["access_level"].isin(selected_access_levels)]
filtered = filtered[
    filtered["connectivity_gap"].fillna(0).between(access_gap_range[0] / 100, access_gap_range[1] / 100)
]
filtered = filtered.sort_values("urgency_score", ascending=False).reset_index(drop=True)
filtered["rank"] = filtered.index + 1
simulated = allocate_laptops(filtered, total_laptops)

# =========================
# 3) KPI ROW
# =========================
metric_cols = st.columns(4)
metric_cols[0].metric("Districts in scope", f"{len(simulated):,}")
metric_cols[1].metric("Students under 18", f"{int(simulated['students_under_18'].fillna(0).sum()):,}")
metric_cols[2].metric("Households with no internet", f"{int(simulated['households_no_internet'].fillna(0).sum()):,}")
metric_cols[3].metric("Households with no computer", f"{int(simulated['households_no_computer'].fillna(0).sum()):,}")

sim_cols = st.columns(3)
sim_cols[0].metric("Laptops allocated", f"{int(simulated['allocated_laptops'].sum()):,}")
sim_cols[1].metric("Estimated students helped", f"{int(simulated['students_connected_est'].sum()):,}")
sim_cols[2].metric(
    "Avg. device-gap closed",
    f"{(simulated['pct_device_gap_closed'].mean() * 100 if len(simulated) else 0):.1f}%",
)

if simulated.empty:
    st.info("No districts match the current filters.")
    st.stop()

# =========================
# 4) VISUALS
# =========================
left, right = st.columns((1.1, 0.9))

with left:
    st.subheader("Urgency ranking")
    ranking_view = simulated.head(top_n)[[
        "rank",
        "district_name",
        "region",
        "access_level",
        "students_under_18",
        "pct_no_internet",
        "pct_no_computer",
        "median_household_income",
        "urgency_score",
        "allocated_laptops",
    ]].copy()
    ranking_view["pct_no_internet"] = (ranking_view["pct_no_internet"] * 100).round(1)
    ranking_view["pct_no_computer"] = (ranking_view["pct_no_computer"] * 100).round(1)
    ranking_view["urgency_score"] = ranking_view["urgency_score"].round(3)
    st.dataframe(ranking_view, use_container_width=True, hide_index=True)

    bar_df = simulated.head(top_n).copy()
    bar_fig = px.bar(
        bar_df.sort_values("urgency_score", ascending=True),
        x="urgency_score",
        y="district_name",
        orientation="h",
        hover_data={
            "students_under_18": True,
            "pct_no_internet": ':.1%',
            "pct_no_computer": ':.1%',
            "median_household_income": ':$,.0f',
            "allocated_laptops": True,
        },
        title="Top districts ranked by urgency score",
    )
    bar_fig.update_layout(height=550, yaxis_title="District", xaxis_title="Urgency score")
    st.plotly_chart(bar_fig, use_container_width=True)

with right:
    st.subheader("Map of filtered districts")

    MAP_DISTRICT_CAP = 300
    map_geoids = simulated["GEOID"].iloc[:MAP_DISTRICT_CAP]

    if len(simulated) > MAP_DISTRICT_CAP:
        st.caption(
            f"⚠️ Showing top {MAP_DISTRICT_CAP} of {len(simulated):,} filtered districts on the map "
            "(ranked by urgency score). Use the sidebar filters or 'Show top districts' slider to narrow down."
        )

    map_df = gdf[gdf["GEOID"].isin(map_geoids)].copy()

    # Remove stale columns before re-merging simulated values
    map_df = map_df.drop(
        columns=[c for c in ["allocated_laptops", "students_connected_est", "pct_device_gap_closed", "rank"] if c in map_df.columns],
        errors="ignore",
    )

    map_df = map_df.merge(
        simulated[["GEOID", "allocated_laptops", "students_connected_est", "pct_device_gap_closed", "rank"]],
        on="GEOID",
        how="inner",
    )

    # Final cleanup: remove unmatched / empty rows so blank districts do not appear
    map_df = map_df.dropna(subset=["urgency_score", "district_name", "geometry"]).copy()
    map_df = map_df[map_df.geometry.is_valid & ~map_df.geometry.is_empty].copy()

    map_df = map_df.to_crs(4326)

    if map_df.empty:
        st.info("No mapped districts match the current filters.")
    else:
        n_selected = len(selected_fips)
        map_zoom = 3.2 if n_selected >= 30 else 4.0 if n_selected >= 10 else 5.8

        map_fig = px.choropleth_mapbox(
            map_df,
            geojson=map_df.geometry.__geo_interface__,
            locations=map_df.index,
            color="urgency_score",
            hover_name="district_name",
            hover_data={
                "region": True,
                "access_level": True,
                "students_under_18": True,
                "pct_no_internet": ':.1%',
                "pct_no_computer": ':.1%',
                "allocated_laptops": True,
                "longitude": False,
                "latitude": False,
            },
            center={
                "lat": float(map_df["latitude"].mean()),
                "lon": float(map_df["longitude"].mean()),
            },
            zoom=map_zoom,
            opacity=0.65,
        )

        map_fig.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=40, b=0),
            height=550,
        )
        st.plotly_chart(map_fig, use_container_width=True)

scatter_left, scatter_right = st.columns(2)

with scatter_left:
    scatter_fig = px.scatter(
        simulated,
        x="students_under_18",
        y="pct_no_internet",
        size="urgency_score",
        color="region",
        hover_name="district_name",
        hover_data={
            "pct_no_computer": ':.1%',
            "median_household_income": ':$,.0f',
            "allocated_laptops": True,
        },
        title="School-age population vs. no internet rate",
    )
    scatter_fig.update_layout(xaxis_title="Students under 18", yaxis_title="No internet share")
    st.plotly_chart(scatter_fig, use_container_width=True)

with scatter_right:
    impact_fig = px.bar(
        simulated.head(top_n).sort_values("allocated_laptops", ascending=True),
        x="allocated_laptops",
        y="district_name",
        orientation="h",
        hover_data={
            "students_connected_est": True,
            "pct_device_gap_closed": ':.1%',
            "households_no_computer": True,
        },
        title="Simulated laptop allocation impact",
    )
    impact_fig.update_layout(xaxis_title="Allocated laptops", yaxis_title="District")
    st.plotly_chart(impact_fig, use_container_width=True)

st.subheader("Policy notes")
st.markdown(
    """
- **Income filter** helps exclude wealthier districts from the grant pool.
- **Region filter** groups NC districts into West, Central, and East using district centroids.
- **Access level** is based on the worse of the no-internet and no-computer rates.
- **Simulation** distributes laptops proportionally to urgency score, while capping allocation at each district's estimated no-computer households.
"""
)

st.download_button(
    "Download filtered district rankings as CSV",
    data=simulated.to_csv(index=False).encode("utf-8"),
    file_name="homework_gap_rankings.csv",
    mime="text/csv",
)

# =========================
# 5) AI ASSISTANT
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "openai/gpt-4o-mini"

st.divider()
st.subheader("🤖 AI Policy Assistant")
st.caption(
    "Ask questions about the current filtered data — e.g. *Which districts should be prioritized?*, "
    "*Summarize the connectivity gap trends*, or *How should we allocate resources?*"
)


def _build_context(sim_df: pd.DataFrame) -> str:
    top = sim_df.head(10)[[
        "district_name", "region", "access_level",
        "students_under_18", "pct_no_internet", "pct_no_computer",
        "median_household_income", "urgency_score", "allocated_laptops",
    ]].copy()
    top["pct_no_internet"] = (top["pct_no_internet"] * 100).round(1)
    top["pct_no_computer"] = (top["pct_no_computer"] * 100).round(1)
    top["urgency_score"] = top["urgency_score"].round(3)
    summary_lines = [
        f"Total districts in scope: {len(sim_df):,}",
        f"Total students under 18: {int(sim_df['students_under_18'].fillna(0).sum()):,}",
        f"Households with no internet: {int(sim_df['households_no_internet'].fillna(0).sum()):,}",
        f"Households with no computer: {int(sim_df['households_no_computer'].fillna(0).sum()):,}",
        f"Laptops allocated: {int(sim_df['allocated_laptops'].sum()):,}",
        f"Estimated students helped: {int(sim_df['students_connected_est'].sum()):,}",
        "",
        "Top 10 highest-urgency districts:",
        top.to_string(index=False),
    ]
    return "\n".join(summary_lines)


SYSTEM_PROMPT = """You are an expert data analyst and education policy advisor helping local government officials
understand the 'Homework Gap' — the disparity in home internet and device access among school-age children.
You are given a real-time snapshot of the dashboard's currently filtered data.
Your role is to:
- Interpret connectivity gap metrics and urgency scores clearly.
- Provide actionable, evidence-based policy recommendations.
- Highlight districts most in need of intervention.
- Explain trade-offs in laptop/resource allocation.
- Be concise, direct, and use plain language suitable for policymakers.
Always ground your answers in the data provided."""

# Session state for chat history
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

# Render existing chat messages
for msg in st.session_state.ai_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask the AI about the dashboard data...")

if user_input:
    st.session_state.ai_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    data_context = _build_context(simulated)
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current dashboard data snapshot:\n\n{data_context}",
        },
    ] + st.session_state.ai_messages

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                )
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=api_messages,
                    max_tokens=1024,
                    temperature=0.3,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ AI error: {e}"

        st.markdown(reply)

    st.session_state.ai_messages.append({"role": "assistant", "content": reply})

if st.session_state.ai_messages:
    if st.button("🗑️ Clear chat", key="clear_ai_chat"):
        st.session_state.ai_messages = []
        st.rerun()

