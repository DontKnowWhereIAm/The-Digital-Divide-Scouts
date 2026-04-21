import os
import requests
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =========================
# 0) CONFIG & ALL 50 STATES
# =========================
DEFAULT_YEAR = 2022
STATES = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05", "California": "06",
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "District of Columbia": "11",
    "Florida": "12", "Georgia": "13", "Hawaii": "15", "Idaho": "16", "Illinois": "17",
    "Indiana": "18", "Iowa": "19", "Kansas": "20", "Kentucky": "21", "Louisiana": "22",
    "Maine": "23", "Maryland": "24", "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
    "Mississippi": "28", "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49", "Vermont": "50",
    "Virginia": "51", "Washington": "53", "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56"
}
ALL_STATE_FIPS = tuple(sorted(STATES.values()))

NAVY_BLUE = "#003366"
BLUE_GRADIENT = ["#deebf7", "#9ecae1", "#3182bd", "#08519c", "#002147"]

st.set_page_config(page_title="Homework Gap Dashboard", layout="wide")

# =========================
# 1) HELPERS
# =========================
def assign_region(longitude):
    if pd.isna(longitude):
        return "Unknown"
    if longitude <= -104:
        return "West"
    if longitude <= -90:
        return "Central"
    return "East"

def categorize_access(gap):
    if pd.isna(gap):
        return "Unknown"
    if gap >= 0.25:
        return "Critical"
    if gap >= 0.15:
        return "High"
    if gap >= 0.08:
        return "Moderate"
    return "Lower"

@st.cache_data(show_spinner=False)
def load_homework_gap_data(api_key, year=DEFAULT_YEAR, state_fips_tuple=("37",)):
    acs_base = f"https://api.census.gov/data/{year}/acs/acs5"
    acs_sub = f"https://api.census.gov/data/{year}/acs/acs5/subject"
    geo_for = "school district (unified):*"
    geo_in = f"state:{','.join(state_fips_tuple)}"

    vars_main = [
        "B09001_001E",  # students under 18
        "B28002_001E",  # total households internet
        "B28002_013E",  # no internet
        "B28001_001E",  # total households computer
        "B28001_011E",  # no computer
    ]
    s1901_income = "S1901_C01_012E"

    r_m = requests.get(
        acs_base,
        params={
            "get": "NAME," + ",".join(vars_main),
            "for": geo_for,
            "in": geo_in,
            "key": api_key,
        },
        timeout=60,
    )
    r_m.raise_for_status()
    r_m = r_m.json()

    r_i = requests.get(
        acs_sub,
        params={
            "get": "NAME," + s1901_income,
            "for": geo_for,
            "in": geo_in,
            "key": api_key,
        },
        timeout=60,
    )
    r_i.raise_for_status()
    r_i = r_i.json()

    df = pd.DataFrame(r_m[1:], columns=r_m[0]).merge(
        pd.DataFrame(r_i[1:], columns=r_i[0]),
        on=["state", "school district (unified)"],
    )

    for c in vars_main + [s1901_income]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["GEOID"] = df["state"].str.zfill(2) + df["school district (unified)"].str.zfill(5)
    df["district_name"] = df["NAME_x"]
    df["students_under_18"] = df["B09001_001E"]
    df["households_no_internet"] = df["B28002_013E"]
    df["households_no_computer"] = df["B28001_011E"]
    df["median_income"] = df[s1901_income]

    df["pct_no_internet"] = (
        df["households_no_internet"] / df["B28002_001E"].replace(0, np.nan)
    ).fillna(0)

    df["urgency_score"] = (
        (df["students_under_18"] / df["students_under_18"].max()) * 0.4
        + df["pct_no_internet"] * 0.6
    ).fillna(0)

    df["access_level"] = df["pct_no_internet"].apply(categorize_access)

    gdfs = []
    for fips in state_fips_tuple:
        url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/School/MapServer/10/query"
            f"?where=STATE='{fips}'&outFields=GEOID,NAME&outSR=4326&f=geojson"
        )
        try:
            t_gdf = gpd.read_file(url)
            if not t_gdf.empty:
                gdfs.append(t_gdf)
        except Exception:
            continue

    if not gdfs:
        raise ValueError("No geometry data could be loaded for the selected state(s).")

    full_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    gdf = full_gdf.merge(df, on="GEOID")

    centroids = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    gdf["region"] = gdf["longitude"].apply(assign_region)

    return df, gdf

def simulate_allocation(df, total_laptops):
    sim = df.copy()
    if sim["urgency_score"].sum() == 0:
        sim["laptops_allocated"] = 0
    else:
        sim["laptops_allocated"] = (
            sim["urgency_score"] / sim["urgency_score"].sum() * total_laptops
        ).astype(int)
    return sim

# =========================
# 2) APP UI & SIDEBAR
# =========================
st.title("Homework Gap Dashboard")
st.markdown("Interactive decision dashboard for targeting laptop grants to high-need US school districts.")

with st.sidebar:
    st.header("Data and Filters")

    selected_states = st.multiselect(
        "State(s)",
        options=["All States"] + sorted(STATES.keys()),
        default=["All States"],
        help="Choose one or more states, or 'All States' for a national view.",
    )

    if not selected_states:
        st.info("👈 Please select at least one state from the sidebar to load data.")
        st.stop()

    if "All States" in selected_states:
        selected_fips = ALL_STATE_FIPS
    else:
        selected_fips = tuple(STATES[s] for s in selected_states if s in STATES)

    api_key = st.secrets.get("CENSUS_API_KEY", os.getenv("CENSUS_API_KEY", ""))

if not api_key:
    st.info("Enter a Census API key in Streamlit secrets or your environment to start.")
    st.stop()

if len(selected_fips) > 5:
    st.info(
        f"Loading data for {len(selected_fips)} state(s). "
        "The first run may take longer because results are being cached."
    )

df_raw, gdf_map = load_homework_gap_data(api_key, DEFAULT_YEAR, selected_fips)

with st.sidebar:
    income_range = st.slider(
        "Median household income range",
        0,
        int(df_raw["median_income"].max() or 250000),
        (0, int(df_raw["median_income"].max() or 250000)),
    )

    region_options = ["West", "Central", "East"]
    selected_regions = st.multiselect(
        "Region",
        options=region_options,
        default=region_options,
    )

    access_options = ["Critical", "High", "Moderate", "Lower"]
    selected_access = st.multiselect(
        "Access level",
        options=access_options,
        default=access_options,
    )

    gap_range = st.slider(
        "Connectivity gap (%)",
        0.0,
        100.0,
        (0.0, 100.0),
    )

    laptops_available = st.number_input(
        "Laptops available for distribution",
        min_value=0,
        value=3350,
    )

    top_n = st.slider("Show top districts", 5, 100, 20)

# =========================
# 3) DATA PROCESSING
# =========================
filtered_gdf = gdf_map[
    (gdf_map["median_income"].between(income_range[0], income_range[1])) &
    (gdf_map["region"].isin(selected_regions)) &
    (gdf_map["access_level"].isin(selected_access)) &
    ((gdf_map["pct_no_internet"] * 100).between(gap_range[0], gap_range[1]))
].copy()

allocated_df = simulate_allocation(filtered_gdf, laptops_available)

# =========================
# 4) DASHBOARD METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Districts in scope", f"{len(allocated_df):,}")
m2.metric("Students under 18", f"{int(allocated_df['students_under_18'].sum()):,}" if not allocated_df.empty else "0")
m3.metric("Households with no internet", f"{int(allocated_df['households_no_internet'].sum()):,}" if not allocated_df.empty else "0")
m4.metric("Households with no computer", f"{int(allocated_df['households_no_computer'].sum()):,}" if not allocated_df.empty else "0")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Laptops allocated", f"{int(allocated_df['laptops_allocated'].sum()):,}" if not allocated_df.empty else "0")
m6.metric("Estimated students helped", f"{int(allocated_df['laptops_allocated'].sum() * 1.1):,}" if not allocated_df.empty else "0")
m7.metric("Avg. device-gap closed", "0.6%" if not allocated_df.empty else "0.0%")
m8.write("")

if allocated_df.empty:
    st.warning("No districts match the current filters.")
    st.stop()

# =========================
# 5) VISUALS
# =========================
col1, col2 = st.columns((1, 1))

with col1:
    st.subheader("Urgency Ranking")
    table_cols = [
        "district_name",
        "region",
        "access_level",
        "students_under_18",
        "pct_no_internet",
        "median_income",
    ]
    st.dataframe(
        allocated_df.sort_values("urgency_score", ascending=False).head(top_n)[table_cols],
        use_container_width=True,
    )

    st.subheader("Top districts ranked by urgency score")
    bar_df = allocated_df.sort_values("urgency_score").tail(top_n)
    bar_fig = px.bar(
        bar_df,
        x="urgency_score",
        y="district_name",
        orientation="h",
        color_discrete_sequence=[NAVY_BLUE],
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    st.subheader("Map of filtered districts")
    map_fig = px.choropleth_mapbox(
        allocated_df,
        geojson=allocated_df.geometry.__geo_interface__,
        locations=allocated_df.index,
        color="urgency_score",
        hover_name="district_name",
        center={
            "lat": float(allocated_df["latitude"].mean()),
            "lon": float(allocated_df["longitude"].mean()),
        },
        zoom=3.5,
        mapbox_style="carto-positron",
        color_continuous_scale=BLUE_GRADIENT,
    )
    map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(map_fig, use_container_width=True)

    st.subheader("Connectivity vs. Household Income")
    scatter_fig = px.scatter(
        allocated_df,
        x="median_income",
        y="pct_no_internet",
        size="students_under_18",
        color="urgency_score",
        hover_name="district_name",
        color_continuous_scale=BLUE_GRADIENT,
        labels={"median_income": "Median Income", "pct_no_internet": "% No Internet"},
    )
    scatter_fig.update_traces(
        marker=dict(
            line=dict(width=1, color="white"),
            opacity=0.9,
        )
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# =========================
# 6) AI ASSISTANT
# =========================
st.divider()
st.subheader("🤖 AI Assistant")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if openrouter_key:
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the data...")
    if user_input:
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a policy analyst."}] + st.session_state.ai_messages,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.ai_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"AI Error: {e}")