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
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "Florida": "12", "Georgia": "13", 
    "Hawaii": "15", "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", 
    "Kansas": "20", "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24", 
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28", 
    "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32", "New Hampshire": "33", 
    "New Jersey": "34", "New Mexico": "35", "New York": "36", "North Carolina": "37", 
    "North Dakota": "38", "Ohio": "39", "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42", 
    "Rhode Island": "44", "South Carolina": "45", "South Dakota": "46", "Tennessee": "47", 
    "Texas": "48", "Utah": "49", "Vermont": "50", "Virginia": "51", "Washington": "53", 
    "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56"
}

NAVY_BLUE = "#003366"
BLUE_GRADIENT = ["#deebf7", "#9ecae1", "#3182bd", "#08519c", "#002147"]

st.set_page_config(page_title="Homework Gap Dashboard", layout="wide")

# =========================
# 1) HELPERS
# =========================
def assign_region(longitude):
    if pd.isna(longitude): return "Unknown"
    if longitude <= -104: return "West"
    if longitude <= -90: return "Central"
    return "East"

def categorize_access(gap):
    if pd.isna(gap): return "Unknown"
    if gap >= 0.25: return "Critical"
    if gap >= 0.15: return "High"
    if gap >= 0.08: return "Moderate"
    return "Lower"

@st.cache_data(show_spinner=False)
def load_homework_gap_data(api_key, year=DEFAULT_YEAR, state_fips_tuple=("37",)):
    acs_base = f"https://api.census.gov/data/{year}/acs/acs5"
    acs_sub = f"https://api.census.gov/data/{year}/acs/acs5/subject"
    geo_for, geo_in = "school district (unified):*", f"state:{','.join(state_fips_tuple)}"

    vars_main = ["B09001_001E", "B28002_001E", "B28002_013E", "B28001_001E", "B28001_011E"]
    s1901_income = "S1901_C01_012E"

    r_m = requests.get(acs_base, params={"get": "NAME," + ",".join(vars_main), "for": geo_for, "in": geo_in, "key": api_key}).json()
    r_i = requests.get(acs_sub, params={"get": "NAME," + s1901_income, "for": geo_for, "in": geo_in, "key": api_key}).json()

    df = pd.DataFrame(r_m[1:], columns=r_m[0]).merge(pd.DataFrame(r_i[1:], columns=r_i[0]), on=["state", "school district (unified)"])
    
    for c in vars_main + [s1901_income]: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["GEOID"] = df["state"].str.zfill(2) + df["school district (unified)"].str.zfill(5)
    df["district_name"] = df["NAME_x"]
    df["students_under_18"] = df["B09001_001E"]
    df["households_no_internet"] = df["B28002_013E"]
    df["households_no_computer"] = df["B28001_011E"]
    df["median_income"] = df[s1901_income]
    
    df["pct_no_internet"] = (df["households_no_internet"] / df["B28002_001E"].replace(0, np.nan)).fillna(0)
    df["pct_no_computer"] = (df["households_no_computer"] / df["B28001_001E"].replace(0, np.nan)).fillna(0)
    
    # Advanced Urgency Scoring
    max_students = df["students_under_18"].max()
    max_income = df["median_income"].max()
    child_comp = (df["students_under_18"] / max_students).fillna(0)
    income_comp = (1 - (df["median_income"] / max_income)).fillna(0)
    
    df["urgency_score"] = (child_comp * 0.40 + df["pct_no_internet"] * 0.25 + df["pct_no_computer"] * 0.25 + income_comp * 0.10).fillna(0)
    df["access_level"] = df["pct_no_internet"].apply(categorize_access)

    gdfs = []
    for fips in state_fips_tuple:
        url = f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/School/MapServer/10/query?where=STATE='{fips}'&outFields=GEOID,NAME&outSR=4326&f=geojson"
        try:
            t_gdf = gpd.read_file(url)
            if not t_gdf.empty: gdfs.append(t_gdf)
        except: continue
    
    full_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    gdf = full_gdf.merge(df, on="GEOID")
    centroids = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    gdf["longitude"], gdf["latitude"] = centroids.x, centroids.y
    gdf["region"] = gdf["longitude"].apply(assign_region)
    return df, gdf

def allocate_laptops(df, total_laptops):
    sim = df.copy()
    weights = sim["urgency_score"]
    if weights.sum() == 0 or total_laptops == 0:
        sim["allocated_laptops"] = 0
        return sim
    allocation = (weights / weights.sum()) * total_laptops
    sim["allocated_laptops"] = np.floor(allocation).astype(int)
    sim["allocated_laptops"] = sim[["allocated_laptops", "households_no_computer"]].min(axis=1)
    return sim

# =========================
# 2) APP UI & SIDEBAR
# =========================
st.title("District Impact Dashboard")

with st.sidebar:
    st.header("Data and Filters")
    
    api_key = st.text_input("Census API key", type="password", value=os.getenv("CENSUS_API_KEY", ""))
    selected_states = st.multiselect("State(s)", options=sorted(STATES.keys()), default=["North Carolina"])
    selected_fips = tuple(STATES[s] for s in selected_states)
    
    api_key = st.secrets["CENSUS_API_KEY"]
    if api_key:
        try:
            df_raw, gdf_raw = load_homework_gap_data(api_key, 2022, selected_fips)
            income_range = st.slider("Median household income", 0, int(df_raw["median_income"].max() or 150000), (0, 150000))
            region_options = sorted(df_raw["region"].unique())
            selected_regions = st.multiselect("Region", options=region_options, default=region_options)
            total_laptops = st.number_input("Laptops to Distribute", min_value=0, value=1000)
            top_n = st.slider("Show top districts", 5, 50, 15)
        except:
            st.error("Error fetching data. Check your API Key.")
            st.stop()

df_raw, gdf_map = load_homework_gap_data(api_key, 2022, selected_fips)

filtered_gdf = gdf_map[
    (gdf_map["median_income"].between(income_range[0], income_range[1])) &
    (gdf_map["region"].isin(selected_regions)) &
    (gdf_map["access_level"].isin(selected_access)) &
    (gdf_map["pct_no_internet"] * 100).between(gap_range[0], gap_range[1])
].copy()
if not api_key:
    st.info("Enter API Key in the sidebar to start.")
    st.stop()

# Filtering & Simulation
filtered_df = df_raw[(df_raw["median_income"].between(income_range[0], income_range[1])) & (df_raw["region"].isin(selected_regions))].copy()
simulated_df = allocate_laptops(filtered_df, total_laptops).sort_values("urgency_score", ascending=False)
filtered_gdf = gdf_raw[gdf_raw["GEOID"].isin(simulated_df["GEOID"])].copy()

# =========================
# 3) DASHBOARD KPI ROW (The missing part!)
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Districts", f"{len(simulated_df)}")
m2.metric("Total Students", f"{int(simulated_df['students_under_18'].sum()):,}")
m3.metric("No Internet HH", f"{int(simulated_df['households_no_internet'].sum()):,}")
m4.metric("Laptops Assigned", f"{int(simulated_df['allocated_laptops'].sum()):,}")

# =========================
# 4) VISUALS
# =========================
col1, col2 = st.columns((1, 1))

with col1:
    st.subheader("Urgency Ranking")
    st.dataframe(simulated_df.head(top_n)[["district_name", "urgency_score", "allocated_laptops", "access_level"]], use_container_width=True, hide_index=True)
    bar_fig = px.bar(simulated_df.head(top_n), x="urgency_score", y="district_name", orientation="h", color_discrete_sequence=[NAVY_BLUE])
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    st.subheader("State Coverage Map")
    map_fig = px.choropleth_mapbox(
        filtered_gdf, geojson=filtered_gdf.geometry.__geo_interface__, locations=filtered_gdf.index,
        color="urgency_score", hover_name="district_name",
        center={"lat": float(filtered_gdf["latitude"].mean()), "lon": float(filtered_gdf["longitude"].mean())},
        zoom=5, mapbox_style="carto-positron", color_continuous_scale=BLUE_GRADIENT
    )
    map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 5) AI ASSISTANT
# =========================
st.divider()
st.subheader("🤖 AI Assistant")
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

if openrouter_key:
    if "ai_messages" not in st.session_state: st.session_state.ai_messages = []
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the data...")
    if user_input:
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            response = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[{"role": "system", "content": "You are a data advisor."}] + st.session_state.ai_messages)
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.ai_messages.append({"role": "assistant", "content": answer})