import os
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================
# 0) CONFIG & COLOR THEME
# =========================
DEFAULT_YEAR = 2022 
STATES = {
    "North Carolina": "37", "South Carolina": "45", "Virginia": "51", "Georgia": "13",
    "Maryland": "24", "Florida": "12", "Tennessee": "47", "Alabama": "01"
}

# Strictly Blue Theme - No Purples
NAVY_BLUE = "#003366"
BLUE_GRADIENT = ["#deebf7", "#9ecae1", "#3182bd", "#08519c", "#002147"]

st.set_page_config(page_title="Homework Gap Dashboard", layout="wide")

# =========================
# 1) HELPERS
# =========================
def assign_region(longitude):
    if pd.isna(longitude): return "Unknown"
    if longitude <= -80: return "West"
    if longitude <= -78: return "Central"
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

    df_m = requests.get(acs_base, params={"get": "NAME," + ",".join(vars_main), "for": geo_for, "in": geo_in, "key": api_key}).json()
    df_i = requests.get(acs_sub, params={"get": "NAME," + s1901_income, "for": geo_for, "in": geo_in, "key": api_key}).json()

    df = pd.DataFrame(df_m[1:], columns=df_m[0]).merge(pd.DataFrame(df_i[1:], columns=df_i[0]), on=["state", "school district (unified)"])
    
    for c in vars_main + [s1901_income]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["GEOID"] = df["state"].str.zfill(2) + df["school district (unified)"].str.zfill(5)
    df["district_name"] = df["NAME_x"]
    df["students_under_18"] = df["B09001_001E"]
    df["median_income"] = df[s1901_income]
    df["pct_no_internet"] = (df["B28002_013E"] / df["B28002_001E"].replace(0, np.nan)).fillna(0)
    
    df["urgency_score"] = ((df["B09001_001E"] / df["B09001_001E"].max()) * 0.4 + df["pct_no_internet"] * 0.6).fillna(0)
    df["access_level"] = df["pct_no_internet"].apply(categorize_access)

    gdfs = []
    for fips in state_fips_tuple:
        for lid in [10, 11, 12]:
            url = f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/School/MapServer/{lid}/query?where=STATE='{fips}'&outFields=GEOID,NAME&outSR=4326&f=geojson"
            try:
                t_gdf = gpd.read_file(url)
                if not t_gdf.empty: gdfs.append(t_gdf)
            except: continue
    
    full_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    data_gdf = full_gdf[full_gdf['GEOID'].isin(df['GEOID'])].merge(df, on="GEOID")
    gdf = full_gdf.sjoin_nearest(data_gdf[['urgency_score', 'district_name', 'pct_no_internet', 'median_income', 'access_level', 'students_under_18', 'geometry']], how="left")
    
    centroids = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    gdf["longitude"], gdf["latitude"] = centroids.x, centroids.y
    gdf["region"] = gdf["longitude"].apply(assign_region)
    
    return df, gdf

# =========================
# 2) APP UI & SIDEBAR
# =========================
st.title("District Impact Dashboard")

with st.sidebar:
    st.header("Data and Filters")
    api_key = st.text_input("Census API key", type="password")
    
    selected_states = st.multiselect("State(s)", options=sorted(STATES.keys()), default=["North Carolina"])
    selected_fips = tuple(STATES[s] for s in selected_states)
    
    if api_key:
        temp_df, _ = load_homework_gap_data(api_key, 2022, selected_fips)
        income_range = st.slider("Median household income range", 0, int(temp_df["median_income"].max() or 150000), (0, 150000))
        region_options = ["West", "Central", "East"]
        selected_regions = st.multiselect("Region", options=region_options, default=region_options)
        access_options = ["Critical", "High", "Moderate", "Lower"]
        selected_access = st.multiselect("Access level", options=access_options, default=access_options)
        gap_range = st.slider("Connectivity gap (%)", 0.0, 100.0, (0.0, 100.0))
        top_n = st.slider("Show top districts", 5, 50, 15, 5)

if not api_key:
    st.info("Enter API Key in the sidebar to start.")
    st.stop()

df_raw, gdf_map = load_homework_gap_data(api_key, 2022, selected_fips)

filtered_gdf = gdf_map[
    (gdf_map["median_income"].between(income_range[0], income_range[1])) &
    (gdf_map["region"].isin(selected_regions)) &
    (gdf_map["access_level"].isin(selected_access)) &
    (gdf_map["pct_no_internet"] * 100).between(gap_range[0], gap_range[1])
].copy()

# =========================
# 3) VISUALS (Unified Dark Blue)
# =========================
col1, col2 = st.columns((1, 1))

if filtered_gdf.empty:
    st.warning("No districts match the current filters.")
else:
    with col1:
        st.subheader("Urgency Ranking")
        bar_df = filtered_gdf.sort_values("urgency_score").tail(top_n)
        bar_fig = px.bar(bar_df, 
                         x="urgency_score", y="district_name", orientation="h",
                         color_discrete_sequence=[NAVY_BLUE])
        st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("Regional Impact Analysis")
        # Unified Box plot colors to match Navy Blue
        reg_fig = px.box(filtered_gdf, x="region", y="urgency_score", color="region",
                         color_discrete_sequence=[NAVY_BLUE, "#004080", "#0059b3"])
        st.plotly_chart(reg_fig, use_container_width=True)

    with col2:
        st.subheader("State Coverage Map")
        map_fig = px.choropleth_mapbox(
            filtered_gdf, 
            geojson=filtered_gdf.geometry.__geo_interface__, 
            locations=filtered_gdf.index,
            color="urgency_score", 
            hover_name="district_name",
            center={"lat": float(filtered_gdf["latitude"].mean()), "lon": float(filtered_gdf["longitude"].mean())},
            zoom=5.5, 
            mapbox_style="carto-positron", 
            opacity=1.0, 
            color_continuous_scale=BLUE_GRADIENT # Pure blue gradient
        )
        map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(map_fig, use_container_width=True)

        st.subheader("Connectivity vs. Household Income")
        scatter_fig = px.scatter(
            filtered_gdf, 
            x="median_income", 
            y="pct_no_internet",
            size="students_under_18", 
            color="urgency_score",
            hover_name="district_name", 
            color_continuous_scale=BLUE_GRADIENT, # Pure blue gradient
            labels={"median_income": "Median Income", "pct_no_internet": "% No Internet"}
        )
        
        scatter_fig.update_traces(
            marker=dict(
                line=dict(width=1, color='white'), 
                opacity=0.9
            )
        )
        st.plotly_chart(scatter_fig, use_container_width=True)