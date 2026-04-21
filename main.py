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

    vars_main = [
        "B09001_001E",  # students under 18
        "B28002_001E",  # total households internet
        "B28002_013E",  # no internet
        "B28001_001E",  # total households computer
        "B28001_011E",  # no computer
    ]
    s1901_income = "S1901_C01_012E"

    dfs_main = []
    dfs_income = []
    gdfs = []

    for fips in state_fips_tuple:
        geo_in = f"state:{fips}"

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
        j_m = r_m.json()
        dfs_main.append(pd.DataFrame(j_m[1:], columns=j_m[0]))

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
        j_i = r_i.json()
        dfs_income.append(pd.DataFrame(j_i[1:], columns=j_i[0]))

        tiger_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/School/MapServer/10/query"
            f"?where=STATE='{fips}'&outFields=GEOID,NAME,STATE&outSR=4326&f=geojson&returnGeometry=true"
        )
        try:
            t_gdf = gpd.read_file(tiger_url)
            if not t_gdf.empty:
                gdfs.append(t_gdf)
        except Exception:
            continue

    if not dfs_main or not dfs_income:
        raise ValueError("No Census data could be loaded for the selected state(s).")

    df_main = pd.concat(dfs_main, ignore_index=True)
    df_income = pd.concat(dfs_income, ignore_index=True)

    df = df_main.merge(
        df_income[["state", "school district (unified)", s1901_income]],
        on=["state", "school district (unified)"],
        how="left",
    )

    for c in vars_main + [s1901_income]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["GEOID"] = df["state"].str.zfill(2) + df["school district (unified)"].str.zfill(5)
    df["district_name"] = df["NAME_x"] if "NAME_x" in df.columns else df["NAME"]
    df["students_under_18"] = df["B09001_001E"]
    df["households_no_internet"] = df["B28002_013E"]
    df["households_no_computer"] = df["B28001_011E"]
    df["median_income"] = df[s1901_income]

    df["pct_no_internet"] = (
        df["households_no_internet"] / df["B28002_001E"].replace(0, np.nan)
    ).fillna(0)

    max_students = df["students_under_18"].max()
    if pd.isna(max_students) or max_students == 0:
        student_component = pd.Series(0.0, index=df.index)
    else:
        student_component = df["students_under_18"] / max_students

    df["urgency_score"] = (student_component * 0.4 + df["pct_no_internet"] * 0.6).fillna(0)
    df["access_level"] = df["pct_no_internet"].apply(categorize_access)

    if not gdfs:
        raise ValueError("No geometry data could be loaded for the selected state(s).")

    full_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    simplify_tol = 0.015 if len(state_fips_tuple) > 20 else 0.005
    full_gdf["geometry"] = full_gdf["geometry"].simplify(
        tolerance=simplify_tol,
        preserve_topology=True,
    )

    gdf = full_gdf.merge(df, on="GEOID", how="inner")

    centroids = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    gdf["region"] = gdf["longitude"].apply(assign_region)

    return df, gdf


def simulate_allocation(df, total_laptops):
    sim = df.copy()
    weight = sim["urgency_score"].fillna(0).clip(lower=0)

    if sim.empty or total_laptops <= 0 or weight.sum() == 0:
        sim["laptops_allocated"] = 0
        sim["students_helped_est"] = 0
        return sim

    raw_allocation = (weight / weight.sum()) * total_laptops
    base_allocation = np.floor(raw_allocation).astype(int)

    remaining = total_laptops - int(base_allocation.sum())
    fractional_order = (raw_allocation - np.floor(raw_allocation)).sort_values(ascending=False).index.tolist()

    final_alloc = base_allocation.copy()
    for idx in fractional_order:
        if remaining <= 0:
            break
        final_alloc.loc[idx] += 1
        remaining -= 1

    sim["laptops_allocated"] = final_alloc.astype(int)
    sim["students_helped_est"] = sim[["laptops_allocated", "students_under_18"]].min(axis=1)
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

    is_all_states = "All States" in selected_states

    if is_all_states:
        selected_fips = ALL_STATE_FIPS
        st.warning("National view may take longer to load. The map is capped at the top 300 districts for performance.")
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

try:
    df_raw, gdf_map = load_homework_gap_data(api_key, DEFAULT_YEAR, selected_fips)
except Exception as e:
    st.error(f"Could not load data for the selected state(s): {e}")
    st.stop()

with st.sidebar:
    income_nonnull = df_raw["median_income"].dropna()
    income_max = int(income_nonnull.max()) if not income_nonnull.empty else 250000

    income_range = st.slider(
        "Median household income range",
        0,
        income_max,
        (0, income_max),
    )

    region_options = sorted(gdf_map["region"].dropna().unique().tolist()) if "region" in gdf_map.columns else ["West", "Central", "East"]
    selected_regions = st.multiselect(
        "Region",
        options=region_options,
        default=region_options,
    )

    access_options = ["Critical", "High", "Moderate", "Lower", "Unknown"]
    selected_access = st.multiselect(
        "Access level",
        options=access_options,
        default=["Critical", "High", "Moderate", "Lower"],
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
        value=3000,
    )

    top_n = st.slider("Show top districts", 5, 100, 20)

# =========================
# 3) DATA PROCESSING
# =========================
filtered_gdf = gdf_map[
    (gdf_map["median_income"].isna() | gdf_map["median_income"].between(income_range[0], income_range[1])) &
    (gdf_map["region"].isin(selected_regions)) &
    (gdf_map["access_level"].isin(selected_access)) &
    ((gdf_map["pct_no_internet"] * 100).between(gap_range[0], gap_range[1]))
].copy()

allocated_df = simulate_allocation(filtered_gdf, int(laptops_available))

# =========================
# 4) DASHBOARD METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Districts in scope", f"{len(allocated_df):,}")
m2.metric("Students under 18", f"{int(allocated_df['students_under_18'].fillna(0).sum()):,}" if not allocated_df.empty else "0")
m3.metric("Households with no internet", f"{int(allocated_df['households_no_internet'].fillna(0).sum()):,}" if not allocated_df.empty else "0")
m4.metric("Households with no computer", f"{int(allocated_df['households_no_computer'].fillna(0).sum()):,}" if not allocated_df.empty else "0")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Laptops allocated", f"{int(allocated_df['laptops_allocated'].sum()):,}" if not allocated_df.empty else "0")
m6.metric("Estimated students helped", f"{int(allocated_df['students_helped_est'].sum()):,}" if not allocated_df.empty else "0")
m7.metric("Avg. laptops per district", f"{allocated_df['laptops_allocated'].mean():.2f}" if not allocated_df.empty else "0.00")
m8.metric("Largest district allocation", f"{int(allocated_df['laptops_allocated'].max()):,}" if not allocated_df.empty else "0")

if allocated_df.empty:
    st.warning("No districts match the current filters.")
    st.stop()

# =========================
# 5) VISUALS
# =========================
left, right = st.columns((1, 1))

with left:
    st.subheader("Urgency ranking")
    table_cols = [
        "district_name",
        "region",
        "access_level",
        "students_under_18",
        "pct_no_internet",
        "median_income",
        "urgency_score",
        "laptops_allocated",
    ]
    ranking_df = allocated_df.sort_values("urgency_score", ascending=False).head(top_n).copy()
    ranking_df["pct_no_internet"] = (ranking_df["pct_no_internet"] * 100).round(1)
    ranking_df["urgency_score"] = ranking_df["urgency_score"].round(3)

    st.dataframe(
        ranking_df[table_cols],
        use_container_width=True,
        hide_index=True,
    )

    bar_df = allocated_df.sort_values("urgency_score", ascending=False).head(top_n).copy()
    urgency_fig = px.bar(
        bar_df.sort_values("urgency_score", ascending=True),
        x="urgency_score",
        y="district_name",
        orientation="h",
        hover_data={
            "students_under_18": True,
            "pct_no_internet": ':.1%',
            "median_income": ':$,.0f',
            "laptops_allocated": True,
        },
        title="Top districts ranked by urgency score",
    )
    urgency_fig.update_layout(
        xaxis_title="Urgency score",
        yaxis_title="District",
        height=550,
    )
    st.plotly_chart(urgency_fig, use_container_width=True)

with right:
    st.subheader("Map of filtered districts")

    MAP_DISTRICT_CAP = 300
    map_df = allocated_df.sort_values("urgency_score", ascending=False).head(MAP_DISTRICT_CAP).copy()

    if len(allocated_df) > MAP_DISTRICT_CAP:
        st.caption(
            f"⚠️ Showing top {MAP_DISTRICT_CAP} of {len(allocated_df):,} filtered districts on the map "
            "(ranked by urgency score). Use the sidebar filters or 'Show top districts' slider to narrow down."
        )

    if map_df.empty:
        st.warning("Adjust filters to see data on map.")
    else:
        zoom = 3.2 if len(selected_fips) >= 30 else 4.0 if len(selected_fips) >= 10 else 5.5

        map_fig = px.choropleth_mapbox(
            map_df,
            geojson=map_df.geometry.__geo_interface__,
            locations=map_df.index,
            color="urgency_score",
            hover_name="district_name",
            hover_data={
                "students_under_18": True,
                "pct_no_internet": ':.1%',
                "laptops_allocated": True,
            },
            center={
                "lat": float(map_df["latitude"].mean()),
                "lon": float(map_df["longitude"].mean()),
            },
            zoom=zoom,
            mapbox_style="carto-positron",
            color_continuous_scale=BLUE_GRADIENT,
        )
        map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550)
        st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 6) ALLOCATION VISUALS
# =========================
alloc_m1, alloc_m2, alloc_m3 = st.columns(3)
alloc_positive = int((allocated_df["laptops_allocated"] > 0).sum())
alloc_zero = int((allocated_df["laptops_allocated"] == 0).sum())
alloc_max = int(allocated_df["laptops_allocated"].max()) if not allocated_df.empty else 0

alloc_m1.metric("Districts receiving at least 1 laptop", f"{alloc_positive:,}")
alloc_m2.metric("Districts receiving 0 laptops", f"{alloc_zero:,}")
alloc_m3.metric("Largest district allocation", f"{alloc_max:,}")

hist_left, hist_right = st.columns(2)

with hist_left:
    st.subheader("Distribution of laptop allocations across districts")

    hist_df = allocated_df.copy()
    max_alloc = int(hist_df["laptops_allocated"].max()) if not hist_df.empty else 0
    nbins = min(50, max(10, max_alloc + 1))

    hist_fig = px.histogram(
        hist_df,
        x="laptops_allocated",
        nbins=nbins,
        hover_data={
            "district_name": False,
            "urgency_score": False,
        },
        title="How many districts received each allocation amount",
    )
    hist_fig.update_layout(
        xaxis_title="Laptops allocated",
        yaxis_title="Number of districts",
        height=450,
    )
    st.plotly_chart(hist_fig, use_container_width=True)

with hist_right:
    st.subheader("Top districts by allocated laptops")

    top_alloc_df = allocated_df.sort_values(
        ["laptops_allocated", "urgency_score"],
        ascending=[False, False]
    ).head(15).copy()

    if top_alloc_df.empty or top_alloc_df["laptops_allocated"].sum() == 0:
        st.info("Enter more than 0 laptops to see top recipient districts.")
    else:
        top_alloc_fig = px.bar(
            top_alloc_df.sort_values(["laptops_allocated", "urgency_score"], ascending=[True, True]),
            x="laptops_allocated",
            y="district_name",
            orientation="h",
            hover_data={
                "urgency_score": ':.3f',
                "students_under_18": True,
                "students_helped_est": True,
                "pct_no_internet": ':.1%',
            },
            title="Top districts receiving laptops",
        )
        top_alloc_fig.update_layout(
            xaxis_title="Allocated laptops",
            yaxis_title="District",
            height=450,
        )
        st.plotly_chart(top_alloc_fig, use_container_width=True)

scatter_left, scatter_right = st.columns(2)

with scatter_left:
    scatter_fig = px.scatter(
        allocated_df,
        x="median_income",
        y="pct_no_internet",
        size="students_under_18",
        color="urgency_score",
        hover_name="district_name",
        hover_data={
            "laptops_allocated": True,
            "students_helped_est": True,
        },
        color_continuous_scale=BLUE_GRADIENT,
        labels={"median_income": "Median Income", "pct_no_internet": "% No Internet"},
        title="Connectivity vs. household income",
    )
    scatter_fig.update_traces(
        marker=dict(
            line=dict(width=1, color="white"),
            opacity=0.9,
        )
    )
    scatter_fig.update_layout(height=450)
    st.plotly_chart(scatter_fig, use_container_width=True)

with scatter_right:
    region_alloc = (
        allocated_df.groupby("region", dropna=False)["laptops_allocated"]
        .sum()
        .reset_index()
        .sort_values("laptops_allocated", ascending=False)
    )

    region_fig = px.bar(
        region_alloc,
        x="region",
        y="laptops_allocated",
        title="Total laptops allocated by region",
    )
    region_fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Allocated laptops",
        height=450,
    )
    st.plotly_chart(region_fig, use_container_width=True)

# =========================
# 7) AI ASSISTANT
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