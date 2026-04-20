Homework Gap Dashboard

An interactive Streamlit dashboard for analyzing the digital divide across U.S. school districts. The application uses U.S. Census data to identify areas with limited internet access and simulate equitable laptop distribution strategies.

Overview
This tool enables users to:

Analyze internet and device access gaps by school district
Filter data by state, income, region, and connectivity levels
Rank districts using an urgency scoring model
Simulate laptop allocation based on need
Visualize results through maps and charts
Interact with an AI assistant for data interpretation


Methodology
The urgency score is calculated using:

Urgency Score =
    (Normalized Student Population × 0.4) +
    (Internet Gap % × 0.6)
Laptop allocation is distributed proportionally based on this score.

Tech Stack
Streamlit (UI)
Pandas, NumPy (data processing)
GeoPandas (geospatial analysis)
Plotly (visualization)
U.S. Census API (data source)
OpenRouter / OpenAI-compatible API (AI assistant)


Local Deployment

1. Clone the Repository
git clone https://github.com/DontKnowWhereIAm/The-Digital-Divide-Scouts.git
cd homework-gap-dashboard

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt
If needed:

pip install streamlit pandas numpy geopandas plotly requests python-dotenv openai

4. Configure Environment Variables
Create a .env file:

OPENROUTER_API_KEY=your_openrouter_api_key
Create .streamlit/secrets.toml:

CENSUS_API_KEY = "your_census_api_key"

5. Run the Application
streamlit run app.py
Access the app at:

http://localhost:8501


Deployed Application
Live URL:
https://the-digital-divide-scouts-4jodss6tbebnyudxapn6al.streamlit.app