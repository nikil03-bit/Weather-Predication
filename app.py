import streamlit as st
import joblib
import pandas as pd
import datetime

# --- 1. PRO CONFIGURATION ---
st.set_page_config(
    page_title=" Weather Predication", 
    page_icon="⛈️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Modern UI) ---
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #dcdcdc;
        padding: 10px;
        border-radius: 10px;
    }
    /* Weather Card Hover Effect */
    .weather-card {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .weather-card:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD THE AI MODEL ---
@st.cache_resource
def load_models():
    try:
        return joblib.load('pontiac_7day_rain_models.pkl')
    except FileNotFoundError:
        return None

pkg = load_models()

# Check if model loaded successfully
if pkg is None:
    st.error("🚨 Critical Error: Model file missing!")
    st.warning("Please run your Training Notebook to generate 'pontiac_7day_rain_models.pkl'.")
    st.stop()

models_t = pkg['temp']
models_rp = pkg['rain_prob']
# We get the EXACT feature list the model was trained on
model_features = pkg['features'] 

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.title(" Control Panel")
st.sidebar.markdown("Adjust parameters to see how the forecast changes.")

with st.sidebar.expander("📅 Date & Time", expanded=True):
    today_date = st.date_input("Analysis Date", datetime.date.today())

with st.sidebar.expander("🌡️ Temperature Data", expanded=True):
    tmax = st.slider("Today's High (°F)", -10, 110, 75)
    tmin = st.slider("Tonight's Low (°F)", -20, 90, 55)
    prev_tmax = st.slider("Yesterday's High (°F)", -10, 110, 70, help="Used to calculate the temperature trend.")

with st.sidebar.expander("💨 Wind & Moisture", expanded=True):
    wind_speed = st.slider("Avg Wind Speed (mph)", 0.0, 40.0, 10.0)
    rain_today = st.number_input("Rainfall Today (in)", 0.0, 10.0, 0.0)

# --- 5. SMART DATA ADAPTER (Prevents Crashes) ---

# A. Create a Dictionary with ALL possible data points
input_dict = {
    'TMAX': tmax,
    'TMIN': tmin,
    'PRCP': rain_today,
    'AWND': wind_speed,
    'month': today_date.month,
    'day_of_year': today_date.timetuple().tm_yday,
    'TMAX_Change': tmax - prev_tmax
}

# B. Check what the model WANTS vs what we HAVE
input_df = pd.DataFrame([input_dict])

for feature in model_features:
    if feature not in input_df.columns:
        # If the model wants "Wind Gust" (WSF5), estimate it
        if feature == 'WSF5':
            input_df['WSF5'] = wind_speed * 1.3 
        # If the model wants "Did it Rain?" (WT01), calculate it
        elif feature == 'WT01':
            input_df['WT01'] = 1 if rain_today > 0.01 else 0
        # If the model wants 3-Day Avg, use today's temp as proxy
        elif feature == 'TMAX_3day_avg':
            input_df['TMAX_3day_avg'] = tmax
        # Fallback: fill with 0 to prevent crash
        else:
            input_df[feature] = 0

# C. Filter: Send ONLY exactly what the model expects, in the correct order
final_input_data = input_df[model_features]

# --- 6. GENERATE PREDICTIONS ---
forecast_data = []

# Loop 7 Days
for day in range(1, 8):
    # Predict using the specific model for that day horizon
    pred_temp = models_t[day].predict(final_input_data)[0]
    pred_prob = models_rp[day].predict_proba(final_input_data)[0][1]
    
    forecast_data.append({
        "Day": day,
        "Date": (today_date + datetime.timedelta(days=day)).strftime('%a %d'),
        "Temp": round(pred_temp, 1),
        "RainProb": round(pred_prob, 2),
        "RainProb%": f"{pred_prob:.0%}"
    })

df_forecast = pd.DataFrame(forecast_data)

# --- 7. VISUALIZATION DASHBOARD ---
st.title(" Weather Predication")
st.markdown(f"**Forecast Horizon:** {today_date.strftime('%B %d')} — {(today_date + datetime.timedelta(days=7)).strftime('%B %d')}")

# Create Tabs for different views
tab1, tab3 = st.tabs(["📊 Dashboard", "💾 Raw Data"])

# TAB 1: THE CARDS (User Friendly)
with tab1:
    cols = st.columns(7)
    for idx, row in df_forecast.iterrows():
        prob = row['RainProb']
        temp = row['Temp']
        
        # Dynamic Icon Logic
        if prob > 0.50:
            bg = "linear-gradient(180deg, #ffffff 0%, #e3f2fd 100%)" # Blue
            icon, status, color = "🌧️", "Rain", "#1565c0"
        elif prob > 0.30:
            bg = "linear-gradient(180deg, #ffffff 0%, #eceff1 100%)" # Grey
            icon, status, color = "🌥️", "Cloudy", "#546e7a"
        elif temp > 85:
            bg = "linear-gradient(180deg, #ffffff 0%, #ffebee 100%)" # Red
            icon, status, color = "🔥", "Hot", "#c62828"
        else:
            bg = "linear-gradient(180deg, #ffffff 0%, #fffde7 100%)" # Yellow
            icon, status, color = "☀️", "Sunny", "#fbc02d"

        # Render HTML Card
        with cols[idx]:
            st.markdown(f"""
            <div class="weather-card" style="background: {bg}; border: 1px solid #eee;">
                <div style="font-weight: bold; color: #555;">{row['Date']}</div>
                <div style="font-size: 35px; margin: 10px 0;">{icon}</div>
                <div style="font-size: 24px; font-weight: 800; color: #333;">{int(temp)}°</div>
                <div style="color: {color}; font-weight: bold; margin-top: 5px;">{status}</div>
                <div style="font-size: 12px; margin-top: 5px; color: #000;">💧 {row['RainProb%']}</div>
            </div>
            """, unsafe_allow_html=True)


# TAB 2: DATA (Transparent)
with tab3:
    st.dataframe(df_forecast.style.background_gradient(subset=['Temp'], cmap='OrRd'), use_container_width=True)
    st.json({"Model Used": "Random Forest Ensemble", "Input Features": model_features})