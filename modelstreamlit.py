import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pycountry_convert as pc
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Weather Forecasting LSTM",
    page_icon="🌤",
    layout="wide"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Custom Card Style */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-image: linear-gradient(to right, #00d2ff 0%, #3a7bd5 51%, #00d2ff 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: 0.5s;
        background-size: 200% auto;
    }
    
    .stButton>button:hover {
        background-position: right center;
    }

    /* DataFrame Styling */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions & Caching ---

@st.cache_data
def get_continent(country_name):
    """Maps a country name to its continent using pycountry_convert."""
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = {
            "AF": "Africa", "AS": "Asia", "EU": "Europe",
            "NA": "North America", "SA": "South America",
            "OC": "Oceania", "AN": "Antarctica"
        }
        return continent_name.get(continent_code, "Unknown")
    except:
        return "Unknown"

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads dataset and adds continent/region column if missing."""
    df = pd.read_csv(file_path)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    
    if "region" not in df.columns:
        # Optimization: Map unique countries only
        unique_countries = df["country"].unique()
        country_to_region = {country: get_continent(country) for country in unique_countries}
        df["region"] = df["country"].map(country_to_region)
    
    return df

# --- Main Application ---

def main():
    st.title("🌤 Weather Forecasting using LSTM")
    st.write("Predicting future weather patterns using Deep Learning (Long Short-Term Memory).")

    # Load Data
    with st.spinner("⏳ Loading global weather dataset..."):
        try:
            df = load_and_preprocess_data("weather.csv")
        except FileNotFoundError:
            st.error("Dataset 'weather.csv' not found. Please ensure it is in the project directory.")
            return

    # Sidebar Controls
    st.sidebar.header("🕹 Control Panel")
    regions = sorted(df["region"].unique())
    selected_region = st.sidebar.selectbox("🌍 Select Region", regions)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This tool uses an LSTM model to forecast temperature, humidity, and wind speed for the next 24 hours based on historical trends."
    )

    # Hero Section / Introduction
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Analyzing Weather Data for: {selected_region}")
        region_stats = df[df["region"] == selected_region].describe()
        st.write(f"Showing historical summary for the {selected_region} region based on {len(df[df['region'] == selected_region])} recorded data points.")
    
    with col2:
        st.image("https://img.icons8.com/isometric-line/200/clouds.png", width=150)

    # Forecasting Logic
    if st.sidebar.button("🚀 Generate 24hr Forecast"):
        with st.status(f"Training LSTM Model for {selected_region}...", expanded=True) as status:
            data_region = df[df["region"] == selected_region].copy()
            data_region = data_region.sort_values("last_updated")
            
            # Feature Selection
            features = ["temperature_celsius", "humidity", "wind_mph"]
            data = data_region[["last_updated"] + features].set_index("last_updated")
            data = data.resample("1h").mean().interpolate()
            
            # Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Prepare Sequences
            time_step = 24
            X, y = [], []
            for i in range(len(scaled_data) - time_step):
                X.append(scaled_data[i:i + time_step])
                y.append(scaled_data[i + time_step])
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 1:
                st.error("Not enough data to train the model for this region.")
                return

            st.write("Building LSTM Architecture...")
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                LSTM(32),
                Dense(len(features))
            ])
            model.compile(loss="mse", optimizer="adam")
            
            st.write("Fitting model (Learning patterns)...")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            st.write("Generating Predictions...")
            future_pred = []
            last_data = scaled_data[-time_step:]
            for _ in range(24):
                pred = model.predict(last_data.reshape(1, time_step, len(features)), verbose=0)[0]
                future_pred.append(pred)
                last_data = np.vstack([last_data[1:], pred])
            
            future_pred = scaler.inverse_transform(future_pred)
            status.update(label="Forecast Complete!", state="complete", expanded=False)

        # Display Results
        st.divider()
        st.subheader(f"📅 24-Hour Forecast Results")
        
        forecast_df = pd.DataFrame(future_pred, columns=["Temp (°C)", "Humidity (%)", "Wind (mph)"])
        forecast_df.index = [f"H +{i}" for i in range(1, 25)]
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Temp", f"{forecast_df['Temp (°C)'].mean():.2f} °C")
        m2.metric("Avg Humidity", f"{forecast_df['Humidity (%)'].mean():.1f} %")
        m3.metric("Avg Wind", f"{forecast_df['Wind (mph)'].mean():.1f} mph")

        tab1, tab2 = st.tabs(["📈 Visualizations", "📄 Detailed Data"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            ax.plot(forecast_df["Temp (°C)"], label="Temperature (°C)", color="#00d2ff", linewidth=2)
            ax.plot(forecast_df["Humidity (%)"], label="Humidity (%)", color="#3a7bd5", linewidth=2)
            ax.plot(forecast_df["Wind (mph)"], label="Wind Speed (mph)", color="#ffffff", linestyle="--")
            ax.set_title(f"Forecasted Trends for {selected_region}", color="white")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)

        with tab2:
            st.dataframe(forecast_df, use_container_width=True)

    # Technical Details
    with st.expander("🛠 Technical Implementation Details"):
        st.markdown("""
        ### Model Architecture:
        - **Type**: Multivariate Long Short-Term Memory (LSTM)
        - **Input**: 24-hour sequence of (Temperature, Humidity, Wind Speed)
        - **Layers**: 2 Stacked LSTM layers + 1 Dense Output layer
        - **Preprocessing**: MinMaxScaler (range [0,1])
        
        ### Caching Mechanism:
        - `st.cache_data`: Applied to data loading and continent mapping to minimize latency.
        - `st.status`: Used for real-time progress tracking during model training.
        """)

if __name__ == "__main__":
    main()
