import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pycountry_convert as pc

def country_to_continent(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "SA": "South America",
            "OC": "Oceania",
            "AN": "Antarctica"
        }
        return continent_name[continent_code]
    except:
        return "Unknown"

df = pd.read_csv("weather.csv")   # <-- Your dataset
df["last_updated"] = pd.to_datetime(df["last_updated"])


df["region"] = df["country"].apply(country_to_continent)


if "region" not in df.columns:
    st.error("❌ No 'region' column found. Create region column first!")
    st.stop()


st.title("🌤 Weather Forecasting using LSTM")
st.write("Forecast Temperature, Humidity, Wind Speed based on selected region.")

# Add gradient background and white text color
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #70003A, #320065);
        background-size: 400% 400%; /* Larger background size for animation */
        animation: gradientAnimation 10s ease infinite; /* Animation definition */
        color: white; /* General text color for the app */
    }
    /* Keyframes for gradient animation */
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Ensure all headings are white */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* Ensure paragraph text is white */
    p {
        color: white !important;
    }
    /* Streamlit's primary header for titles (fallback/specific targets) */
    .st-emotion-cache-16txt4v {{ color: white !important; }}
    .st-emotion-cache-czk5ad {{ color: white !important; }}
    .st-emotion-cache-1wq06v7 {{ color: white !important; }}
    .st-emotion-cache-ue6h4q {{ color: white !important; }}
    /* Selectbox label color */
    .st-emotion-cache-nahz7x p {{ color: white !important; }}
    /* Button transparency */
    .st-emotion-cache-lgl2c4 {{ background-color: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.4); color: white !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

regions = sorted(df["region"].unique())
region = st.selectbox("🌍 Select Region", regions)

if st.button("Generate Forecast"):
    st.success(f"Training LSTM model for **{region}** region... Please wait")

    # Filter data for region
    data_region = df[df["region"] == region].copy()
    data_region = data_region.sort_values("last_updated")

    # Select features
    data = data_region[["last_updated","temperature_celsius","humidity","wind_mph"]]
    data = data.set_index("last_updated")
    data = data.resample("1H").mean().interpolate()

    # Scale values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Create sequences (24 hrs → next hour)
    X, y = [], []
    time_step = 24

    for i in range(len(scaled)-time_step):
        X.append(scaled[i:i+time_step])
        y.append(scaled[i+time_step])

    X, y = np.array(X), np.array(y)

    # Build LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(3)  # Temperature, Humidity, Wind
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(X, y, epochs=8, batch_size=32, verbose=0)

    # Forecast next 24 hours
    future_pred = []
    last_data = scaled[-time_step:]

    for _ in range(24):
        pred = model.predict(last_data.reshape(1,time_step,3), verbose=0)[0]
        future_pred.append(pred)
        last_data = np.vstack([last_data[1:], pred])

    # Inverse scaling to original values
    future_pred = scaler.inverse_transform(future_pred)

    # RESULTS
    st.subheader(f"📅 24-Hour Forecast for {region}")
    forecast_df = pd.DataFrame(future_pred, columns=["Temperature °C","Humidity %","Wind mph"])
    forecast_df.index = [f"H +{i}" for i in range(1,25)]
    st.dataframe(forecast_df)

    # PLOTS
    st.subheader("📈 Forecast Graph")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(forecast_df["Temperature °C"], label="Temp")
    ax.plot(forecast_df["Humidity %"] , label="Humidity")
    ax.plot(forecast_df["Wind mph"], label="Wind")
    ax.legend()
    st.pyplot(fig)