from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor  # Fallback model
import os
import uvicorn
from typing import List

# --- TensorFlow Robust Loading ---
HAS_TENSORFLOW = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TENSORFLOW = True
    print("✅ TensorFlow loaded successfully. Using LSTM model.")
except ImportError:
    print("⚠️ TensorFlow failed to load (DLL issue). Using RandomForest fallback.")

app = FastAPI(title="Weather Forecasting API")

# Setup templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Data Logic ---

def get_continent(country_name):
    """Maps a country name to its continent."""
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

# Cache for the dataset
data_cache = None

def load_data():
    global data_cache
    if data_cache is None:
        if not os.path.exists("weather.csv"):
            raise FileNotFoundError("weather.csv not found")
        
        df = pd.read_csv("weather.csv")
        df["last_updated"] = pd.to_datetime(df["last_updated"])
        
        # Optimize continent mapping
        unique_countries = df["country"].unique()
        country_to_region = {c: get_continent(c) for c in unique_countries}
        df["region"] = df["country"].map(country_to_region)
        
        data_cache = df
    return data_cache

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/regions")
async def get_regions():
    try:
        df = load_data()
        regions = sorted(df["region"].unique().tolist())
        return {"regions": regions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/{region}")
async def get_forecast(region: str):
    try:
        df = load_data()
        data_region = df[df["region"] == region].copy()
        if data_region.empty:
            raise HTTPException(status_code=404, detail="Region not found")
        
        data_region = data_region.sort_values("last_updated")
        features = ["temperature_celsius", "humidity", "wind_mph"]
        data = data_region[["last_updated"] + features].set_index("last_updated")
        data = data.resample("1H").mean().interpolate()
        
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
            raise HTTPException(status_code=400, detail="Insufficient data for forecasting in this region")

        if HAS_TENSORFLOW:
            # --- LSTM Architecture ---
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                LSTM(16),
                Dense(len(features))
            ])
            model.compile(loss="mse", optimizer="adam")
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            # Predict
            future_pred = []
            last_data = scaled_data[-time_step:]
            for _ in range(24):
                pred = model.predict(last_data.reshape(1, time_step, len(features)), verbose=0)[0]
                future_pred.append(pred)
                last_data = np.vstack([last_data[1:], pred])
        else:
            # --- Scikit-Learn Fallback (RandomForest) ---
            # Flatten X for traditional ML model
            X_flat = X.reshape(X.shape[0], -1)
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_flat, y)
            
            # Predict recursively
            future_pred = []
            last_data = scaled_data[-time_step:]
            for _ in range(24):
                pred = model.predict(last_data.reshape(1, -1))[0]
                future_pred.append(pred)
                last_data = np.vstack([last_data[1:], pred])
        
        future_pred = np.array(future_pred)
        future_pred = scaler.inverse_transform(future_pred)
        
        # Format response
        forecast = []
        for i, pred in enumerate(future_pred):
            forecast.append({
                "hour": i + 1,
                "temp": float(pred[0]),
                "humidity": float(pred[1]),
                "wind": float(pred[2])
            })
            
        return {
            "region": region,
            "forecast": forecast,
            "model_type": "LSTM" if HAS_TENSORFLOW else "RandomForest (Fallback)",
            "stats": {
                "avg_temp": float(np.mean(future_pred[:, 0])),
                "avg_humidity": float(np.mean(future_pred[:, 1])),
                "avg_wind": float(np.mean(future_pred[:, 2]))
            }
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
