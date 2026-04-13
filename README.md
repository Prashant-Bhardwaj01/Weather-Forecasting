# 🌤 WeatherAI - LSTM Forecasting Dashboard

A professional, full-stack weather forecasting application built with **FastAPI**, **Deep Learning (LSTM)**, and a custom **Vanilla Web Frontend**. This project demonstrates a robust machine learning deployment architecture, making it ideal for technical interviews.

## 🚀 Overview

WeatherAI uses Long Short-Term Memory (LSTM) networks to predict global weather patterns. Unlike simplified prototypes, this project features a dedicated **FastAPI** backend and a high-end, responsive **Vanilla JS/CSS** dashboard with real-time visualizations using **Chart.js**.

### Key Features:
- **Full-Stack Architecture**: Clean separation of concerns between AI logic and UI.
- **Deep Learning Model**: Multivariate LSTM architecture for time-series forecasting.
- **Premium Dashboard**: Custom-built UI with Glassmorphism, dynamic charts, and metrics.
- **Modern DevOps**: Fully containerized with Docker and GitHub Actions for automated builds.

## 🛠 Tech Stack

- **Backend**: FastAPI (Python), Uvicorn, Jinja2
- **Deep Learning**: TensorFlow, Scikit-Learn (MinMaxScaler)
- **Data Engineering**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **DevOps**: Docker, GitHub Actions

## 📦 Setup & Installation

### Local Execution (Standard)
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   uvicorn backend:app --reload
   ```
3. **Open in Browser**: Navigate to `http://localhost:8000`

### Docker Execution (Recommended)
1. **Build the Image**:
   ```bash
   docker build -t weather-forecast .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8000:8000 weather-forecast
   ```
   Navigate to `http://localhost:8000`

## 🧠 Model Architecture
The forecasting engine uses a Multivariate LSTM model:
- **Input**: 24-hour sequence of (Temperature, Humidity, Wind Speed).
- **Layers**: 2 Stacked LSTM layers to capture temporal dependencies.
- **Output**: Dense layer predicting the next 24-hour window.

## 🤖 CI/CD (GitHub Actions)
A GitHub Actions workflow is included in `.github/workflows/docker-image.yml` to automatically build the Docker image on every push to the `main` or `master` branches.

## 📄 License
This project is open-source and available under the MIT License.
