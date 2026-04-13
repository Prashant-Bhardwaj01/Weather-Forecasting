# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies with retries and timeout
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Healthcheck to ensure the container is running correctly
HEALTHCHECK CMD curl --fail http://localhost:8000/api/regions

# Command to run the application
ENTRYPOINT ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
