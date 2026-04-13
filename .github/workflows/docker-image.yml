name: Docker Image Build

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]

jobs:

  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    
    - name: Build the Docker image
      run: docker build . --file Dockerfile --tag weather-forecasting:$(date +%s)
