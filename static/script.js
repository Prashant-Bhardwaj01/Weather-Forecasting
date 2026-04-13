document.addEventListener('DOMContentLoaded', () => {
    const regionSelect = document.getElementById('regionSelect');
    const forecastBtn = document.getElementById('forecastBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const forecastTableBody = document.querySelector('#forecastTable tbody');
    
    let forecastChart = null;
    let distChart = null;

    // 1. Fetch available regions
    fetch('/api/regions')
        .then(response => response.json())
        .then(data => {
            regionSelect.innerHTML = '<option value="" disabled selected>-- Select a Region --</option>';
            data.regions.forEach(region => {
                const option = document.createElement('option');
                option.value = region;
                option.textContent = region;
                regionSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching regions:', error);
            alert('Failed to load regions. Please check if the backend is running.');
        });

    // 2. Handle forecast generation
    forecastBtn.addEventListener('click', () => {
        const region = regionSelect.value;
        if (!region) {
            alert('Please select a region first!');
            return;
        }

        // Reset UI
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        forecastTableBody.innerHTML = '';

        // Fetch forecast
        fetch(`/api/forecast/${region}`)
            .then(response => {
                if (!response.ok) throw new Error('Forecast generation failed');
                return response.json();
            })
            .then(data => {
                updateDashboard(data);
                loading.classList.add('hidden');
                results.classList.remove('hidden');
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error generating forecast. Check console for details.');
                loading.classList.add('hidden');
            });
    });

    function updateDashboard(data) {
        console.log(`Forecast powered by: ${data.model_type}`);
        // Update Metrics
        document.getElementById('avgTemp').textContent = `${data.stats.avg_temp.toFixed(2)} °C`;
        document.getElementById('avgHumidity').textContent = `${data.stats.avg_humidity.toFixed(1)} %`;
        document.getElementById('avgWind').textContent = `${data.stats.avg_wind.toFixed(1)} mph`;

        // Update Table
        data.forecast.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>H +${item.hour}</td>
                <td>${item.temp.toFixed(2)}</td>
                <td>${item.humidity.toFixed(1)}</td>
                <td>${item.wind.toFixed(1)}</td>
            `;
            forecastTableBody.appendChild(row);
        });

        // Update Trend Chart
        updateTrendChart(data);
        
        // Update Distribution Chart
        updateDistributionChart(data);
    }

    function updateTrendChart(data) {
        const labels = data.forecast.map(item => `H+${item.hour}`);
        const temps = data.forecast.map(item => item.temp);
        const humidities = data.forecast.map(item => item.humidity);
        const winds = data.forecast.map(item => item.wind);

        if (forecastChart) {
            forecastChart.destroy();
        }

        const ctx = document.getElementById('forecastChart').getContext('2d');
        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: temps,
                        borderColor: '#00d2ff',
                        backgroundColor: 'rgba(0, 210, 255, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Humidity (%)',
                        data: humidities,
                        borderColor: '#3a7bd5',
                        backgroundColor: 'rgba(58, 123, 213, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Wind Speed (mph)',
                        data: winds,
                        borderColor: '#ffffff',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'white' } }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                    }
                }
            }
        });
    }

    function updateDistributionChart(data) {
        if (distChart) {
            distChart.destroy();
        }

        const ctx = document.getElementById('distChart').getContext('2d');
        distChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Avg Temp (°C)', 'Avg Humidity (%)', 'Avg Wind (mph)'],
                datasets: [{
                    label: 'Regional Distribution',
                    data: [data.stats.avg_temp, data.stats.avg_humidity, data.stats.avg_wind],
                    backgroundColor: 'rgba(0, 210, 255, 0.2)',
                    borderColor: '#00d2ff',
                    pointBackgroundColor: '#00d2ff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'white' } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: 'white' },
                        ticks: { display: false, backdropColor: 'transparent' }
                    }
                }
            }
        });
    }
});
