import requests
import time
import statistics

API_URL = "http://127.0.0.1:8000"

input_data = {
    "Police_Force": 1,
    "Speed_limit": 60,
    "Light_Conditions": "Daylight: Street light present",
    "Weather_Conditions": "Fine without high winds",
    "Road_Surface_Conditions": "Dry",
    "Special_Conditions_at_Site": "Roadworks",
    "Carriageway_Hazards": "None",
    "Urban_or_Rural_Area": 1,
    "Year": 2020
}

predict_times = []
explain_times = []
total_times = []

print("=" * 60)
print("AI TRAFFIC ACCIDENT PROJECT - LATENCY TEST")
print("=" * 60)

for i in range(10):

    # Prediction latency
    start = time.perf_counter()

    predict_response = requests.post(
        f"{API_URL}/predict",
        json=input_data
    )

    predict_end = time.perf_counter()

    predict_latency = (predict_end - start) * 1000
    predict_times.append(predict_latency)

    # SHAP explanation latency
    start = time.perf_counter()

    explain_response = requests.post(
        f"{API_URL}/explain",
        json=input_data
    )

    explain_end = time.perf_counter()

    explain_latency = (explain_end - start) * 1000
    explain_times.append(explain_latency)

    # Total latency
    total_latency = predict_latency + explain_latency
    total_times.append(total_latency)

    print(
        f"Test {i + 1:02d} | "
        f"Prediction: {predict_latency:.2f} ms | "
        f"SHAP: {explain_latency:.2f} ms | "
        f"Total: {total_latency:.2f} ms"
    )

print("\n" + "=" * 60)
print("LATENCY RESULTS")
print("=" * 60)

print(
    f"Average Prediction Latency : "
    f"{statistics.mean(predict_times):.2f} ms"
)

print(
    f"Average SHAP Latency       : "
    f"{statistics.mean(explain_times):.2f} ms"
)

print(
    f"Average Total Latency      : "
    f"{statistics.mean(total_times):.2f} ms"
)

print(
    f"Minimum Total Latency      : "
    f"{min(total_times):.2f} ms"
)

print(
    f"Maximum Total Latency      : "
    f"{max(total_times):.2f} ms"
)

print("=" * 60)