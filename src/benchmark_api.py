from pathlib import Path
from time import perf_counter
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_URL = "http://127.0.0.1:8000/predict"

def generate_payload(seed: int):
    rng = np.random.default_rng(seed)
    return {
        "soil_moisture": float(rng.uniform(5, 90)),
        "temperature": float(rng.uniform(10, 42)),
        "rainfall": float(rng.uniform(0, 80)),
        "humidity": float(rng.uniform(15, 95)),
        "sunlight": float(rng.uniform(1, 13)),
        "ndvi": float(rng.uniform(0.15, 0.95)),
    }

def main(n_requests: int = 500):
    latencies = []
    successful_requests = 0

    total_start = perf_counter()

    for i in range(n_requests):
        payload = generate_payload(i)

        start = perf_counter()
        response = requests.post(API_URL, json=payload, timeout=10)
        end = perf_counter()

        response.raise_for_status()

        latencies.append((end - start) * 1000)
        successful_requests += 1

    total_end = perf_counter()
    total_time_s = total_end - total_start

    result = {
        "mode": "fastapi_single",
        "batch_size": 1,
        "n_predictions": successful_requests,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_predictions_per_second": float(successful_requests / total_time_s),
    }

    df = pd.DataFrame([result])
    output_path = RESULTS_DIR / "api_benchmark_results.csv"
    df.to_csv(output_path, index=False)

    print(df)
    print(f"\nSaved API benchmark results to: {output_path}")

if __name__ == "__main__":
    main()
