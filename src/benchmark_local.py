from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "irrigation_model.joblib"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "soil_moisture",
    "temperature",
    "rainfall",
    "humidity",
    "sunlight",
    "ndvi",
]

def generate_inputs(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "soil_moisture": rng.uniform(5, 90, n),
        "temperature": rng.uniform(10, 42, n),
        "rainfall": rng.uniform(0, 80, n),
        "humidity": rng.uniform(15, 95, n),
        "sunlight": rng.uniform(1, 13, n),
        "ndvi": rng.uniform(0.15, 0.95, n),
    })[FEATURES]

def summarize_latencies(latencies_ms, total_time_s, n_predictions):
    return {
        "n_predictions": n_predictions,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_predictions_per_second": float(n_predictions / total_time_s),
    }

def benchmark_single_inference(model, n_requests: int = 1000):
    inputs = generate_inputs(n_requests)
    latencies = []

    total_start = perf_counter()

    for i in range(n_requests):
        row = inputs.iloc[[i]]
        start = perf_counter()
        model.predict_proba(row)
        end = perf_counter()
        latencies.append((end - start) * 1000)

    total_end = perf_counter()

    result = summarize_latencies(latencies, total_end - total_start, n_requests)
    result["mode"] = "local_single"
    result["batch_size"] = 1
    return result

def benchmark_batch_inference(model, batch_sizes=(1, 8, 32, 128, 512, 1000)):
    rows = []

    for batch_size in batch_sizes:
        inputs = generate_inputs(batch_size)

        start = perf_counter()
        model.predict_proba(inputs)
        end = perf_counter()

        total_time_s = end - start
        latency_ms = total_time_s * 1000

        rows.append({
            "mode": "local_batch",
            "batch_size": batch_size,
            "n_predictions": batch_size,
            "mean_latency_ms": latency_ms,
            "p50_latency_ms": latency_ms,
            "p95_latency_ms": latency_ms,
            "p99_latency_ms": latency_ms,
            "throughput_predictions_per_second": batch_size / total_time_s,
        })

    return rows

def main():
    model = joblib.load(MODEL_PATH)

    results = []
    results.append(benchmark_single_inference(model))
    results.extend(benchmark_batch_inference(model))

    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "local_benchmark_results.csv"
    df.to_csv(output_path, index=False)

    print(df)
    print(f"\nSaved local benchmark results to: {output_path}")

if __name__ == "__main__":
    main()
