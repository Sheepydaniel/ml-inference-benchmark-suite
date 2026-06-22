from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

URL = "http://127.0.0.1:8000/predict_batch"


def make_reading(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    return {
        "soil_moisture": float(rng.uniform(5, 90)),
        "temperature": float(rng.uniform(10, 42)),
        "rainfall": float(rng.uniform(0, 80)),
        "humidity": float(rng.uniform(15, 95)),
        "sunlight": float(rng.uniform(1, 13)),
        "ndvi": float(rng.uniform(0.15, 0.95)),
    }


def percentiles(times):
    return {
        "mean_latency_ms": float(np.mean(times)),
        "p50_latency_ms": float(np.percentile(times, 50)),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "p99_latency_ms": float(np.percentile(times, 99)),
    }


def test_batch_size(size: int, runs: int = 10) -> dict:
    times = []
    total_time = 0.0

    for run in range(runs):
        items = [make_reading(size * 1000 + run * 100 + i) for i in range(size)]

        start = perf_counter()
        r = requests.post(URL, json={"items": items}, timeout=30)
        elapsed = perf_counter() - start

        r.raise_for_status()

        times.append(elapsed * 1000)
        total_time += elapsed

    row = {
        "mode": "fastapi_batch",
        "batch_size": size,
        "runs": runs,
        "n_predictions": size * runs,
        "throughput_predictions_per_second": float((size * runs) / total_time),
    }

    row.update(percentiles(times))
    return row


def main():
    rows = []

    for size in [1, 8, 32, 128, 512]:
        print(f"batch size {size}")
        rows.append(test_batch_size(size))

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / "batch_api_benchmark_results.csv"
    df.to_csv(out_file, index=False)

    print(df)
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
