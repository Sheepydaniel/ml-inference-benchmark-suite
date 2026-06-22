import asyncio
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

URL = "http://127.0.0.1:8000/predict"


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


async def hit_api(client: httpx.AsyncClient, item: dict) -> dict:
    start = perf_counter()

    try:
        r = await client.post(URL, json=item, timeout=30)
        r.raise_for_status()
        ok = True
    except Exception:
        ok = False

    return {
        "ok": ok,
        "latency_ms": (perf_counter() - start) * 1000,
    }


async def run_level(concurrency: int, total: int = 200) -> dict:
    items = [make_reading(i) for i in range(total)]
    times = []
    ok_count = 0

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    start = perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:
        for i in range(0, total, concurrency):
            chunk = items[i:i + concurrency]
            jobs = [hit_api(client, item) for item in chunk]
            results = await asyncio.gather(*jobs)

            for result in results:
                if result["ok"]:
                    ok_count += 1
                    times.append(result["latency_ms"])

    elapsed = perf_counter() - start

    if not times:
        return {
            "mode": "fastapi_concurrent",
            "concurrency": concurrency,
            "n_requests": total,
            "successful_requests": 0,
            "error_rate": 1.0,
            "mean_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "throughput_requests_per_second": 0,
        }

    return {
        "mode": "fastapi_concurrent",
        "concurrency": concurrency,
        "n_requests": total,
        "successful_requests": ok_count,
        "error_rate": float((total - ok_count) / total),
        "mean_latency_ms": float(np.mean(times)),
        "p50_latency_ms": float(np.percentile(times, 50)),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "p99_latency_ms": float(np.percentile(times, 99)),
        "throughput_requests_per_second": float(ok_count / elapsed),
    }


async def main_async():
    rows = []

    for concurrency in [1, 5, 10, 25, 50]:
        print(f"testing concurrency={concurrency}")
        rows.append(await run_level(concurrency))

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / "concurrent_api_benchmark_results.csv"
    df.to_csv(out_file, index=False)

    print(df)
    print(f"\nsaved {out_file}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
