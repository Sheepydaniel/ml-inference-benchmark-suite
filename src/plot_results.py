from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
FIGURES = BASE_DIR / "figures"
FIGURES.mkdir(exist_ok=True)


def bar_chart(labels, values, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    print(f"saved {FIGURES / filename}")


def line_chart(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    print(f"saved {FIGURES / filename}")


def plot_local_and_api():
    files = [
        RESULTS / "local_benchmark_results.csv",
        RESULTS / "api_benchmark_results.csv",
    ]

    frames = [pd.read_csv(f) for f in files if f.exists()]

    if not frames:
        print("no local/api benchmark results found")
        return

    df = pd.concat(frames, ignore_index=True)
    labels = df["mode"] + " b" + df["batch_size"].astype(str)

    bar_chart(
        labels,
        df["p95_latency_ms"],
        "p95 latency (ms)",
        "p95 latency by inference path",
        "p95_latency_comparison.png",
    )

    bar_chart(
        labels,
        df["throughput_predictions_per_second"],
        "predictions/sec",
        "throughput by inference path",
        "throughput_comparison.png",
    )


def plot_batch_api():
    f = RESULTS / "batch_api_benchmark_results.csv"

    if not f.exists():
        print("no batch api results found")
        return

    df = pd.read_csv(f)

    line_chart(
        df["batch_size"],
        df["p95_latency_ms"],
        "batch size",
        "p95 latency (ms)",
        "FastAPI batch latency",
        "batch_api_p95_latency.png",
    )

    line_chart(
        df["batch_size"],
        df["throughput_predictions_per_second"],
        "batch size",
        "predictions/sec",
        "FastAPI batch throughput",
        "batch_api_throughput.png",
    )


def plot_concurrency():
    f = RESULTS / "concurrent_api_benchmark_results.csv"

    if not f.exists():
        print("no concurrency results found")
        return

    df = pd.read_csv(f)

    line_chart(
        df["concurrency"],
        df["p95_latency_ms"],
        "concurrent requests",
        "p95 latency (ms)",
        "FastAPI latency under concurrency",
        "concurrent_api_p95_latency.png",
    )

    line_chart(
        df["concurrency"],
        df["throughput_requests_per_second"],
        "concurrent requests",
        "requests/sec",
        "FastAPI throughput under concurrency",
        "concurrent_api_throughput.png",
    )


def main():
    plot_local_and_api()
    plot_batch_api()
    plot_concurrency()


if __name__ == "__main__":
    main()
