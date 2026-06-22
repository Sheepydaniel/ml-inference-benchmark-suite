from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_results():
    frames = []

    local_path = RESULTS_DIR / "local_benchmark_results.csv"
    api_path = RESULTS_DIR / "api_benchmark_results.csv"

    if local_path.exists():
        frames.append(pd.read_csv(local_path))

    if api_path.exists():
        frames.append(pd.read_csv(api_path))

    if not frames:
        raise FileNotFoundError("No benchmark result files found. Run benchmark scripts first.")

    return pd.concat(frames, ignore_index=True)

def plot_latency(df):
    labels = df["mode"] + "_batch_" + df["batch_size"].astype(str)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, df["p95_latency_ms"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("p95 Latency (ms)")
    plt.title("ML Inference p95 Latency by Serving Mode")
    plt.tight_layout()

    output = FIGURES_DIR / "p95_latency_comparison.png"
    plt.savefig(output, dpi=200)
    print(f"Saved: {output}")

def plot_throughput(df):
    labels = df["mode"] + "_batch_" + df["batch_size"].astype(str)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, df["throughput_predictions_per_second"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Predictions per Second")
    plt.title("ML Inference Throughput by Serving Mode")
    plt.tight_layout()

    output = FIGURES_DIR / "throughput_comparison.png"
    plt.savefig(output, dpi=200)
    print(f"Saved: {output}")

def main():
    df = load_results()
    plot_latency(df)
    plot_throughput(df)

if __name__ == "__main__":
    main()
