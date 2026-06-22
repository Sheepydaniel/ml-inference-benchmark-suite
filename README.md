# ML Inference Benchmark Suite

A reproducible benchmarking project for measuring machine learning inference latency, throughput, batching behavior, and deployment overhead across local Python inference and FastAPI model serving.

## Motivation

Training a machine learning model is only one part of building a useful ML system. Once a model is deployed, the system also needs to be fast, reliable, reproducible, and easy to evaluate. This project explores the systems side of ML by benchmarking how different inference setups affect real-world performance.

The project compares local model inference against API-based serving and measures practical metrics such as p95 latency, p99 latency, throughput, and batching performance.

## System Overview

This project uses a synthetic smart-irrigation classification task based on environmental sensor-style features:

* Soil moisture
* Temperature
* Rainfall
* Humidity
* Sunlight
* NDVI

A Random Forest classifier predicts whether irrigation is needed. The model is then served through two inference paths:

1. Local Python inference
2. FastAPI model-serving endpoint

## Metrics Measured

The benchmark scripts measure:

* Mean latency
* p50 latency
* p95 latency
* p99 latency
* Throughput in predictions per second
* Batch-size effects on inference speed
* Deployment overhead from API-based serving

## Current Results

The FastAPI benchmark was run over 500 single-prediction requests.

Current API benchmark result:

* p95 latency: approximately 173.8 ms
* p99 latency: approximately 208.3 ms
* throughput: approximately 8.5 predictions per second

The local benchmark also compares single prediction against batch prediction across different batch sizes.

## Project Structure

```text
ml-inference-benchmark-suite/
  figures/
    p95_latency_comparison.png
    throughput_comparison.png
  models/
    irrigation_model.joblib
  results/
    api_benchmark_results.csv
    local_benchmark_results.csv
    synthetic_irrigation_data.csv
    training_metrics.json
  src/
    train_model.py
    serve_fastapi.py
    benchmark_local.py
    benchmark_api.py
    plot_results.py
  requirements.txt
  README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_model.py
```

Run local inference benchmarks:

```bash
python src/benchmark_local.py
```

Start the FastAPI server:

```bash
uvicorn src.serve_fastapi:app --reload
```

In a second terminal, run the API benchmark:

```bash
python src/benchmark_api.py
```

Generate plots:

```bash
python src/plot_results.py
```

## What I Learned

This project helped me understand that ML deployment is not only about model accuracy. A useful ML system also depends on inference speed, serving overhead, reproducibility, and clear performance measurement.

The biggest takeaway is that different serving paths introduce different tradeoffs. Local inference is simpler and often faster, while API serving introduces overhead but better represents real deployment conditions. Batching can improve throughput, but it also changes how latency should be interpreted.

## Next Improvements

Planned improvements include:

* Adding ONNX Runtime benchmarking
* Comparing CPU vs GPU inference
* Adding concurrent request testing
* Measuring memory usage
* Adding transformer model inference benchmarks
* Testing batch inference through the API
* Creating a dashboard for visualizing benchmark results
