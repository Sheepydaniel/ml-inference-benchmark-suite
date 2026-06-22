# ML Inference Benchmark Suite

A small benchmarking project for measuring inference latency, throughput, batching behavior, and API-serving overhead in machine learning systems.

## Overview

This project compares different ways of running inference for a trained machine learning model. The goal is to understand how deployment choices affect real system performance, not just model accuracy.

The current benchmark uses a smart-irrigation classification model and compares:

1. Local Python inference
2. FastAPI-based model serving
3. Batch inference across multiple input sizes

The benchmark records latency and throughput metrics, then generates figures for easier comparison.

## Why This Project

In many ML projects, the focus is on training a model and reporting accuracy. In practice, deployment introduces a different set of problems: response time, throughput, reproducibility, serving overhead, and reliability under repeated requests.

This project is meant to study those engineering tradeoffs in a simple, reproducible setting.

## Model Task

The model predicts whether a crop needs irrigation based on environmental features:

* Soil moisture
* Temperature
* Rainfall
* Humidity
* Sunlight
* NDVI

The classifier is trained on synthetic sensor-style data and saved as a reusable model artifact.

## Benchmarks

The project currently measures:

* Mean latency
* p50 latency
* p95 latency
* p99 latency
* Throughput in predictions per second
* Batch-size effects
* FastAPI serving overhead

The API benchmark was run over 500 single-prediction requests.

Current FastAPI result:

* p95 latency: ~173.8 ms
* p99 latency: ~208.3 ms
* Throughput: ~8.5 predictions/second

## Repository Structure

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

## Running the Project

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

## Results

Benchmark outputs are saved in the `results/` directory as CSV files. Generated figures are saved in the `figures/` directory.

The current results show the expected tradeoff between local inference and API-based serving. Local inference avoids request overhead, while FastAPI serving better represents a real deployment setting where predictions are exposed through an endpoint.

Batch inference also changes the performance profile. Larger batches can improve throughput, but latency has to be interpreted differently because each request may represent multiple predictions.

## Next Steps

Planned improvements:

* Add concurrent request benchmarking
* Add ONNX Runtime inference comparison
* Compare CPU and GPU inference where available
* Measure memory usage
* Add transformer-model inference benchmarks
* Add batch inference through the FastAPI endpoint
* Build a simple dashboard for benchmark visualization
