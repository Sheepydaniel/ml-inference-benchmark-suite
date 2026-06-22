# ML Inference Benchmark Suite

A small benchmarking project for measuring inference latency, throughput, batching behavior, and API-serving overhead in machine learning systems.

## Overview

This project compares different ways of running inference for a trained machine learning model. The goal is to understand how deployment choices affect real system performance, not just model accuracy.

The current benchmark uses a smart-irrigation classification model and compares:

1. Local Python inference
2. Single-request FastAPI inference
3. Batch inference through FastAPI
4. Concurrent API requests

The benchmark records latency and throughput metrics, then generates plots for easier comparison.

## Why This Project

In many ML projects, the main focus is training a model and reporting accuracy. In practice, deployment introduces a different set of problems: response time, throughput, reproducibility, serving overhead, and reliability under repeated requests.

This project studies those engineering tradeoffs in a simple, reproducible setting.

## Model Task

The model predicts whether a crop needs irrigation based on environmental sensor-style features:

* Soil moisture
* Temperature
* Rainfall
* Humidity
* Sunlight
* NDVI

The classifier is trained on synthetic smart-irrigation data and saved as a reusable model artifact.

## Benchmarks

The project currently measures:

* Mean latency
* p50 latency
* p95 latency
* p99 latency
* Throughput in predictions per second
* Batch-size effects
* FastAPI serving overhead
* Concurrent request behavior

## Current Results

The benchmark outputs are saved in the `results/` directory as CSV files.

Current result files:

* `local_benchmark_results.csv`
* `api_benchmark_results.csv`
* `batch_api_benchmark_results.csv`
* `concurrent_api_benchmark_results.csv`
* `training_metrics.json`

Generated plots are saved in the `figures/` directory.

Current figure files:

* `p95_latency_comparison.png`
* `throughput_comparison.png`
* `batch_api_p95_latency.png`
* `batch_api_throughput.png`
* `concurrent_api_p95_latency.png`
* `concurrent_api_throughput.png`

The initial FastAPI single-request benchmark was run over 500 prediction requests and measured approximately:

* p95 latency: ~173.8 ms
* p99 latency: ~208.3 ms
* Throughput: ~8.5 predictions/second

Batch and concurrency benchmarks were then added to compare how the system behaves under larger request patterns.

## Repository Structure

```text
ml-inference-benchmark-suite/
  figures/
    batch_api_p95_latency.png
    batch_api_throughput.png
    concurrent_api_p95_latency.png
    concurrent_api_throughput.png
    p95_latency_comparison.png
    throughput_comparison.png

  models/
    irrigation_model.joblib

  results/
    api_benchmark_results.csv
    batch_api_benchmark_results.csv
    concurrent_api_benchmark_results.csv
    local_benchmark_results.csv
    synthetic_irrigation_data.csv
    training_metrics.json

  src/
    benchmark_api.py
    benchmark_batch_api.py
    benchmark_concurrent_api.py
    benchmark_local.py
    plot_results.py
    serve_fastapi.py
    train_model.py

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
python -m uvicorn src.serve_fastapi:app --reload
```

In a second terminal, run the single-request API benchmark:

```bash
python src/benchmark_api.py
```

Run the batch API benchmark:

```bash
python src/benchmark_batch_api.py
```

Run the concurrent request benchmark:

```bash
python src/benchmark_concurrent_api.py
```

Generate plots:

```bash
python src/plot_results.py
```

## API Endpoints

The FastAPI server exposes three main endpoints:

```text
GET /health
POST /predict
POST /predict_batch
```

Example single prediction request:

```json
{
  "soil_moisture": 35,
  "temperature": 29,
  "rainfall": 10,
  "humidity": 45,
  "sunlight": 8,
  "ndvi": 0.62
}
```

Example batch request:

```json
{
  "items": [
    {
      "soil_moisture": 35,
      "temperature": 29,
      "rainfall": 10,
      "humidity": 45,
      "sunlight": 8,
      "ndvi": 0.62
    },
    {
      "soil_moisture": 70,
      "temperature": 22,
      "rainfall": 40,
      "humidity": 65,
      "sunlight": 5,
      "ndvi": 0.78
    }
  ]
}
```

## What This Shows

This project is not meant to maximize model accuracy. It is meant to study the systems side of ML inference.

The main takeaways are:

* Local inference avoids API overhead and is useful as a baseline.
* API serving better represents a real deployment setting.
* Batch inference can improve throughput, but latency has to be interpreted differently.
* Concurrent requests reveal how the serving path behaves under repeated load.
* p95 and p99 latency are more useful than average latency when thinking about user-facing reliability.

## Completed Improvements

* Added FastAPI model-serving endpoint
* Added local inference benchmarking
* Added single-request API benchmarking
* Added batch inference endpoint
* Added batch-size benchmarking
* Added concurrent request benchmarking
* Generated latency and throughput plots for local, API, batch, and concurrent inference paths
* Saved benchmark results in reproducible CSV files

## Next Steps

Planned improvements:

* Add ONNX Runtime inference comparison
* Compare CPU and GPU inference where available
* Measure memory usage
* Add transformer-model inference benchmarks
* Add repeated warm-start vs cold-start tests
* Build a simple dashboard for benchmark visualization

## Tech Stack

* Python
* FastAPI
* scikit-learn
* pandas
* NumPy
* matplotlib
* requests
* httpx
* joblib

## License

This project is released under the MIT License.
