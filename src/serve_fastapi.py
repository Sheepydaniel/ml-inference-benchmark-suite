from pathlib import Path
from time import perf_counter
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "irrigation_model.joblib"

FEATURES = [
    "soil_moisture",
    "temperature",
    "rainfall",
    "humidity",
    "sunlight",
    "ndvi",
]

app = FastAPI(
    title="ML Inference Benchmarking API",
    description="FastAPI model-serving endpoint for measuring ML inference latency and throughput.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

class IrrigationInput(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=-20, le=60)
    rainfall: float = Field(..., ge=0, le=200)
    humidity: float = Field(..., ge=0, le=100)
    sunlight: float = Field(..., ge=0, le=24)
    ndvi: float = Field(..., ge=0, le=1)

@app.get("/")
def root():
    return {
        "message": "ML Inference Benchmarking API is running.",
        "endpoint": "POST /predict",
    }

@app.post("/predict")
def predict(input_data: IrrigationInput):
    start = perf_counter()

    row = pd.DataFrame([input_data.model_dump()])[FEATURES]

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    end = perf_counter()
    latency_ms = (end - start) * 1000

    return {
        "prediction": prediction,
        "label": "Needs irrigation" if prediction == 1 else "No irrigation needed",
        "probability_needs_water": round(probability, 4),
        "latency_ms": round(latency_ms, 4),
    }
