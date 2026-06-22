from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "irrigation_model.joblib"

FEATURES = [
    "soil_moisture",
    "temperature",
    "rainfall",
    "humidity",
    "sunlight",
    "ndvi",
]

app = FastAPI(title="Irrigation Model API", version="1.1")

model = joblib.load(MODEL_FILE)


class SensorReading(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=-20, le=60)
    rainfall: float = Field(..., ge=0, le=200)
    humidity: float = Field(..., ge=0, le=100)
    sunlight: float = Field(..., ge=0, le=24)
    ndvi: float = Field(..., ge=0, le=1)


class SensorBatch(BaseModel):
    items: list[SensorReading]


def as_row(reading: SensorReading) -> dict:
    # Pydantic v2 uses model_dump; v1 uses dict.
    return reading.model_dump() if hasattr(reading, "model_dump") else reading.dict()


@app.get("/")
def home():
    return {
        "status": "running",
        "single": "/predict",
        "batch": "/predict_batch",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(reading: SensorReading):
    start = perf_counter()

    x = pd.DataFrame([as_row(reading)])[FEATURES]
    pred = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][1])

    ms = (perf_counter() - start) * 1000

    return {
        "prediction": pred,
        "label": "Needs irrigation" if pred else "No irrigation needed",
        "probability": round(prob, 4),
        "latency_ms": round(ms, 4),
    }


@app.post("/predict_batch")
def predict_batch(batch: SensorBatch):
    start = perf_counter()

    x = pd.DataFrame([as_row(item) for item in batch.items])[FEATURES]
    preds = model.predict(x)
    probs = model.predict_proba(x)[:, 1]

    ms = (perf_counter() - start) * 1000

    rows = []
    for pred, prob in zip(preds, probs):
        pred = int(pred)
        rows.append({
            "prediction": pred,
            "label": "Needs irrigation" if pred else "No irrigation needed",
            "probability": round(float(prob), 4),
        })

    return {
        "count": len(rows),
        "latency_ms": round(ms, 4),
        "predictions_per_second": round(len(rows) / (ms / 1000), 4),
        "results": rows,
    }
