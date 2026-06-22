from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "soil_moisture",
    "temperature",
    "rainfall",
    "humidity",
    "sunlight",
    "ndvi",
]

def create_synthetic_irrigation_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    soil_moisture = rng.uniform(5, 90, n_samples)
    temperature = rng.uniform(10, 42, n_samples)
    rainfall = rng.uniform(0, 80, n_samples)
    humidity = rng.uniform(15, 95, n_samples)
    sunlight = rng.uniform(1, 13, n_samples)
    ndvi = rng.uniform(0.15, 0.95, n_samples)

    water_stress_score = (
        (100 - soil_moisture) * 0.35
        + temperature * 0.25
        + (100 - humidity) * 0.15
        + (80 - rainfall) * 0.15
        + sunlight * 0.07
        + (1 - ndvi) * 20
    )

    noise = rng.normal(0, 5, n_samples)
    needs_water = (water_stress_score + noise > np.median(water_stress_score)).astype(int)

    return pd.DataFrame({
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity,
        "sunlight": sunlight,
        "ndvi": ndvi,
        "needs_water": needs_water,
    })

def main():
    df = create_synthetic_irrigation_data()

    X = df[FEATURES]
    y = df["needs_water"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds).tolist()

    joblib.dump(model, MODELS_DIR / "irrigation_model.joblib")

    df.to_csv(RESULTS_DIR / "synthetic_irrigation_data.csv", index=False)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "features": FEATURES,
        "model_type": "RandomForestClassifier",
        "n_samples": len(df),
    }

    with open(RESULTS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved model to: {MODELS_DIR / 'irrigation_model.joblib'}")

if __name__ == "__main__":
    main()
