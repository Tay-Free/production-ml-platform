
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Production ML Platform API")

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run `python -m app.training.train` first."
    )

saved_artifact = joblib.load(MODEL_PATH)
model = saved_artifact["model"]
feature_names = saved_artifact["feature_names"]
target_names = saved_artifact["target_names"]


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="The 30 numeric features required by the model",
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not np.isfinite(request.features).all():
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    try:
        prediction = int(model.predict([request.features])[0])
        probabilities = model.predict_proba([request.features])[0]

        return {
            "prediction": target_names[prediction],
            "confidence": round(float(probabilities[prediction]), 4),
            "probabilities": {
                target_names[i]: round(float(probabilities[i]), 4)
                for i in range(len(target_names))
            },
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
