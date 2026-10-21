
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "features": [
                17.99, 10.38, 122.8, 1001, 0.1184,
                0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                1.095, 0.9053, 8.589, 153.4, 0.006399,
                0.04904, 0.05373, 0.05373, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019, 0.1622,
                0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["malignant", "benign"]
    assert 0 <= data["confidence"] <= 1
    assert set(data["probabilities"].keys()) == {"malignant", "benign"}


def test_prediction_rejects_wrong_number_of_features():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 422



def test_prediction_rejects_infinite_values():
    features = [1.0] * 30
    features[0] = float("inf")

    import json

    payload = json.dumps(
        {"features": features},
        allow_nan=True,
    )

    response = client.post(
        "/predict",
        content=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422

