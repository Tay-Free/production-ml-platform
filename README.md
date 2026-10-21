# Production ML Platform

A machine learning deployment project that trains a model, serves predictions through a REST API, packages the application with Docker, and deploys it to Kubernetes.

## Features

- **Model training:** Scikit-learn pipeline using StandardScaler and Logistic Regression.
- **REST API:** FastAPI endpoints for health checks and model predictions.
- **Input validation:** Rejects invalid feature counts and non-finite values.
- **Automated testing:** Pytest tests for API functionality and invalid inputs.
- **Containerization:** Docker image with a health check.
- **Kubernetes deployment:** Deployment, Service, readiness probes, and liveness probes.

## Tech Stack

Python, scikit-learn, FastAPI, Pydantic, pytest, Docker, Kubernetes, and Git.

## Project Structure

```text
app/
  api/main.py
  training/train.py
artifacts/
  model.joblib
tests/
  test_api.py
k8s/
  deployment.yaml
  service.yaml
Dockerfile
requirements.txt
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m app.training.train
```

Start the API:

```bash
python -m uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

## Run Tests

```bash
python -m pytest -v
```

## Run with Docker

```bash
docker build -t production-ml-platform .
docker run --rm -p 8000:8000 production-ml-platform
```

## Deploy with Kubernetes

Ensure Docker Desktop Kubernetes is enabled and the image is available to the cluster. Apply the manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

Access the API through port forwarding:

```bash
kubectl port-forward service/ml-platform-service 8000:8000
```

Then visit `http://127.0.0.1:8000/docs`.

## Model Evaluation

The model achieved **98.25% accuracy** on the held-out test set for the Breast Cancer Wisconsin dataset. This is an educational demonstration, not a medical diagnostic tool.

## Future Improvements

- Add Prometheus metrics and monitoring dashboards.
- Add model versioning and experiment tracking.
- Implement continuous integration and automated deployment.
