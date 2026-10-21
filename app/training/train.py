
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    # Load the sample dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build a preprocessing and classification pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )

    # Train and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
        )
    )

    # Save the trained model and its metadata
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": dataset.feature_names.tolist(),
            "target_names": dataset.target_names.tolist(),
        },
        output_dir / "model.joblib",
    )

    print(f"Model saved to {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    train_model()
