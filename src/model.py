"""Random Forest model helpers for melody-to-chord learning."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "random_forest_baseline.joblib"
RANDOM_STATE = 42


def train_model(X, y, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """Fit a reproducible baseline classifier on prepared numerical features."""
    model = RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict_chord(model: RandomForestClassifier, features) -> str:
    """Predict one suitable chord label from one prepared feature row."""
    row = np.asarray(features)
    if row.ndim == 1:
        row = row.reshape(1, -1)
    return str(model.predict(row)[0])


def save_model(model: RandomForestClassifier, model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained model using joblib."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    """Load a previously trained baseline model."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run python -m src.train_model first.")
    return joblib.load(path)
