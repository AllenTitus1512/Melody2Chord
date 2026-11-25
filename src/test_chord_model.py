"""Run offline predictions on real held-out POP909 song rows."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from src.feature_extraction import FeatureExtractionError, V2_TRAINING_FEATURE_NAMES, extract_dataset_features_v2
except ModuleNotFoundError:
    from feature_extraction import FeatureExtractionError, V2_TRAINING_FEATURE_NAMES, extract_dataset_features_v2

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "training" / "melody_chord_dataset.csv"
MODEL_PATH = ROOT / "models" / "chord_random_forest_v2.joblib"
METADATA_PATH = ROOT / "models" / "chord_model_v2_metadata.json"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    # The offline smoke test needs no parallel workers; this avoids platform
    # process/thread restrictions without changing the saved estimator.
    model.n_jobs = 1
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["feature_names"] != list(V2_TRAINING_FEATURE_NAMES):
        raise ValueError("Saved model feature order does not match the shared feature extractor.")
    frame = pd.read_csv(DATASET_PATH)
    kept_rows = []
    vectors = []
    for index, row in frame.iterrows():
        try:
            vectors.append(extract_dataset_features_v2(row))
            kept_rows.append(index)
        except FeatureExtractionError:
            continue
    usable = frame.loc[kept_rows].reset_index(drop=True)
    features = np.vstack(vectors)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_indices = next(splitter.split(features, usable["chord_label"], groups=usable["song_id"].astype(str)))
    for index in test_indices[:5]:
        row = usable.iloc[index]
        vector = pd.DataFrame([features[index]], columns=V2_TRAINING_FEATURE_NAMES)
        predicted = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        print(f"Song ID: {row['song_id']}")
        print(f"Actual chord: {row['chord_label']}")
        print(f"Predicted chord: {predicted}")
        print("Prediction probabilities:")
        print(dict(sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)))
        print()


if __name__ == "__main__":
    main()
