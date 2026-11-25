"""Demonstrate LSTM predictions on real songs held out from all fitting."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "model_results"
MPL_CONFIG_DIR = RESULTS_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
TEMP_DIR = RESULTS_DIR / ".tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
tempfile.tempdir = str(TEMP_DIR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

try:
    from src.lstm_feature_extraction import (FEATURE_NAMES, extract_melody_sequence,
                                             scale_and_pad_sequence)
except ModuleNotFoundError:
    from lstm_feature_extraction import (FEATURE_NAMES, extract_melody_sequence,
                                         scale_and_pad_sequence)
try:
    from src.lstm_layers import MaskAwareLSTM  # Registers the saved custom LSTM layer.
except ModuleNotFoundError:
    from lstm_layers import MaskAwareLSTM  # noqa: F401

DATASET_PATH = ROOT / "data" / "training" / "melody_chord_dataset.csv"
MODEL_PATH = ROOT / "models" / "chord_lstm.keras"
METADATA_PATH = ROOT / "models" / "chord_lstm_metadata.json"
SCALER_PATH = ROOT / "models" / "chord_lstm_scaler.joblib"


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Saved LSTM feature order does not match the shared sequence extractor.")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    frame = pd.read_csv(DATASET_PATH)
    sequences = [extract_melody_sequence(row) for _, row in frame.iterrows()]
    groups = frame["song_id"].astype(str).reset_index(drop=True)
    row_indices = np.arange(len(frame))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_indices = next(splitter.split(row_indices, groups=groups))
    test_song_ids = set(groups.iloc[test_indices])
    if len(test_song_ids) != metadata["testing_song_count"]:
        raise ValueError("Recreated test-song split does not match saved metadata.")
    sequence_length = int(metadata["max_sequence_length"])
    selected_indices: list[int] = []
    selected_songs: set[str] = set()
    for index in test_indices:
        song_id = groups.iloc[index]
        if song_id not in selected_songs:
            selected_indices.append(int(index))
            selected_songs.add(song_id)
        if len(selected_indices) == 5:
            break
    if len(selected_indices) < 5:
        raise ValueError("Could not select five distinct held-out test songs.")

    batch = np.stack([scale_and_pad_sequence(sequences[index], scaler, sequence_length)
                      for index in selected_indices])
    probabilities = model.predict(batch, verbose=0)
    classes = metadata["chord_classes"]
    reverse_mapping = {int(value): key for key, value in metadata["chord_label_mapping"].items()}
    for batch_index, source_index in enumerate(selected_indices):
        row = frame.iloc[source_index]
        predicted = classes[int(np.argmax(probabilities[batch_index]))]
        print(f"Song ID: {row['song_id']}")
        print(f"Actual chord: {row['chord_label']}")
        print(f"Predicted chord: {predicted}")
        print("Prediction probabilities:")
        for class_index, class_name in enumerate(classes):
            print(f"{class_name}: {probabilities[batch_index, class_index]:.6f}")
        print()
        if reverse_mapping[int(metadata["chord_label_mapping"][row["chord_label"]])] != row["chord_label"]:
            raise ValueError("Saved chord mapping is inconsistent with the test dataset.")


if __name__ == "__main__":
    main()
