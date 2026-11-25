"""Compatibility and route tests using the saved model and real POP909 rows."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import chord_prediction
from src.lstm_feature_extraction import FEATURE_NAMES, extract_melody_sequence, scale_and_pad_sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training" / "melody_chord_dataset.csv"
METADATA = json.loads((ROOT / "models" / "chord_lstm_metadata.json").read_text(encoding="utf-8"))


class ChordPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(DATASET)
        cls.row = cls.frame.iloc[0]

    def setUp(self):
        chord_prediction.clear_model_cache()

    def test_saved_model_and_feature_shape_load(self):
        model, metadata, scaler = chord_prediction._load_assets()
        self.assertEqual(model.input_shape[1:], (METADATA["max_sequence_length"], len(FEATURE_NAMES)))
        self.assertEqual(metadata["feature_names"], list(FEATURE_NAMES))
        self.assertIsNotNone(scaler)

    def test_real_dataset_sequence_uses_training_layout_and_padding(self):
        sequence = extract_melody_sequence(self.row)
        self.assertEqual(sequence.shape[1], METADATA["feature_count"])
        model, metadata, scaler = chord_prediction._load_assets()
        padded = scale_and_pad_sequence(sequence, scaler, metadata["max_sequence_length"])
        self.assertEqual(padded.shape, tuple(model.input_shape[1:]))
        self.assertTrue(np.isfinite(padded).all())

    def test_valid_saved_model_prediction_has_trained_label_and_probabilities(self):
        result = chord_prediction.predict_chords(
            self.row["melody_notes"].split(","),
            [float(value) for value in self.row["note_frequencies"].split(",")],
            [float(value) for value in self.row["note_durations"].split(",")],
            self.row["musical_key"],
        )
        self.assertIn(result["chord"], METADATA["chord_classes"])
        self.assertLessEqual(len(result["alternatives"]), 3)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)

    def test_missing_model_is_reported_without_fabricated_prediction(self):
        with patch.object(chord_prediction, "MODEL_PATH", ROOT / "models" / "missing-test-model.keras"):
            with self.assertRaisesRegex(chord_prediction.ChordPredictionError, "unavailable"):
                chord_prediction._load_assets()

    def test_empty_melody_is_rejected(self):
        with self.assertRaisesRegex(chord_prediction.ChordPredictionError, "At least two"):
            chord_prediction.predict_chords([], [], [], "C Major")

    def test_unsupported_key_format_is_rejected(self):
        self.assertTrue(chord_prediction.is_supported_musical_key("G Major"))
        self.assertFalse(chord_prediction.is_supported_musical_key("H Major"))
        self.assertFalse(chord_prediction.is_supported_musical_key(""))

    def test_prediction_windows_use_positive_overlaps_and_skip_sparse_windows(self):
        events = [
            {"start": 0.0, "end": 0.9, "note": "C4", "frequency": 261.63},
            {"start": 0.9, "end": 1.8, "note": "E4", "frequency": 329.63},
            {"start": 1.8, "end": 2.7, "note": "G4", "frequency": 392.0},
            {"start": 2.7, "end": 3.6, "note": "A4", "frequency": 440.0},
        ]
        windows = chord_prediction.make_prediction_windows(events)
        self.assertEqual(len(windows), 2)
        np.testing.assert_allclose(windows[0]["durations"], [0.9, 0.9])
        np.testing.assert_allclose(windows[1]["durations"], [0.9, 0.9])

    def test_existing_routes_and_prediction_route_are_registered(self):
        import app as flask_app
        route_paths = {rule.rule for rule in flask_app.app.url_map.iter_rules()}
        self.assertTrue({"/", "/upload", "/api/extract-melody", "/api/detect-key", "/api/predict-chords"}.issubset(route_paths))
        # Exercise route flow with a real dataset melody/key and genuine saved-model inference.
        notes = self.row["melody_notes"].split(",")
        frequencies = [float(value) for value in self.row["note_frequencies"].split(",")]
        durations = [float(value) for value in self.row["note_durations"].split(",")]
        timeline, cursor = [], 0.0
        for note, frequency, duration in zip(notes, frequencies, durations):
            timeline.append({"start": cursor, "end": cursor + duration, "duration": duration,
                             "note": note, "frequency": frequency})
            cursor += duration
        melody = {"notes": notes, "note_count": len(notes), "melody_duration": cursor, "timeline": timeline}
        test_client = flask_app.app.test_client()
        self.assertEqual(test_client.get("/").status_code, 200)
        self.assertEqual(test_client.post("/upload", data={}).status_code, 400)
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=melody), \
             patch.object(flask_app, "detect_key", return_value={"key": self.row["musical_key"], "confidence": 50.0, "scale_notes": []}):
            response = test_client.post("/api/predict-chords", json={
                "upload_id": "real-row-test.wav", "musical_key": self.row["musical_key"]})
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["predictions"])
        self.assertEqual(payload["musical_key"], self.row["musical_key"])
        self.assertTrue(all(p["chord"] in METADATA["chord_classes"] for p in payload["predictions"]))
        # Existing endpoints remain callable and retain their existing response contracts.
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=melody):
            self.assertIn("melody", test_client.post("/api/extract-melody", json={"upload_id": "x.wav"}).get_json())
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=melody), \
             patch.object(flask_app, "detect_key", return_value={"key": "C Major", "confidence": 55, "scale_notes": []}):
            self.assertIn("key_analysis", test_client.post("/api/detect-key", json={"upload_id": "x.wav"}).get_json())


if __name__ == "__main__":
    unittest.main()
