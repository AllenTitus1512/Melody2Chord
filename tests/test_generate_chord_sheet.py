"""End-to-end route orchestration tests with mocked upstream model outputs."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import app as flask_app

ROOT = Path(__file__).resolve().parents[1]


class GenerateChordSheetRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.app.test_client()
        self.melody = {"timeline": [{"start": 0, "end": 1, "note": "C4", "frequency": 261.63}]}
        self.words = [{"word": "Hello", "start": 0.0, "end": 0.8}]
        self.window = {"start": 0, "end": 1, "notes": ["C4", "E4"],
                       "frequencies": [261.63, 329.63], "durations": [0.5, 0.5]}
        self.alignment = {"lines": [{"words": self.words,
                                     "chords": [{"chord": "C", "word_index": 0}]}],
                          "warnings": []}

    def test_full_pipeline_runs_once_and_returns_aligned_sheet(self):
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=self.melody) as extract, \
             patch.object(flask_app, "detect_key", return_value={"key": "C Major"}) as detect, \
             patch.object(flask_app, "transcribe_lyrics", return_value={"words": self.words,
                                                                          "language": "en",
                                                                          "metadata": {"language_probability": 0.98}}) as transcribe, \
             patch.object(flask_app, "make_prediction_windows", return_value=[self.window]), \
             patch.object(flask_app, "predict_chords", return_value={"chord": "C", "probability": 0.8,
                                                                     "alternatives": []}) as predict, \
             patch.object(flask_app, "align_lyrics_and_chords", return_value=self.alignment) as align:
            response = self.client.post("/api/generate-chord-sheet", json={"upload_id": "recording.wav"})

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertTrue(result["success"])
        self.assertEqual(result["musical_key"], "C Major")
        self.assertEqual(result["alignment"], self.alignment)
        extract.assert_called_once()
        detect.assert_called_once_with(self.melody)
        transcribe.assert_called_once_with(ROOT / "app.py", language=None)
        predict.assert_called_once()
        align.assert_called_once()

    def test_transcription_failure_does_not_predict_or_align(self):
        from src.lyrics_transcription import LyricsTranscriptionError
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=self.melody), \
             patch.object(flask_app, "detect_key", return_value={"key": "C Major"}), \
             patch.object(flask_app, "transcribe_lyrics", side_effect=LyricsTranscriptionError("Transcription failed.")), \
             patch.object(flask_app, "predict_chords") as predict, \
             patch.object(flask_app, "align_lyrics_and_chords") as align:
            response = self.client.post("/api/generate-chord-sheet", json={"upload_id": "recording.wav"})

        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.get_json()["success"])
        predict.assert_not_called()
        align.assert_not_called()

    def test_current_upload_key_is_reused(self):
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=self.melody), \
             patch.object(flask_app, "detect_key") as detect, \
             patch.object(flask_app, "transcribe_lyrics", return_value={"words": self.words}), \
             patch.object(flask_app, "make_prediction_windows", return_value=[self.window]), \
             patch.object(flask_app, "predict_chords", return_value={"chord": "C", "probability": 0.8,
                                                                     "alternatives": []}), \
             patch.object(flask_app, "align_lyrics_and_chords", return_value=self.alignment):
            response = self.client.post("/api/generate-chord-sheet", json={
                "upload_id": "recording.wav", "musical_key": "G Major"})

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["musical_key"], "G Major")
        self.assertIsNone(response.get_json()["key_analysis"])
        detect.assert_not_called()

    def test_detected_language_confidence_and_romanized_words_reach_alignment(self):
        timed_word = {"word": "تم", "start": 0.2, "end": 0.5, "probability": 0.9}
        with patch.object(flask_app, "get_upload_path", return_value=ROOT / "app.py"), \
             patch.object(flask_app, "extract_melody", return_value=self.melody), \
             patch.object(flask_app, "detect_key", return_value={"key": "C Major"}), \
             patch.object(flask_app, "transcribe_lyrics", return_value={
                 "language": "ur", "metadata": {"language_probability": 0.42},
                 "words": [timed_word], "transcript": "تم",
             }), \
             patch.object(flask_app, "make_prediction_windows", return_value=[self.window]), \
             patch.object(flask_app, "predict_chords", return_value={"chord": "C", "probability": 0.8,
                                                                       "alternatives": []}), \
             patch.object(flask_app, "align_lyrics_and_chords", return_value=self.alignment) as align:
            response = self.client.post("/api/generate-chord-sheet", json={"upload_id": "recording.wav"})

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["detected_language"], "ur")
        self.assertEqual(result["language_probability"], 0.42)
        self.assertEqual(result["language_confidence"], "low")
        self.assertIsNotNone(result["language_warning"])
        aligned_words = align.call_args.args[0]
        self.assertEqual(aligned_words[0]["word"], "tum")
        self.assertEqual(aligned_words[0]["start"], 0.2)
        self.assertEqual(aligned_words[0]["end"], 0.5)

    def test_missing_upload_is_rejected(self):
        with patch.object(flask_app, "get_upload_path", return_value=None):
            response = self.client.post("/api/generate-chord-sheet", json={"upload_id": "missing.wav"})
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
