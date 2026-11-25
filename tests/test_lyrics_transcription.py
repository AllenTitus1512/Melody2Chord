"""Unit and Flask-route tests for lyrics transcription.

The unit tests use mocked faster-whisper outputs; they validate adapter behavior,
not recognition accuracy. Real model transcription is verified separately when run.
"""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import lyrics_transcription as lyrics

ROOT = Path(__file__).resolve().parents[1]


class FakeWhisperModel:
    def __init__(self, segments=None, info=None, error=None):
        self.segments = segments or []
        self.info = info or SimpleNamespace(language="en", language_probability=0.98, duration=2.0)
        self.error = error
        self.options = None

    def transcribe(self, audio, **kwargs):
        self.options = kwargs
        if self.error:
            raise self.error
        return iter(self.segments), self.info


class LyricsTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = ROOT / "data" / "model_results" / ".tmp" / f"lyrics-test-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=False)
        self.audio_path = self.temp_dir / "sample.wav"
        self.audio_path.write_bytes(b"test audio bytes")

    def tearDown(self):
        for path in self.temp_dir.iterdir():
            path.unlink()
        self.temp_dir.rmdir()
        lyrics.clear_model_cache()

    @staticmethod
    def _segment(text, words):
        return SimpleNamespace(text=text, words=words)

    def test_mocked_transcription_structure_and_word_timestamps(self):
        model = FakeWhisperModel(segments=[self._segment(" We sing again", [
            SimpleNamespace(word=" We", start=0.1, end=0.4, probability=0.96),
            SimpleNamespace(word=" sing", start=0.4, end=0.9, probability=0.91),
            SimpleNamespace(word=" again", start=0.9, end=1.4, probability=None),
        ])])
        with patch.object(lyrics, "_load_model", return_value=model):
            result = lyrics.transcribe_lyrics(self.audio_path, language="en")
        self.assertEqual(result["transcript"], "We sing again")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["words"][0], {"word": "We", "start": 0.1, "end": 0.4, "probability": 0.96})
        self.assertEqual(result["words"][2], {"word": "again", "start": 0.9, "end": 1.4})
        self.assertTrue(model.options["word_timestamps"])
        self.assertFalse(model.options["vad_filter"])
        self.assertEqual(model.options["language"], "en")

    def test_automatic_detection_uses_whisper_language_and_probability(self):
        model = FakeWhisperModel(
            segments=[self._segment(" नमस्ते", [SimpleNamespace(word=" नमस्ते", start=0.1, end=0.8)])],
            info=SimpleNamespace(language="hi", language_probability=0.41, duration=1.0),
        )
        with patch.object(lyrics, "_load_model", return_value=model):
            result = lyrics.transcribe_lyrics(self.audio_path)
        self.assertIsNone(model.options["language"])
        self.assertEqual(result["language"], "hi")
        self.assertEqual(result["metadata"]["language_probability"], 0.41)

    def test_missing_and_invalid_word_timestamps_are_skipped(self):
        model = FakeWhisperModel(segments=[self._segment(" We sing", [
            SimpleNamespace(word=" We", start=0.1, end=0.4),
            SimpleNamespace(word=" sing", start=float("nan"), end=0.8),
            SimpleNamespace(word="", start=0.8, end=1.0),
            SimpleNamespace(word=" unknown", start=None, end=1.1),
        ])])
        with patch.object(lyrics, "_load_model", return_value=model):
            result = lyrics.transcribe_lyrics(self.audio_path)
        self.assertEqual([item["word"] for item in result["words"]], ["We"])
        self.assertEqual(result["metadata"]["skipped_word_timestamps"], 3)

    def test_empty_transcription_is_a_clear_error(self):
        model = FakeWhisperModel(segments=[self._segment(" ", [])])
        with patch.object(lyrics, "_load_model", return_value=model):
            with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "No lyrics were recognized"):
                lyrics.transcribe_lyrics(self.audio_path)

    def test_transcription_errors_are_wrapped_safely(self):
        model = FakeWhisperModel(error=RuntimeError("private backend detail"))
        with patch.object(lyrics, "_load_model", return_value=model):
            with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "clear vocal recording") as raised:
                lyrics.transcribe_lyrics(self.audio_path)
        self.assertNotIn("private backend detail", str(raised.exception))

    def test_missing_empty_and_unsupported_audio_are_rejected(self):
        with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "not found"):
            lyrics.transcribe_lyrics(self.temp_dir / "missing.wav")
        empty = self.temp_dir / "empty.wav"
        empty.touch()
        with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "empty"):
            lyrics.transcribe_lyrics(empty)
        unsupported = self.temp_dir / "sample.ogg"
        unsupported.write_bytes(b"data")
        with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "Only WAV and MP3"):
            lyrics.transcribe_lyrics(unsupported)

    def test_model_initialization_failure_has_download_guidance(self):
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("private download error")):
            with self.assertLogs("src.lyrics_transcription", level="ERROR") as logs:
                with self.assertRaisesRegex(lyrics.LyricsTranscriptionError, "internet connection") as raised:
                    lyrics._load_model("base")
        self.assertNotIn("private download error", str(raised.exception))
        log_output = "\n".join(logs.output)
        self.assertIn("RuntimeError: private download error", log_output)
        self.assertIn("Traceback", log_output)
        self.assertIn("Whisper cache diagnostics", log_output)

    def test_lyrics_endpoint_reuses_existing_upload_and_returns_consistent_json(self):
        import app as flask_app
        client = flask_app.app.test_client()
        self.assertEqual(client.post("/api/extract-lyrics", json={}).status_code, 404)
        expected = {"language": "en", "transcript": "We sing", "words": [
            {"word": "We", "start": 0.1, "end": 0.4}], "metadata": {"model_size": "base"},
            "message": "Lyrics transcribed successfully."}
        with patch.object(flask_app, "get_upload_path", return_value=self.audio_path), \
             patch.object(flask_app, "transcribe_lyrics", return_value=expected) as transcribe:
            response = client.post("/api/extract-lyrics", json={"upload_id": "existing-audio.wav"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["words"], expected["words"])
        transcribe.assert_called_once_with(self.audio_path, language=None)

    def test_endpoint_rejects_empty_stored_audio(self):
        import app as flask_app
        empty = self.temp_dir / "empty-upload.wav"
        empty.touch()
        with patch.object(flask_app, "get_upload_path", return_value=empty):
            response = flask_app.app.test_client().post("/api/extract-lyrics", json={"upload_id": "empty.wav"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.get_json()["error"])

    def test_endpoint_returns_safe_error_for_invalid_audio_content(self):
        import app as flask_app
        fake_model = FakeWhisperModel(error=RuntimeError("private decoder detail"))
        client = flask_app.app.test_client()
        with patch.object(flask_app, "get_upload_path", return_value=self.audio_path), \
             patch.object(lyrics, "_load_model", return_value=fake_model):
            response = client.post("/api/extract-lyrics", json={"upload_id": "invalid.wav"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private decoder detail", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
