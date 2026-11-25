"""Upload route tests; valid-file metadata is mocked unless stated otherwise."""
from __future__ import annotations

import io
import re
import struct
import unittest
import wave
import uuid
from pathlib import Path
from unittest.mock import patch

import app as flask_app


def make_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(struct.pack("<h", 0) * 8000)
    return buffer.getvalue()


class AudioUploadTests(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.app.test_client()
        self.token = uuid.uuid4().hex
        self.created_uploads = []
        self.addCleanup(self.cleanup_created_uploads)

    def cleanup_created_uploads(self):
        for upload_id in self.created_uploads:
            path = flask_app.get_upload_path(upload_id)
            if path:
                path.unlink(missing_ok=True)

    def test_supported_audio_upload_is_saved_and_referenced(self):
        with patch.object(flask_app, "get_audio_info", return_value={"duration": 1.0, "sampling_rate": 8000}):
            response = self.client.post("/upload", data={
                "audio": (io.BytesIO(make_wav()), "voice.wav")
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()["audio"]
        self.created_uploads.append(data["upload_id"])
        self.assertEqual(data["filename"], "voice.wav")
        stored_path = flask_app.get_upload_path(data["upload_id"])
        self.assertIsNotNone(stored_path)
        self.assertTrue(stored_path.is_file())
        melody = {"notes": ["C4"], "note_count": 1, "melody_duration": 1.0, "timeline": []}
        with patch.object(flask_app, "extract_melody", return_value=melody) as extract:
            downstream = self.client.post("/api/extract-melody", json={"upload_id": data["upload_id"]})
        self.assertEqual(downstream.status_code, 200)
        extract.assert_called_once_with(stored_path)

    def test_valid_wav_is_uploaded_and_decoded_without_metadata_mock(self):
        response = self.client.post("/upload", data={
            "audio": (io.BytesIO(make_wav()), f"decoded-{self.token}.wav")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()["audio"]
        self.created_uploads.append(data["upload_id"])
        self.assertAlmostEqual(data["duration"], 1.0, places=2)
        self.assertGreater(data["sampling_rate"], 0)

    def test_missing_file_is_rejected(self):
        response = self.client.post("/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("No audio file", response.get_json()["error"])

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post("/upload", data={
            "audio": (io.BytesIO(b"audio"), "voice.flac")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only WAV and MP3", response.get_json()["error"])

    def test_empty_upload_is_rejected_and_removed(self):
        response = self.client.post("/upload", data={
            "audio": (io.BytesIO(b""), f"empty-{self.token}.wav")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.get_json()["error"])
        self.assertEqual(list(Path(flask_app.UPLOAD_FOLDER).glob(f"*empty-{self.token}.wav")), [])

    def test_malformed_audio_is_rejected_and_removed(self):
        response = self.client.post("/upload", data={
            "audio": (io.BytesIO(b"not an audio stream"), f"broken-{self.token}.wav")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unable to read", response.get_json()["error"])
        self.assertEqual(list(Path(flask_app.UPLOAD_FOLDER).glob(f"*broken-{self.token}.wav")), [])

    def test_audio_metadata_failure_returns_safe_error_and_removes_upload(self):
        with patch.object(flask_app, "get_audio_info", side_effect=RuntimeError("private storage detail")):
            response = self.client.post("/upload", data={
                "audio": (io.BytesIO(make_wav()), f"voice-{self.token}.wav")
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "The audio file could not be processed.")
        self.assertNotIn("private storage detail", response.get_data(as_text=True))
        self.assertEqual(list(Path(flask_app.UPLOAD_FOLDER).glob(f"*voice-{self.token}.wav")), [])

    def test_upload_save_failure_returns_json_error(self):
        missing_parent = Path(flask_app.UPLOAD_FOLDER) / ".upload-not-a-directory"
        missing_parent.write_text("file, not directory", encoding="utf-8")
        try:
            with patch.object(flask_app, "UPLOAD_FOLDER", missing_parent):
                # Supply the real request file while forcing its destination directory absent.
                response = self.client.post("/upload", data={
                    "audio": (io.BytesIO(make_wav()), "voice.wav")
                }, content_type="multipart/form-data")
        finally:
            missing_parent.unlink(missing_ok=True)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "The audio upload could not be saved. Please try again.")

    def test_rendered_upload_controls_and_single_submit_handler_are_connected(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        script = (Path(__file__).resolve().parents[1] / "static" / "js" / "script.js").read_text(encoding="utf-8")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="audio-file" name="audio" type="file"', html)
        self.assertIn('for="audio-file"', html)
        self.assertIn('id="analyze-button" type="submit"', html)
        self.assertIn("script.js?v=upload-debug-1", html)
        self.assertEqual(script.count('fileInput.addEventListener("change"'), 1)
        self.assertEqual(script.count('form.addEventListener("submit"'), 1)
        self.assertIn('formData.append("audio", file)', script)
        self.assertIn('fetch("/upload"', script)
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        script_ids = set(re.findall(r'getElementById\("([^"]+)"\)', script))
        self.assertEqual(script_ids - html_ids, set())
        self.assertIn('const chordPanel = document.getElementById("chord-panel")', script)


if __name__ == "__main__":
    unittest.main()
