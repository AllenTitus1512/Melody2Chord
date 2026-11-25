"""Flask application entry point for Melody2Chord."""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from src.audio_processing import AudioProcessingError, extract_melody, get_audio_info
from src.key_detection import KeyDetectionError, detect_key
from src.chord_prediction import (ChordPredictionError, is_supported_musical_key,
                                  make_prediction_windows, predict_chords)
from src.lyrics_transcription import LyricsTranscriptionError, transcribe_lyrics
from src.lyrics_romanization import romanize_transcription
from src.lyric_chord_alignment import LyricChordAlignmentError, align_lyrics_and_chords

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

def allowed_file(filename: str) -> bool:
    """Return whether the filename has a permitted extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_path(upload_id: str) -> Path | None:
    """Resolve an upload identifier safely to a file inside the upload directory."""
    safe_name = secure_filename(upload_id)
    candidate = (UPLOAD_FOLDER / safe_name).resolve()
    return candidate if candidate.parent == UPLOAD_FOLDER.resolve() and candidate.is_file() else None

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    """Save a permitted upload and return basic audio metadata."""
    app.logger.info("Received POST /upload (content_length=%s)", request.content_length)
    if "audio" not in request.files:
        app.logger.warning("Upload request did not contain multipart field 'audio'")
        return jsonify(error="No audio file was included in the request."), 400
    audio_file = request.files["audio"]
    if not audio_file.filename:
        app.logger.warning("Upload request contained an empty filename")
        return jsonify(error="Please choose an audio file first."), 400
    safe_original_name = secure_filename(audio_file.filename)
    app.logger.info("Upload file received: filename=%s", safe_original_name or "(invalid)")
    if not allowed_file(audio_file.filename):
        app.logger.warning("Rejected upload with unsupported extension: filename=%s", safe_original_name)
        return jsonify(error="Only WAV and MP3 audio files are supported."), 400

    original_name = safe_original_name
    upload_id = f"{uuid.uuid4().hex}_{original_name}"
    file_path = UPLOAD_FOLDER / upload_id
    try:
        audio_file.save(file_path)
    except Exception:
        file_path.unlink(missing_ok=True)
        app.logger.exception("Could not save uploaded audio")
        return jsonify(error="The audio upload could not be saved. Please try again."), 500
    try:
        audio_info = get_audio_info(file_path)
    except AudioProcessingError as error:
        file_path.unlink(missing_ok=True)
        app.logger.exception("Uploaded audio could not be decoded or processed")
        return jsonify(error=str(error)), 400
    except Exception:
        file_path.unlink(missing_ok=True)
        app.logger.exception("Unexpected audio-processing error")
        return jsonify(error="The audio file could not be processed."), 500

    app.logger.info("Upload processed successfully: filename=%s bytes=%s", original_name, file_path.stat().st_size)
    return jsonify(message="Audio uploaded and analyzed successfully.", audio={
        "filename": original_name, "duration": audio_info["duration"],
        "sampling_rate": audio_info["sampling_rate"], "upload_id": upload_id,
    })

@app.route("/api/extract-melody", methods=["POST"])
def extract_uploaded_melody():
    """Extract melody data for an audio file that was already uploaded."""
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id", "")
    audio_path = get_upload_path(upload_id) if isinstance(upload_id, str) else None
    if audio_path is None:
        return jsonify(error="The uploaded audio file was not found. Please upload it again."), 404
    try:
        return jsonify(melody=extract_melody(audio_path))
    except AudioProcessingError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Unexpected melody-extraction error")
        return jsonify(error="Melody extraction could not be completed for this recording."), 500

@app.route("/api/detect-key", methods=["POST"])
def detect_uploaded_key():
    """Reuse melody extraction, then determine its most likely musical key."""
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id", "")
    audio_path = get_upload_path(upload_id) if isinstance(upload_id, str) else None
    if audio_path is None:
        return jsonify(error="The uploaded audio file was not found. Please upload it again."), 404
    try:
        melody = extract_melody(audio_path)
        return jsonify(key_analysis=detect_key(melody))
    except (AudioProcessingError, KeyDetectionError) as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Unexpected key-detection error")
        return jsonify(error="Key detection could not be completed for this recording."), 500

@app.route("/api/predict-chords", methods=["POST"])
def predict_uploaded_chords():
    """Suggest chords from fixed windows of an uploaded, extracted vocal melody."""
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id", "")
    audio_path = get_upload_path(upload_id) if isinstance(upload_id, str) else None
    if audio_path is None:
        return jsonify(error="The uploaded audio file was not found. Please upload it again."), 404
    musical_key = payload.get("musical_key")
    if not is_supported_musical_key(musical_key):
        return jsonify(error="Detect a supported major or minor key before generating chord suggestions."), 422
    try:
        melody = extract_melody(audio_path)
        windows = make_prediction_windows(melody.get("timeline", []))
        suggestions = []
        for window in windows:
            prediction = predict_chords(window["notes"], window["frequencies"],
                                        window["durations"], musical_key)
            suggestions.append({"start": round(window["start"], 3), "end": round(window["end"], 3),
                                "chord": prediction["chord"], "probability": prediction["probability"],
                                "alternatives": prediction["alternatives"]})
        return jsonify(musical_key=musical_key, predictions=suggestions,
                       window_seconds=1.8,
                       explanation=("Chord suggestions are generated by a machine-learning model trained on "
                                    "annotated musical data. Multiple chord progressions may harmonize with the same melody."))
    except (AudioProcessingError, KeyDetectionError) as error:
        return jsonify(error=str(error)), 422
    except ChordPredictionError as error:
        app.logger.warning("Chord prediction could not be completed: %s", error)
        return jsonify(error=str(error)), error.status_code
    except Exception:
        app.logger.exception("Unexpected chord-prediction error")
        return jsonify(error="Chord suggestions could not be generated for this recording."), 500

@app.route("/api/extract-lyrics", methods=["POST"])
def extract_uploaded_lyrics():
    """Transcribe actual lyrics and word timestamps from the existing secure upload."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}
    upload_id = payload.get("upload_id", "")
    audio_path = get_upload_path(upload_id) if isinstance(upload_id, str) else None
    if audio_path is None:
        return jsonify(error="The uploaded audio file was not found. Please upload it again."), 404
    try:
        result = transcribe_lyrics(audio_path, language=payload.get("language"))
        result = romanize_transcription(result)
        return jsonify(success=True, language=result["language"], transcript=result["transcript"],
                       words=result["words"], metadata=result["metadata"], message=result["message"],
                       original_transcript=result["original_transcript"])
    except LyricsTranscriptionError as error:
        app.logger.warning("Lyrics extraction could not be completed: %s", error)
        return jsonify(success=False, error=str(error)), error.status_code
    except Exception:
        app.logger.exception("Unexpected lyrics-transcription error")
        return jsonify(success=False, error="Lyrics could not be extracted from this recording."), 500

@app.route("/api/align-lyrics-chords", methods=["POST"])
def align_uploaded_lyrics_and_chords():
    """Align the existing transcription and model prediction responses by timestamps."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Provide lyric words and chord predictions to align."), 400
    words = payload.get("words")
    predictions = payload.get("predictions")
    if not isinstance(words, list) or not isinstance(predictions, list):
        return jsonify(error="Lyric words and chord predictions must be lists."), 400
    if len(words) > 10000 or len(predictions) > 2000:
        return jsonify(error="The alignment request contains too many items."), 413
    try:
        alignment = align_lyrics_and_chords(words, predictions)
        return jsonify(success=True, alignment=alignment)
    except LyricChordAlignmentError as error:
        app.logger.exception("Lyrics/chord alignment could not load trained class metadata")
        return jsonify(success=False, error=str(error)), 503
    except ValueError as error:
        return jsonify(success=False, error=str(error)), 400
    except Exception:
        app.logger.exception("Unexpected lyrics/chord alignment error")
        return jsonify(success=False, error="Lyrics and chord predictions could not be aligned."), 500

@app.route("/api/generate-chord-sheet", methods=["POST"])
def generate_uploaded_chord_sheet():
    """Run the complete melody, key, lyric, LSTM, and alignment pipeline for one upload."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(success=False, error="Provide a valid uploaded audio reference."), 400
    upload_id = payload.get("upload_id", "")
    audio_path = get_upload_path(upload_id) if isinstance(upload_id, str) else None
    if audio_path is None:
        return jsonify(success=False, error="The uploaded audio file was not found. Please upload it again."), 404

    try:
        melody = extract_melody(audio_path)
        supplied_key = payload.get("musical_key")
        if is_supported_musical_key(supplied_key):
            musical_key = supplied_key
            key_analysis = None
        else:
            key_analysis = detect_key(melody)
            musical_key = key_analysis.get("key")
        if not is_supported_musical_key(musical_key):
            raise KeyDetectionError("A supported musical key could not be detected from this recording.")

        # No language is passed here: faster-whisper detects it from this audio.
        lyrics = transcribe_lyrics(audio_path, language=None)
        detected_language = lyrics.get("language")
        transcription_metadata = lyrics.get("metadata") or {}
        language_probability = transcription_metadata.get("language_probability")
        if isinstance(language_probability, (int, float)):
            language_confidence = (
                "low" if language_probability < 0.5
                else "moderate" if language_probability < 0.8
                else "high"
            )
        else:
            language_confidence = "unavailable"
        if language_confidence == "low":
            app.logger.warning(
                "Whisper language detection has low confidence for current upload: language=%s probability=%s",
                detected_language, language_probability,
            )
        else:
            app.logger.info(
                "Whisper language detection for current upload: language=%s probability=%s confidence=%s",
                detected_language, language_probability, language_confidence,
            )
        lyrics = romanize_transcription(lyrics)
        words = lyrics.get("words", [])
        if not words:
            return jsonify(success=False, error="No timed lyrics were recognized. Try a clearer vocal recording."), 422

        windows = make_prediction_windows(melody.get("timeline", []))
        predictions = []
        for window in windows:
            prediction = predict_chords(window["notes"], window["frequencies"],
                                        window["durations"], musical_key)
            predictions.append({"start": round(window["start"], 3), "end": round(window["end"], 3),
                                "chord": prediction["chord"], "probability": prediction["probability"],
                                "alternatives": prediction["alternatives"]})
        if not predictions:
            return jsonify(success=False, error="The melody did not contain enough notes for chord prediction."), 422

        alignment = align_lyrics_and_chords(words, predictions)
        if not any(line.get("chords") for line in alignment.get("lines", [])):
            return jsonify(success=False, error="Chord predictions did not overlap the recognized lyric timestamps."), 422
        return jsonify(success=True, musical_key=musical_key, key_analysis=key_analysis,
                       detected_language=detected_language,
                       language_probability=language_probability,
                       language_confidence=language_confidence,
                       language_warning=("Whisper was uncertain about the language. Lyrics may be less accurate."
                                         if language_confidence == "low" else None),
                       alignment=alignment)
    except (AudioProcessingError, KeyDetectionError, LyricsTranscriptionError) as error:
        app.logger.warning("Chord sheet generation could not be completed: %s", error)
        status_code = error.status_code if isinstance(error, LyricsTranscriptionError) else 422
        return jsonify(success=False, error=str(error)), status_code
    except ChordPredictionError as error:
        app.logger.warning("Chord sheet prediction could not be completed: %s", error)
        return jsonify(success=False, error=str(error)), error.status_code
    except LyricChordAlignmentError as error:
        app.logger.exception("Chord sheet alignment could not load trained chord class metadata")
        return jsonify(success=False, error=str(error)), 503
    except Exception:
        app.logger.exception("Unexpected chord sheet generation error")
        return jsonify(success=False, error="The chord sheet could not be generated for this recording."), 500

@app.errorhandler(413)
def file_too_large(_error):
    app.logger.warning("Rejected request exceeding the %s-byte upload limit", MAX_UPLOAD_SIZE)
    return jsonify(error="File is too large. Please upload an audio file below 50 MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
